import math
import numpy as np
import torch
import torch.nn.functional as F
from .augment  import AugmentPipe
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from .utils import computeModelParametersNorm1, computeModelGradientsNorm1


class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        default_lambda_NCE = 0.0
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_GAN2', type=float, default=0.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_SIM', type=float, default=1.0, help='weight for similarity loss')
        parser.add_argument('--lambda_IDT', type=float, default=1.0, help='weight for identity loss')
        parser.add_argument('--lambda_NCE', type=float, default=default_lambda_NCE, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--ada', type=bool, default=True, help='adaptive distrcrimator augmentation')
        parser.add_argument('--ada_target', type=float, default=0.6, help='E[D_train]')
        parser.add_argument('--ada_interval', type=int, default=20, help='ADA interval')
        parser.add_argument('--ada_speed', type=int, default=500, help='ADA speed')
        parser.add_argument('--sim_augment_p', type=float, default=1, help='similarity module image augmentation probability')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,3,6,9,12', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if default_lambda_NCE > 0:
            if opt.CUT_mode.lower() == "cut":
                parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
            elif opt.CUT_mode.lower() == "fastcut":
                parser.set_defaults(
                    nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                    n_epochs=150, n_epochs_decay=50
                )
            else:
                raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'D_gp', 'G', "SIM", "S", "S_pos", "S_neg", "S_aug", "S_GP", 'idt']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        opt.ada = opt.ada and opt.gan_mode == 'wgangp'

        self.loss_G_GAN = 0
        self.loss_D_real = 0
        self.loss_D_fake = 0
        self.loss_D_gp = 0
        self.loss_G = 0
        self.loss_NEC = 0
        self.loss_SIM = 0
        self.loss_S = 0
        self.loss_S_pos = 0
        self.loss_S_neg = 0
        self.loss_S_aug = 0
        self.loss_S_GP = 0
        self.loss_idt = 0

        if self.isTrain:
            if opt.nce_idt:
                self.loss_names += ['NCE_Y']
            if opt.nce_idt or opt.lambda_SIM > 0:
                self.visual_names += ['idt_B']

        self.adaptive_scale = 0
        self.sim_latest_n = 10
        self.sim_latest_n_histories = []

        self.sampling_images = False
        if self.isTrain:
            self.model_names = ['G', 'F', 'D', 'D2', 'S']
        else:  # during test time, only load G
            self.model_names = ['G']

        self.augment_p = 0
        self.EDTRAIN = 0
        if opt.ada:
            self.augment_pipe_dis = AugmentPipe(
                rotate=1,
                brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1,
                imgfilter=1,
                noise=1
            ).train().requires_grad_(False).to(self.device)
            self.augment_pipe_dis.p.copy_(torch.as_tensor(self.augment_p))

        self.augment_pipe_sim = AugmentPipe(
            brightness=0.7, contrast=0.7, lumaflip=0.7, hue=0.7, saturation=0.7,
            cutout=1, cutout_size=0.25).train().requires_grad_(False).to(self.device)
        self.augment_pipe_sim.p.copy_(torch.as_tensor(opt.sim_augment_p))

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netS = networks.define_S(netS='patch', gpu_ids=self.gpu_ids)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netD2 = networks.define_D(0, 0, 'mlp', gpu_ids=self.gpu_ids)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionSelf = torch.nn.L1Loss().to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_S = torch.optim.Adam(self.netS.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_D2)
            self.optimizers.append(self.optimizer_S)

    def report_real_validity(self, validity):
        if not self.opt.ada:
            return

        assert self.opt.gan_mode == 'wgangp'
        if not hasattr(self, "validity_histories"):
            self.validity_histories = []
        
        validity = torch.mean(validity.detach().cpu().view(validity.size(0), -1), dim=1)
        self.validity_histories.append(validity)
        if len(self.validity_histories) >= self.opt.ada_interval:
            val = torch.cat(self.validity_histories, dim=0)
            n_positive_items = torch.where(val > 0, torch.ones_like(val), torch.zeros_like(val)).sum().item()
            EDT = n_positive_items / val.size(0)
            self.EDTRAIN = EDT
            adjust = np.sign(EDT - self.opt.ada_target) * (self.opt.batch_size * self.opt.ada_interval) / (self.opt.ada_speed * 1000)
            self.augment_p = min(max(self.augment_pipe_dis.p.item() + adjust, 0), 1)
            self.augment_pipe_dis.p.copy_(torch.as_tensor(self.augment_p))
            self.validity_histories = []

    def update_sim_scale(self, pos: float, neg: float, gradients_norm: float):
        self.sim_latest_n_histories.append((pos, neg, gradients_norm))
        if len(self.sim_latest_n_histories) < self.sim_latest_n:
            return

        sum = 0
        gradients_sum = 0
        for p, n, g in self.sim_latest_n_histories:
            sum -= (p + n)
            gradients_sum += g
        sum /= self.sim_latest_n
        gradients_sum /= self.sim_latest_n
        scale_div = max(gradients_sum * math.log(1 + max(sum, 0)), 1)
        self.adaptive_scale = 1 / max(1, scale_div)
        self.sim_latest_n_histories = []

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update D2
        if self.opt.lambda_GAN2 > 0:
            self.set_requires_grad(self.netD2, True)
            self.optimizer_D2.zero_grad()
            self.loss_D2 = self.compute_D2_loss()
            self.loss_D2.backward()
            self.optimizer_D2.step()
        else:
            self.loss_D2_real = 0
            self.loss_D2_fake = 0
            self.loss_D2 = 0

        # update S
        if self.opt.lambda_SIM > 0:
            self.set_requires_grad(self.netS, True)
            self.optimizer_S.zero_grad()
            self.loss_S = self.compute_S_loss()
            self.loss_S.backward()
            self.optimizer_S.step()
        else:
            self.loss_S_pos = 0
            self.loss_S_neg = 0
            self.loss_S = 0

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample' and self.opt.lambda_NCE > 0.0:
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()

        self.gparam, self.gparamn = computeModelParametersNorm1(self.netG)
        self.ggrad, nx = computeModelGradientsNorm1(self.netG)
        assert nx == self.gparamn
        self.gparam = self.gparam.item()
        self.ggrad = self.ggrad.item()
        self.gparam_avg = self.gparam / self.gparamn
        self.ggrad_avg = self.ggrad / self.gparamn

        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample' and self.opt.lambda_NCE > 0.0:
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if (self.opt.nce_idt or self.opt.lambda_IDT > 0 or self.sampling_images)and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        if hasattr(self, 'idt_B'):
            del self.idt_B

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.real.size(0) > self.real_A.size(0):
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(self.augment_pipe_dis(fake) if self.opt.ada else fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.augment_pipe_dis(self.real_B) if self.opt.ada else self.real_B)
        self.report_real_validity(-self.pred_real)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        if self.opt.gan_mode == 'wgangp':
            self.loss_D_gp, gradients = networks.cal_gradient_penalty(self.netD, self.real_B, fake, self.device, lambda_gp=1)
            self.dis_grad_norm = torch.norm(gradients).item()
            self.loss_D = self.loss_D + self.loss_D_gp * 10
        return self.loss_D

    def compute_D2_loss(self):
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD2(fake)
        self.loss_D2_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        pred_real = self.netD2(self.real)
        self.loss_D2_real = self.criterionGAN(pred_real, True).mean()

        # combine loss and calculate gradients
        self.loss_D2 = (self.loss_D2_fake + self.loss_D2_real) * 0.5
        return self.loss_D2

    def getNoSelfIdx(self, max, ma):
        ma = min(max - 1, ma)
        ans = []
        for i in range(max):
            n = []
            for j in range(max):
                if i == j:
                    continue
                n.append(j)
            ans.append(np.random.choice(n, ma))
        return ans

    def compute_gradient_penalty(self, S, pos, neg):
        device = pos.get_device()
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((pos.size(0), 1, 1, 1))).to(device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * pos + ((1 - alpha) * neg)).requires_grad_(True)
        d_interpolates = S(interpolates)
        vn = torch.autograd.Variable(torch.Tensor(pos.shape[0], 1).to(device).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=vn,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradients_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean()
        return gradient_penalty, gradients_norm
            
    def compute_S_loss(self):
        if hasattr(self, "prev_fake"):
            neg = torch.cat([self.real, self.prev_fake], dim=1)
        else:
            neg = torch.cat([self.real, torch.flip(self.real, [3])], dim=1)

        fake = torch.cat([self.fake_B.detach(), self.idt_B.detach()], dim=0) if hasattr(self, "idt_B") else self.fake_B.detach()
        pos = torch.cat([self.real, fake], dim=1)
        pos_similarity = self.netS(pos)
        self.loss_S_pos = torch.abs(pos_similarity - 1).mean()
        neg_similarity = self.netS(neg)
        self.loss_S_neg = torch.abs(neg_similarity).mean()
        self.prev_fake = fake

        aug_real = self.augment_pipe_sim(self.real)
        aug_sample = torch.cat([aug_real, fake], dim=1)
        aug_fake = self.augment_pipe_sim(fake)
        aug_2 = torch.cat([self.real, aug_fake], dim=1)
        aug_sample = torch.cat([aug_sample, aug_2], dim=0)
        aug_similarity = self.netS(aug_sample)

        self.loss_G_GP = 0
        # self.loss_S_GP, gradients_norm = self.compute_gradient_penalty(self.netS, pos, neg)
        # self.update_sim_scale(self.loss_S_pos.item(), self.loss_S_neg.item(), 0)
        self.adaptive_scale = 1
        alpha = 0.85
        self.loss_S_aug = torch.abs(aug_similarity - alpha).mean()
        self.pos_similarity = pos_similarity.mean().item()
        self.neg_similarity = neg_similarity.mean().item()
        self.aug_similarity = aug_similarity.mean().item()
        return self.loss_S_pos + self.loss_S_neg + self.loss_S_aug

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B

        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(self.augment_pipe_dis(fake) if self.opt.ada else fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_SIM > 0.0:
            fcat = torch.cat([self.fake_B, self.idt_B], dim=0) if hasattr(self, 'idt_B') else self.fake_B
            cat_real_fake = torch.cat([self.real, fcat], dim=1)
            self.loss_SIM = torch.abs(self.netS(cat_real_fake) - 1).mean() * self.opt.lambda_SIM
        else:
            self.loss_SIM = 0

        if self.opt.lambda_IDT > 0.0:
            self.loss_idt = self.criterionSelf(self.idt_B, self.real_B) * self.opt.lambda_IDT

        if self.opt.lambda_GAN2 > 0.0:
            pred_fake2 = self.netD2(fake)
            self.loss_G_GAN2 = self.criterionGAN(pred_fake2, True).mean() * self.opt.lambda_GAN2
        else:
            self.loss_G_GAN2 = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_idt + self.loss_G_GAN + self.loss_G_GAN2 + loss_NCE_both + self.loss_SIM * self.adaptive_scale
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
