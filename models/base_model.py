import os
from typing import Any, Callable, List, Tuple
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod

from data import CustomDatasetDataLoader, create_dataset
from . import networks
from metrics.all_score import calculate_scores_given_paths, calculate_scores_given_iter
from util.util import copyconf, tensor2im, save_image


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    def dict_grad_hook_factory(add_func=lambda x: x):
        saved_dict = dict()

        def hook_gen(name):
            def grad_hook(grad):
                saved_vals = add_func(grad)
                saved_dict[name] = saved_vals
            return grad_hook
        return hook_gen, saved_dict

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = opt.epoch
            self.load_networks(load_suffix)

        self.print_networks(opt.verbose)

    def parallelize(self):
        for name in self.model_names:
            if isinstance(name, str) and len(self.opt.gpu_ids) > 0:
                net = getattr(self, 'net' + name)
                setattr(self, 'net' + name, torch.nn.DataParallel(net, self.opt.gpu_ids))

    def data_dependent_initialize(self, data):
        pass

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                if self.opt.isTrain and self.opt.pretrained_name is not None:
                    load_dir = os.path.join(self.opt.checkpoints_dir, self.opt.pretrained_name)
                else:
                    load_dir = self.save_dir

                load_path = os.path.join(load_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                #    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def generate_visuals_for_evaluation(self, data, mode):
        return {}

    def translate_images(
        self,
        dataset: CustomDatasetDataLoader,
        action: Tuple[Callable[[torch.Tensor,torch.Tensor,List[str]],Any],str],
        AtoB: bool = None, num_test: int = 50
        ):
        AtoB = AtoB if AtoB is not None else self.opt.direction == 'AtoB'
        fake_A_image_path: str = None
        real_A_image_path: str = None
        fake_B_image_path: str = None
        real_B_image_path: str = None
        if isinstance(action, str):
            action_path = action
            fake_B_image_path = os.path.join(action_path, f"fake_{'B' if AtoB else 'A'}")
            real_B_image_path = os.path.join(action_path, f"real_{'B' if AtoB else 'A'}")
            os.makedirs(fake_B_image_path, exist_ok=True)
            os.makedirs(real_B_image_path, exist_ok=True)
            setuped: bool = False
            def setup_A_path():
                nonlocal setuped, fake_A_image_path, real_A_image_path
                if setuped:
                    return
                fake_A_image_path = os.path.join(action_path, f"fake_{'A' if AtoB else 'B'}")
                real_A_image_path = os.path.join(action_path, f"real_{'A' if AtoB else 'B'}")
                os.makedirs(fake_A_image_path, exist_ok=True)
                os.makedirs(real_A_image_path, exist_ok=True)
                setuped = True

            def save_action(fake_B, real_B, fake_A, real_A, name):
                assert fake_B.size(0) == real_B.size(0) == len(name)
                if fake_A is not None and real_A is not None:
                    setup_A_path()
                    assert fake_A.size(0) == real_A.size(0) == len(name)

                for i, img_path in enumerate(name):
                    bname = os.path.basename(img_path)
                    f, r = fake_B[i], real_B[i]
                    fp, rp = os.path.join(fake_B_image_path, bname), os.path.join(real_B_image_path, bname)
                    save_image(tensor2im(f.unsqueeze(0)), fp)
                    save_image(tensor2im(r.unsqueeze(0)), rp)

                    if fake_A is not None and real_A is not None:
                        f, r = fake_A[i], real_A[i]
                        fp, rp = os.path.join(fake_A_image_path, bname), os.path.join(real_A_image_path, bname)
                        save_image(tensor2im(f.unsqueeze(0)), fp)
                        save_image(tensor2im(r.unsqueeze(0)), rp)

            action = save_action

        for fb, rb, fa, ra, p in self.translate_images_iter(dataset, num_test):
            action(fb, rb, fa, ra, p)
        
        if fake_B_image_path is not None:
            return real_B_image_path, fake_B_image_path, real_A_image_path, fake_A_image_path

    def translate_images_iter(
        self,
        dataset: CustomDatasetDataLoader,
        num_test: int = 50
    ):
        for i, data in enumerate(dataset):
            if i >= num_test:  # only apply our model to opt.num_test images.
                break

            self.set_input(data)  # unpack data from data loader
            self.test()           # run inference
            visuals = self.get_current_visuals()  # get image results
            img_path: List[str] = self.get_image_paths()     # get image paths
            fake_B_images: torch.Tensor = visuals['fake_B']
            real_B_images: torch.Tensor = visuals['real_B']
            fake_A_images: torch.Tensor = None
            real_A_images: torch.Tensor = None
            if 'real_A' in visuals and 'fake_A' in visuals:
                fake_A_images = visuals['fake_A']
                real_A_images = visuals['real_A']
            yield fake_B_images, real_B_images, fake_A_images, real_A_images, img_path

    def translate_test_images(self, epoch = 0, num_test=50):
        result_dir = os.path.abspath(os.path.join(".", "results", self.opt.name, f"test_{epoch}"))
        test_dataset = create_dataset(copyconf(self.opt, phase="test", batch_size=self.opt.val_batch_size))
        return self.translate_images(test_dataset, result_dir, num_test=num_test)

    def eval_metrics(self, epoch = 0, num_test=50) -> List[dict]:
        real_B_dir, fake_B_dir, real_A_dir, fake_A_dir = self.translate_test_images(epoch, num_test=num_test)
        if fake_B_dir is None:
            return {}

        ans = []
        result = calculate_scores_given_paths(
                                [fake_B_dir, real_B_dir], device=self.device, batch_size=50, dims=2048,
                                use_fid_inception=True, torch_svd=self.opt.torch_svd)
        result = result[0]
        _, kid, fid = result
        kid_m, kid_std = kid
        db = {}
        db['FID'] = fid
        db['KID'] = kid_m
        db['KID_std'] = kid_std
        ans.append(db)

        if real_A_dir and fake_A_dir:
            result = calculate_scores_given_paths(
                                    [fake_A_dir, real_A_dir], device=self.device, batch_size=50, dims=2048,
                                    use_fid_inception=True, torch_svd=self.opt.torch_svd)
            result = result[0]
            _, kid, fid = result
            kid_m, kid_std = kid
            da = {}
            da['FID'] = fid
            da['KID'] = kid_m
            da['KID_std'] = kid_std
            ans.append(da)

        return ans

    def eval_metrics_no(self, num_test=50) -> List[dict]:
        test_dataset = create_dataset(copyconf(self.opt, phase="test", batch_size=self.opt.val_batch_size))
        images = self.translate_images_iter(test_dataset, num_test=num_test)
        stats = calculate_scores_given_iter(
                                map(lambda b: [b[0], b[1], b[2], b[3]], images), self.device, dims=2048,
                                use_fid_inception=True, torch_svd=self.opt.torch_svd)

        ans = []
        for fid, kid_m, kid_std in stats:
            d = {}
            d['FID'] = fid
            d['KID'] = kid_m
            d['KID_std'] = kid_std
            ans.append(d)
        return ans
