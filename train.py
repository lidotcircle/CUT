"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
import json
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import WDVisualizer
import util.util as util 


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options

    test_dataset = create_dataset(util.copyconf(opt, phase="test", batch_size=opt.val_batch_size))
    def sample_image():
        _, data = next(enumerate(test_dataset))
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference

    visualizer = WDVisualizer(opt)   # create a visualizer that display/save images and plots
    logger = visualizer.logger
    def info(msg: str):
        logger.info(msg)
        print(msg)
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        info(f"epoch {epoch} / ({opt.n_epochs} + {opt.n_epochs_decay})")

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)
                model.parallelize()
                model.print_networks(True)
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                sample_image()
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), total_iters, save_result)
                validity_stats_name = [
                    "val_loss_G", "val_loss_D_real", "val_loss_D_fake",
                    "val_loss_G_A", "val_loss_D_A_real", "val_loss_D_A_fake",
                    "val_loss_G_B", "val_loss_D_B_real", "val_loss_D_B_fake"
                ]
                validity_stats = {}
                for attr in validity_stats_name:
                    if hasattr(model, attr):
                        val = getattr(model, attr)
                        if isinstance(val, torch.Tensor):
                            val = val.mean().item()
                        validity_stats[attr] = val
                if len(validity_stats) > 0:
                    visualizer.logger.send(validity_stats, "validity_stats", True)
                    print(json.dumps(validity_stats))

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch >= opt.metric_start_epoch and (epoch - opt.epoch_count) % opt.metric_eval_freq == 0:
            metrics_stats_array = model.eval_metrics_no(epoch=epoch)
            send_stats = {}
            send_stats['epoch'] = epoch
            for i in range(len(metrics_stats_array)):
                metrics = metrics_stats_array[i]
                for k in metrics:
                    send_stats[f'{k}_{i}'] = metrics[k]
            visualizer.logger.send(send_stats, "Metrics", True)

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        info('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()  # update learning rates in the beginning of every epoch.
