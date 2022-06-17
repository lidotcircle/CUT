import json
import time
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

    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    test_dataset = create_dataset(util.copyconf(opt, phase="test", batch_size=opt.val_batch_size))
    def sample_image():
        _, data = next(enumerate(test_dataset))
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference

    visualizer = WDVisualizer(opt)   # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        # visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        dataset.set_epoch(epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            batches_done = (epoch - 1) * dataset_size + (i + 1) * batch_size
            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                sample_image()
                visualizer.display_current_results(model.get_current_visuals(), batches_done, save_result)
                validity_stats = {}
                if hasattr(model, 'validation_loss_fake'):
                    validity_stats['validation_loss_fake'] = model.validation_loss_fake
                if hasattr(model, 'validation_loss_real'):
                    validity_stats['validation_loss_real'] = model.validation_loss_real
                if hasattr(model, 'augment_p'):
                    validity_stats['augment_p'] = model.augment_p
                if hasattr(model, 'ada_r_v'):
                    validity_stats['ada_r_v'] = model.ada_r_v
                if len(validity_stats) > 0:
                    visualizer.logger.send(validity_stats, "validity_stats", True)
                    print(json.dumps(validity_stats))

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                try:
                    losses['generator_param_norm'] = model.g_param_norm
                    losses['generator_grad_norm'] = model.g_grad_norm
                    losses['generator_param_norm_avg'] = model.g_param_norm_avg
                    losses['generator_grad_norm_avg'] = model.g_grad_norm_avg
                except:
                    pass
                visualizer.print_current_losses(batches_done, epoch_iter, losses, optimize_time, t_data)
                visualizer.plot_current_losses(batches_done, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch >= opt.metric_start_epoch and (epoch - opt.epoch_count) % opt.metric_eval_freq == 0:
            metrics_stats = model.eval_metrics(epoch=epoch)
            metrics_stats['epoch'] = epoch
            visualizer.logger.send(metrics_stats, "Metrics", True)

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
