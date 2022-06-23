"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import datetime
import io
import os

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import WDVisualizer
import matplotlib.pyplot as plt


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.num_test = opt.num_test if opt.num_test > 0 else float("inf")

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    visualizer = WDVisualizer(opt)
    logger = visualizer.logger
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # traverse all epoch for the evaluation
    files_list = os.listdir(opt.checkpoints_dir + '/' + opt.name)
    epoches = []
    for file in files_list:
        if 'net_G' in file and 'latest' not in file:
            name = file.split('_')
            epoches.append(int(name[0]))
    epoches = sorted(set(epoches))

    fid_values = {}
    kid_values = {}
    for epoch in epoches:
        opt.epoch = epoch
        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)
        metrics_array = model.eval_metrics(epoch=epoch, num_test=opt.num_test) if opt.generate_image else model.eval_metrics_no(num_test=opt.num_test)
        send_stats = {}
        send_stats['epoch'] = epoch 
        for i in range(len(metrics_array)):
            metrics = metrics_array[i]
            kid_values[epoch] = kid_values[epoch] if epoch in kid_values else  []
            fid_values[epoch] = fid_values[epoch] if epoch in fid_values else []
            kid_values[epoch].append(metrics['KID'])
            fid_values[epoch].append(metrics['FID'])
            send_stats[f'KID_{i}'] = metrics['KID']
            send_stats[f'FID_{i}'] = metrics['FID']
        logger.send(send_stats, f"{now}_{opt.name}_metrics", True)

    print (fid_values)
    print (kid_values)
    for metric_name, values in zip(["FID", "KID"], [fid_values, kid_values]):
        x = []
        ys = []
        for key in sorted(values.keys()):
            x.append(key)
            stats = values[key]
            for i in range(len(stats)):
                if len(ys) <= i:
                    ys.append([])
                ys[i].append(stats[i])
        plt.figure()
        lines = []
        for i in range(len(ys)):
            y = ys[i]
            line, = plt.plot(x, y, label = f'{metric_name}_{i}')
            lines.append(line)
            for a, b in zip(x, y):
                plt.text(a, b, str(round(b, 3)))
        plt.legend(handles=lines)
        plt.xlabel('Epoch')
        plt.ylabel(f'{metric_name} on test set')
        plt.title(opt.name)
        img_buf = io.BytesIO()
        plt.savefig(img_buf)

        image_name = "%s_%s_%s.jpg" % (opt.name, metric_name, now)
        logger.sendBlobFile(img_buf, image_name, f"/eval_metrics/{image_name}", opt.name)
