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
import os
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from data.base_dataset import add_groundtruth
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

if __name__ == '__main__':

    opt = TrainOptions().parse()   # get training options
    writer = SummaryWriter(os.path.join('../Runs/', opt.name) + '/')
    # os.environ["CUDA_VISIBLE_DEVICES"] = '6'
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    # DataSet={dataset,dataset_size}
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    
    # epoch_count=1  niter=100   niter_decay=100   range(200)
    # all_epoch_num=200     data_number = 2963+3096
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        all_loss=[]

        for i,data in enumerate(dataset):  # inner loop within one epoch
            # print('no:%d  pathA=%s  pathB=%s' % (i,data['A_paths'],data['B_paths']))
            iter_start_time = time.time()   # timer for computation per iteration
            if total_iters % opt.print_freq == 0:     # print_freq=100  init_total_iters=0
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size # batch_size=1
            epoch_iter += opt.batch_size
            # FIXME:function is in CyclgGANModel class
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:       # display_freq=400    display images on visdom and save images to a HTML file
            # if total_iters % 50 == 0:
                save_result = total_iters % opt.update_html_freq == 0   # update_html_freq=1000
                if opt.use_gt > 0:
                    data = add_groundtruth(data, opt)  # add groundtruth into data
                model.compute_visuals(data)
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if epoch_iter % opt.print_freq == 0:         # print_freq=100   print training losses and save logging information to the disk
            # if epoch_iter % 30 == 0:
                losses = model.get_current_losses()
                # all_loss.append(losses)
                # n = all_loss['D_A']
                # m = list(map(list,zip(*all_loss)))
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:     # display_id=1
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # save_latest_freq=5000   cache our latest model every <save_latest_freq> iterations
            # if total_iters % 500 == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        writer.add_scalars('train_AtoB', {'D_A loss':losses['D_A'],
                                          'G_A loss':losses['G_A'],
                                          'cycle_A loss':losses['cycle_A']}, epoch)

        writer.add_scalars('train_BtoA', {'D_B loss':losses['D_B'],
                                          'G_B loss':losses['G_B'],
                                          'cycle_B loss':losses['cycle_B']}, epoch)

        if opt.lambda_identity > 0.0:
            writer.add_scalars('train_AtoB', {'idt_A loss':losses['idt_A']},epoch)
            writer.add_scalars('train_BtoA', {'idt_B loss':losses['idt_B']},epoch)

        if epoch % opt.save_epoch_freq == 0:              # save_epoch_freq=5   cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

    writer.close()