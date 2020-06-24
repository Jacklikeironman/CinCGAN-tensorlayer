import os, time
import numpy as np

import tensorflow as tf
import tensorlayer as tl
from model import CinCGAN_d, CinCGAN_g
from utils import *
from config import config
from loss import TV_loss

batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
n_epoch_init = config.TRAIN.n_epoch_init
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))

def train():
    save_dir_gan = 'sample/{}_gan'.format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = 'checkpoint'
    tl.files.exists_or_mkdir(checkpoint_dir)
    log_dir = 'log'
    tl.files.exists_or_mkdir(log_dir)

    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    train_lr_bicubic_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_BICUBIC_img_path, regx='.*.png', printable=False))
    train_lr_unknown_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_UNKNOWN_img_path, regx='.*.png', printable=False))

    train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    train_lr_bicubic_imgs = tl.vis.read_images(train_lr_bicubic_img_list, path=config.TRAIN.lr_BICUBIC_img_path, n_threads=32)
    train_lr_unknown_imgs = tl.vis.read_images(train_lr_unknown_img_list, path=config.TRAIN.lr_UNKNOWN_img_path, n_threads=32)

    ####-----------------train--------------------#######
    x_image = tf.placeholder('float32', [batch_size, 96, 96, 3], name='lr_unknown_images')
    y_image = tf.placeholder('float32', [batch_size, 96, 96, 3], name='lr_bicubic_images')

    net_g1 = CinCGAN_g(x_image, strides=1, is_train=True, reuse=False, name='generator_1')   #unknown 2 bicubic
    net_g2 = CinCGAN_g(net_g1.outputs, strides=1, is_train=True, reuse=False, name='generator_2')  #bicubic 2 unknown


    net_d_real = CinCGAN_d(y_image, strides=1, is_train=True, reuse=False, name='discriminator_1')
    net_d_fake = CinCGAN_d(net_g1.outputs, strides=1, is_train=True, reuse=True, name='discriminator_1')

    ####-----------------test--------------------#######
    net_g_test = CinCGAN_g(x_image, strides=1, is_train=False, reuse=True, name='generator_1')

    ####-----------------define loss-------------#######
    d_lr_loss_real = tl.cost.mean_squared_error(net_d_real.outputs, tf.ones_like(net_d_real.outputs), is_mean=True)
    d_lr_loss_fake = tl.cost.mean_squared_error(net_d_fake.outputs, tf.zeros_like(net_d_fake.outputs), is_mean=True)
    d_lr_loss = d_lr_loss_fake + d_lr_loss_real

    lr_gan_loss = tl.cost.mean_squared_error(net_d_fake.outputs, tf.ones_like(net_d_fake.outputs), is_mean=True)
    lr_cyc_loss = tl.cost.mean_squared_error(net_g2.outputs, x_image, is_mean=True)
    lr_idt_loss = tl.cost.absolute_difference_error(CinCGAN_g(y_image, strides=1, is_train=True, reuse=True, name='generator_1').outputs, y_image, is_mean=True)
    lr_TV_loss = TV_loss(net_g1.outputs)

    g_lr_loss = lr_gan_loss + 10 * lr_cyc_loss + 5 * lr_idt_loss + 0.5 * lr_TV_loss

    tf.summary.scalar('d_lr_loss', d_lr_loss)
    tf.summary.scalar('g_lr_loss', g_lr_loss)
    merged_summary_op = tf.summary.merge_all()

    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if 'generator' in var.name]
    d_vars = [var for var in t_vars if 'discriminator' in var.name]

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)

    g_optim = tf.train.AdamOptimizer(lr_init, beta1=beta1).minimize(g_lr_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_init, beta1=beta1).minimize(d_lr_loss, var_list=d_vars)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)
    summary_writer.add_graph(sess.graph)
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g1_{}.npz'.format(tl.global_flag['mode']), network=net_g1)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g2_{}.npz'.format(tl.global_flag['mode']), network=net_g2)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/d1_{}.npz'.format(tl.global_flag['mode']), network=net_d_real)


    sample_bicubic_imgs = train_lr_bicubic_imgs[0:batch_size]
    sample_bicubic_imgs_96 = tl.prepro.threading_data(sample_bicubic_imgs, fn=crop_sub_img_fn_96, is_random=False)
    sample_unknown_imgs = train_lr_unknown_imgs[0:batch_size]
    sample_unknown_imgs_96 = tl.prepro.threading_data(sample_unknown_imgs, fn=crop_sub_img_fn_96, is_random=False)
    tl.vis.save_images(sample_bicubic_imgs_96, [ni, ni], save_dir_gan + '/train_sample_B_96.png')
    tl.vis.save_images(sample_unknown_imgs_96, [ni, ni], save_dir_gan + '/train_sample_U_96.png')

    counter = 0

    for epoch in range(0, n_epoch_init + 1):
        ## update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for CycleGAN_1)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for CycleGAN_1)" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0

        for idx in range(0, len(train_lr_bicubic_imgs), batch_size):
            step_time = time.time()
            x_imgs_96 = tl.prepro.threading_data(train_lr_unknown_imgs[idx:idx + batch_size], fn=crop_sub_img_fn_96, is_random=True)
            y_imgs_96 = tl.prepro.threading_data(train_lr_bicubic_imgs[idx:idx + batch_size], fn=crop_sub_img_fn_96, is_random=True)
            counter += 1

            errD, _ = sess.run([d_lr_loss, d_optim], {x_image:x_imgs_96, y_image:y_imgs_96})
            errG, errA, errC, errI, errTV, _ = sess.run([g_lr_loss, lr_gan_loss, lr_cyc_loss, lr_idt_loss, lr_TV_loss, g_optim],
                                                 {x_image:x_imgs_96, y_image:y_imgs_96})

            summary = sess.run(merged_summary_op, feed_dict={x_image:x_imgs_96, y_image:y_imgs_96})
            summary_writer.add_summary(summary, counter)

            print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (gan: %.6f clc: %.6f idt: %.6f TV: %.6f)" %
                  (epoch, n_epoch_init, n_iter, time.time() - step_time, errD, errG, errA, errC, errI, errTV))

            total_d_loss += errD
            total_g_loss += errG
            n_iter +=1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch,
                                                                                n_epoch_init,
                                                                                time.time() - epoch_time,
                                                                                total_d_loss / n_iter,
                                                                                total_g_loss / n_iter)

        print(log)
        if (epoch != 0) and (epoch % 10 == 0):
            out = sess.run(net_g_test.outputs, {x_image: sample_unknown_imgs_96})  #; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_gan + '/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(net_g1.all_params, name=checkpoint_dir + '/g1_{}.npz'.format(tl.global_flag['mode']), sess=sess)
            tl.files.save_npz(net_g2.all_params, name=checkpoint_dir + '/g2_{}.npz'.format(tl.global_flag['mode']), sess=sess)
            tl.files.save_npz(net_d_real.all_params, name=checkpoint_dir + '/d1_{}.npz'.format(tl.global_flag['mode']), sess=sess)

def evaluate():
    save_dir = 'sample/{}'.format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = 'checkpoint'

    valid_lr_bicubic_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_BICUBIC_img_path, regx='.*.png', printable=False))
    valid_lr_unknown_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_UNKNOWN_img_path, regx='.*.png', printable=False))

    valid_lr_bicubic_imgs = tl.vis.read_images(valid_lr_bicubic_img_list, path=config.VALID.lr_BICUBIC_img_path, n_threads=32)
    valid_lr_unknown_imgs = tl.vis.read_images(valid_lr_unknown_img_list, path=config.VALID.lr_UNKNOWN_img_path, n_threads=32)

    valid_lr_bicubic_imgs_96 = tl.prepro.threading_data(valid_lr_bicubic_imgs, fn=crop_sub_img_fn_96, is_random=False)
    valid_lr_unknown_imgs_96 = tl.prepro.threading_data(valid_lr_unknown_imgs, fn=crop_sub_img_fn_96, is_random=False)

    imid = 0
    valid_lr_bicubic_img_96 = valid_lr_bicubic_imgs_96[imid]
    valid_lr_unknown_img_96 = valid_lr_unknown_imgs_96[imid]

    # valid_lr_unknown_img_96 = (valid_lr_unknown_img_96 / 127.5) - 1
    x_images = tf.placeholder('float32', [1, None, None, 3], name='input_image')

    net_g1 = CinCGAN_g(x_images, strides=1, is_train=False, reuse=False)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g1_cincgan.npz', network=net_g1)

    start_time = time.time()
    out = sess.run(net_g1.outputs, {x_images:[valid_lr_unknown_img_96]})
    print("took: %4.4fs" % (time.time() - start_time))

    print("[*] save images")
    tl.vis.save_image(out[0], save_dir + '/valid_gen.png')
    tl.vis.save_image(valid_lr_unknown_img_96, save_dir + '/valid_lr_unknown.png')
    tl.vis.save_image(valid_lr_bicubic_img_96, save_dir + '/valid_lr_bicubic.png')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='cincgan', help='cincgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'cincgan':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")



