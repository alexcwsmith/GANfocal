import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from GANfocal_DataLoader import Train_dataset, Test_dataset
import math
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import gaussian_filter
from GANfocal_Utils import smooth_gan_labels, subPixelConv3d
import os
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from keras.layers.convolutional import UpSampling3D
import numpy as np
import pandas as pd
from skimage import io


def lrelu1(x):
    return tf.maximum(x, 0.25 * x)


def lrelu2(x):
    return tf.maximum(x, 0.4 * x)


def discriminator(
    input_disc,
    kernel,
    img_zdepth,
    img_width,
    img_height,
    batch_size=1,
    reuse=None,
    is_train=True,
):
    w_init = tf.random_normal_initializer(stddev=0.02)
    with tf.compat.v1.variable_scope("SRGAN_d", reuse=reuse):
        #        tl.layers.set_name_reuse(reuse)
        input_disc.set_shape(
            [batch_size, img_zdepth, img_height, img_width, 1],
        )  # switched width height
        x = InputLayer(input_disc, name="in")
        # in Conv3dLayer the shape= argument defines sahpe of filters:
        # (filter_depth, filter_height, filter_width, in_channels, out_channels)
        # strides defines the sliding window for corresponding input dimensions
        x = Conv3dLayer(
            x,
            act=lrelu2,
            shape=(kernel, kernel, kernel, 1, 32),
            strides=(1, 1, 1, 1, 1),
            padding="SAME",
            W_init=w_init,
            name="conv1",
        )
        x = Conv3dLayer(
            x,
            shape=(kernel, kernel, kernel, 32, 32),
            strides=(1, 2, 2, 2, 1),
            padding="SAME",
            W_init=w_init,
            name="conv2",
        )

        x = BatchNormLayer(
            x, decay=0.9, is_train=is_train, name="BN1-conv2", act=lrelu2
        )

        x = Conv3dLayer(
            x,
            shape=(kernel, kernel, kernel, 32, 64),
            strides=(1, 1, 1, 1, 1),
            padding="SAME",
            W_init=w_init,
            name="conv3",
        )
        x = BatchNormLayer(x, is_train=is_train, name="BN1-conv3", act=lrelu2)
        x = Conv3dLayer(
            x,
            shape=(kernel, kernel, kernel, 64, 64),
            strides=(1, 2, 2, 2, 1),
            padding="SAME",
            W_init=w_init,
            name="conv4",
        )
        x = BatchNormLayer(x, is_train=is_train, name="BN1-conv4", act=lrelu2)

        x = Conv3dLayer(
            x,
            shape=(kernel, kernel, kernel, 64, 128),
            strides=(1, 1, 1, 1, 1),
            padding="SAME",
            W_init=w_init,
            name="conv5",
        )
        x = BatchNormLayer(x, is_train=is_train, name="BN1-conv5", act=lrelu2)
        x = Conv3dLayer(
            x,
            shape=(kernel, kernel, kernel, 128, 128),
            strides=(1, 2, 2, 2, 1),
            padding="SAME",
            W_init=w_init,
            name="conv6",
        )
        x = BatchNormLayer(x, is_train=is_train, name="BN1-conv6", act=lrelu2)

        x = Conv3dLayer(
            x,
            shape=(kernel, kernel, kernel, 128, 256),
            strides=(1, 1, 1, 1, 1),
            padding="SAME",
            W_init=w_init,
            name="conv7",
        )
        x = BatchNormLayer(x, is_train=is_train, name="BN1-conv7", act=lrelu2)
        x = Conv3dLayer(
            x,
            shape=(kernel, kernel, kernel, 256, 256),
            strides=(1, 2, 2, 2, 1),
            padding="SAME",
            W_init=w_init,
            name="conv8",
        )
        x = BatchNormLayer(x, is_train=is_train, name="BN1-conv8", act=lrelu2)

        x = FlattenLayer(x, name="flatten")
        x = DenseLayer(x, n_units=1024, act=lrelu2, name="dense1")
        x = DenseLayer(x, n_units=1, name="dense2")

        logits = x.outputs
        x.outputs = tf.nn.sigmoid(x.outputs, name="output")

        return x, logits


def generator(
    input_gen,
    kernel,
    feature_size,
    img_zdepth,
    img_height,
    img_width,
    subpixel_NN=True,
    nn=True,
    upscaling_factor=2,
    num_blocks=6,
    reuse=None,
    is_train=True,
):
    w_init = tf.random_normal_initializer(stddev=0.02)

    w_init_subpixel1 = np.random.normal(scale=0.02, size=(3, 3, 3, 64, feature_size))
    w_init_subpixel1 = zoom(w_init_subpixel1, (2, 2, 2, 1, 1), order=0)
    w_init_subpixel1_last = tf.constant_initializer(w_init_subpixel1)
    w_init_subpixel2 = np.random.normal(scale=0.02, size=(3, 3, 3, 64, 64))
    w_init_subpixel2 = zoom(w_init_subpixel2, (2, 2, 2, 1, 1), order=0)
    w_init_subpixel2_last = tf.constant_initializer(w_init_subpixel2)

    with tf.compat.v1.variable_scope("SRGAN_g", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        x = InputLayer(input_gen, name="in")
        x = Conv3dLayer(
            x,
            shape=(kernel, kernel, kernel, 1, feature_size),
            strides=(1, 1, 1, 1, 1),
            padding="SAME",
            W_init=w_init,
            name="conv1",
        )
        x = BatchNormLayer(x, act=lrelu1, is_train=is_train, name="BN-conv1")
        inputRB = x
        inputadd = x

        # residual blocks
        for i in range(num_blocks):
            x = Conv3dLayer(
                x,
                shape=(kernel, kernel, kernel, feature_size, feature_size),
                strides=(1, 1, 1, 1, 1),
                padding="SAME",
                W_init=w_init,
                name="conv1-rb/%s" % i,
            )
            x = BatchNormLayer(x, act=lrelu1, is_train=is_train, name="BN1-rb/%s" % i)
            x = Conv3dLayer(
                x,
                shape=(kernel, kernel, kernel, feature_size, feature_size),
                strides=(1, 1, 1, 1, 1),
                padding="SAME",
                W_init=w_init,
                name="conv2-rb/%s" % i,
            )
            x = BatchNormLayer(
                x,
                is_train=is_train,
                name="BN2-rb/%s" % i,
            )
            # short skip connection
            x = ElementwiseLayer([x, inputadd], tf.add, name="add-rb/%s" % i)
            inputadd = x

        # large skip connection
        x = Conv3dLayer(
            x,
            shape=(kernel, kernel, kernel, feature_size, feature_size),
            strides=(1, 1, 1, 1, 1),
            padding="SAME",
            W_init=w_init,
            name="conv2",
        )
        x = BatchNormLayer(x, is_train=is_train, name="BN-conv2")
        x = ElementwiseLayer([x, inputRB], tf.add, name="add-conv2")

        # ____________SUBPIXEL-NN______________#

        if subpixel_NN:
            # upscaling block 1
            if upscaling_factor == 4:
                img_zdepth_deconv = int(img_zdepth / 2)
                img_height_deconv = int(img_height / 2)
                img_width_deconv = int(img_width / 2)
            else:
                img_zdepth_deconv = img_zdepth
                img_height_deconv = img_height
                img_width_deconv = img_width

            x = DeConv3dLayer(
                x,
                shape=(kernel * 2, kernel * 2, kernel * 2, 64, feature_size),
                act=lrelu1,
                strides=(1, 2, 2, 2, 1),
                output_shape=(
                    tf.shape(input_gen)[0],
                    img_zdepth_deconv,
                    img_height_deconv,
                    img_width_deconv,
                    64,
                ),
                padding="SAME",
                W_init=w_init_subpixel1_last,
                name="conv1-ub-subpixelnn/1",
            )

            # upscaling block 2
            if upscaling_factor == 4:
                x = DeConv3dLayer(
                    x,
                    shape=(kernel * 2, kernel * 2, kernel * 2, 64, 64),
                    act=lrelu1,
                    strides=(1, 2, 2, 2, 1),
                    padding="SAME",
                    output_shape=(
                        tf.shape(input_gen)[0],
                        img_zdepth,
                        img_height,
                        img_width,
                        64,
                    ),
                    W_init=w_init_subpixel2_last,
                    name="conv1-ub-subpixelnn/2",
                )

            x = Conv3dLayer(
                x,
                shape=(kernel, kernel, kernel, 64, 1),
                strides=(1, 1, 1, 1, 1),
                padding="SAME",
                W_init=w_init,
                name="convlast-subpixelnn",
            )

        # ____________RC______________#

        elif nn:
            # upscaling block 1
            x = Conv3dLayer(
                x,
                shape=(kernel, kernel, kernel, feature_size, 64),
                act=lrelu1,
                strides=(
                    1,
                    1,
                    1,
                    1,
                    1,
                ),
                padding="SAME",
                W_init=w_init,
                name="conv1-ub/1",
            )
            x = UpSampling3D(name="UpSampling3D_1")(x.outputs)
            x = Conv3dLayer(
                InputLayer(x, name="in ub1 conv2"),
                shape=(kernel, kernel, kernel, 64, 64),
                act=lrelu1,
                strides=(
                    1,
                    1,
                    1,
                    1,
                    1,
                ),
                padding="SAME",
                W_init=w_init,
                name="conv2-ub/1",
            )

            # upscaling block 2
            if upscaling_factor == 4:
                x = Conv3dLayer(
                    x,
                    shape=(kernel, kernel, kernel, 64, 64),
                    act=lrelu1,
                    strides=(
                        1,
                        1,
                        1,
                        1,
                        1,
                    ),
                    padding="SAME",
                    W_init=w_init,
                    name="conv1-ub/2",
                )
                x = UpSampling3D(name="UpSampling3D_1")(x.outputs)
                x = Conv3dLayer(
                    InputLayer(x, name="in ub2 conv2"),
                    shape=(kernel, kernel, kernel, 64, 64),
                    act=lrelu1,
                    strides=(
                        1,
                        1,
                        1,
                        1,
                        1,
                    ),
                    padding="SAME",
                    W_init=w_init,
                    name="conv2-ub/2",
                )

            x = Conv3dLayer(
                x,
                shape=(kernel, kernel, kernel, 64, 1),
                strides=(
                    1,
                    1,
                    1,
                    1,
                    1,
                ),
                act=tf.nn.tanh,
                padding="SAME",
                W_init=w_init,
                name="convlast",
            )

        # ____________SUBPIXEL - BASELINE______________#

        else:

            if upscaling_factor == 4:
                steps_to_end = 2
            else:
                steps_to_end = 1

            # upscaling block 1
            x = Conv3dLayer(
                x,
                shape=(kernel, kernel, kernel, feature_size, 64),
                act=lrelu1,
                strides=(
                    1,
                    1,
                    1,
                    1,
                    1,
                ),
                padding="SAME",
                W_init=w_init,
                name="conv1-ub/1",
            )
            arguments = {
                "img_zdepth": img_zdepth,
                "img_height": img_height,
                "img_width": img_width,
                "stepsToEnd": steps_to_end,
                "n_out_channel": int(64 / 8),
            }
            x = LambdaLayer(x, fn=subPixelConv3d, fn_args=arguments, name="SubPixel1")

            # upscaling block 2
            if upscaling_factor == 4:
                x = Conv3dLayer(
                    x,
                    shape=(kernel, kernel, kernel, int((64) / 8), 64),
                    act=lrelu1,
                    strides=(
                        1,
                        1,
                        1,
                        1,
                        1,
                    ),
                    padding="SAME",
                    W_init=w_init,
                    name="conv1-ub/2",
                )
                arguments = {
                    "img_zdepth": img_zdepth,
                    "img_height": img_height,
                    "img_width": img_width,
                    "stepsToEnd": 1,
                    "n_out_channel": int(64 / 8),
                }
                x = LambdaLayer(
                    x, fn=subPixelConv3d, fn_args=arguments, name="SubPixel2"
                )

            x = Conv3dLayer(
                x,
                shape=(kernel, kernel, kernel, int(64 / 8), 1),
                strides=(
                    1,
                    1,
                    1,
                    1,
                    1,
                ),
                padding="SAME",
                W_init=w_init,
                name="convlast",
            )

        return x


def train(
    modelName,
    img_zdepth,
    img_width,
    img_height,
    kernel,
    img_path=os.getcwd(),
    result_dir=os.getcwd(),
    checkpoint_dir=os.getcwd(),
    upscaling_factor=2,
    residual_blocks=6,
    feature_size=64,
    subpixel_NN=True,
    nn=False,
    restore=None,
    batch_size=1,
    epochs=10,
    saveiters=9,
    train_fraction=0.8,
):
    
    traindata = Train_dataset(
        data_path=img_path,
        zdepth=img_zdepth,
        height=img_height,
        width=img_width,
        batch_size=batch_size,
        train_portion=train_fraction,
    )

    iterations_train = math.ceil((len(traindata.file_list)) / batch_size)
    with open(os.path.join(result_dir, "TrainingParameters.txt"), "a+") as f:
        f.write(
            modelName + "\n" 
            + "N images: " + str(len(traindata.file_list)) + "\n"
            + "Upscaling factor: " + str(upscaling_factor) + "\n"
            + "Residual blocks: " + str(residual_blocks) + "\n"
            + "Feature size: " + str(feature_size) + "\n"
            + "Kernel size: " + str(kernel) + "\n"
            + "Subpixel_NN: " + str(subpixel_NN) + "\n"
            + "NN: " + str(nn) + "\n"
        )
        f.close()
    # ##========================== DEFINE MODEL ============================##
    t_input_gen = tf.compat.v1.placeholder(
        "float32",
        (int(batch_size), img_zdepth / 2, img_height / 2, img_width / 2, 1),
        name="t_image_input_to_SRGAN_generator",
    )

    t_target_image = tf.compat.v1.placeholder(
        "float32",
        (int(batch_size), img_zdepth, img_height, img_width, 1),
        name="t_target_image",
    )


    net_gen = generator(
        input_gen=t_input_gen,
        kernel=kernel,
        num_blocks=residual_blocks,
        upscaling_factor=upscaling_factor,
        img_zdepth=traindata.zdepth,
        img_height=traindata.height,
        img_width=traindata.width,
        subpixel_NN=subpixel_NN,
        nn=nn,
        feature_size=feature_size,
        is_train=True,
        reuse=False,
    )
    net_d, disc_out_real = discriminator(
        input_disc=t_target_image,
        img_zdepth=traindata.zdepth,
        img_height=traindata.height,
        img_width=traindata.width,
        kernel=kernel,
        is_train=True,
        reuse=False,
    )
    _, disc_out_fake = discriminator(
        input_disc=net_gen.outputs,
        img_zdepth=traindata.zdepth,
        img_height=traindata.height,
        img_width=traindata.width,
        kernel=kernel,
        is_train=True,
        reuse=True,
    )

    # test
    gen_test = generator(
        t_input_gen,
        kernel,
        feature_size,
        img_zdepth,
        img_height,
        img_width,
        num_blocks=residual_blocks,
        upscaling_factor=upscaling_factor,
        subpixel_NN=subpixel_NN,
        nn=nn,
        is_train=True,
        reuse=True,
    )

    # ###========================== DEFINE TRAIN OPS ==========================###
    run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    if np.random.uniform() > 0.1:
        # give correct classifications
        y_gan_real = tf.ones_like(disc_out_real)
        y_gan_fake = tf.zeros_like(disc_out_real)
    else:
        # give wrong classifications (noisy labels)
        y_gan_real = tf.zeros_like(disc_out_real)
        y_gan_fake = tf.ones_like(disc_out_real)

    d_loss_real = tf.reduce_mean(
        tf.square(disc_out_real - smooth_gan_labels(y_gan_real)), name="d_loss_real"
    )
    d_loss_fake = tf.reduce_mean(
        tf.square(disc_out_fake - smooth_gan_labels(y_gan_fake)), name="d_loss_fake"
    )
    d_loss = d_loss_real + d_loss_fake

    mse_loss = tf.reduce_sum(
        tf.square(net_gen.outputs - t_target_image),
        axis=[0, 1, 2, 3, 4],
        name="g_loss_mse",
    )

    dx_real = t_target_image[:, :, :, 1:, :] - t_target_image[:, :, :, :-1, :]
    dy_real = t_target_image[:, :, 1:, :, :] - t_target_image[:, :, :-1, :, :]
    dz_real = t_target_image[:, 1:, :, :, :] - t_target_image[:, -1:, :, :, :]
    dx_fake = net_gen.outputs[:, :, :, 1:, :] - net_gen.outputs[:, :, :, :-1, :]
    dy_fake = net_gen.outputs[:, :, 1:, :, :] - net_gen.outputs[:, :, :-1, :, :]
    dz_fake = net_gen.outputs[:, 1:, :, :, :] - net_gen.outputs[:, :-1, :, :, :]

    gd_loss = (
        tf.reduce_sum(tf.square(tf.abs(dx_real) - tf.abs(dx_fake)))
        + tf.reduce_sum(tf.square(tf.abs(dy_real) - tf.abs(dy_fake)))
        + tf.reduce_sum(tf.square(tf.abs(dz_real) - tf.abs(dz_fake)))
    )
    # g_gan loss was 10e-2
    gen_lossrate = 20e-4
    g_gan_loss = gen_lossrate * tf.reduce_mean(
        tf.square(disc_out_fake - smooth_gan_labels(tf.ones_like(disc_out_real))),
        name="g_loss_gan",
    )

    g_loss = mse_loss + g_gan_loss + gd_loss

    g_vars = tl.layers.get_variables_with_name("SRGAN_g", True, True)
    d_vars = tl.layers.get_variables_with_name("SRGAN_d", True, True)

    with tf.compat.v1.variable_scope("learning_rate"):
        lr_v = tf.Variable(1e-4, trainable=False)
    global_step = tf.Variable(0, trainable=False)
    decay_rate = 0.1  # was 0,5
    decay_steps = 1000  # every 2 epochs (more or less)
    learning_rate = tf.compat.v1.train.inverse_time_decay(
        lr_v, global_step=global_step, decay_rate=decay_rate, decay_steps=decay_steps
    )

    # Optimizers
    g_optim = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(
        g_loss, var_list=g_vars
    )
    d_optim = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(
        d_loss, var_list=d_vars
    )

    gpu_options = tf.compat.v1.GPUOptions(
        per_process_gpu_memory_fraction=0.9, allow_growth=True
    )
    session = tf.compat.v1.Session(
        config=tf.compat.v1.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False
        )
    )
    tl.layers.initialize_global_variables(session)
    saver = tf.compat.v1.train.Saver()

    if restore is not None:
        saver.restore(session, tf.train.latest_checkpoint(restore))
        val_restore = int(
            tf.train.latest_checkpoint(restore).split("/")[-1].split("-")[-1]
        )
    #     val_restore = 0 * epochs
    else:
        val_restore = 0

    array_psnr = []
    array_ssim = []
    step = val_restore
    k = 0
    for j in range(val_restore, epochs + val_restore):
        for i in range(0, iterations_train):
            # ====================== LOAD DATA =========================== #
            hr_total = traindata.dataToTensor(i, blur=False)
            f, e = os.path.splitext(traindata.file_list[i])

            print("{}".format(i))
            hr = hr_total

            # NORMALIZING
            for t in range(0, hr.shape[0]):
                normfactor = (np.amax(hr[t])) / 2
                if normfactor != 0:
                    hr[t] = (hr[t] - normfactor) / normfactor
            # xt_total is now hr_total
            # xt is now hr
            # x_generator is now lr_generator

            lr_generator = gaussian_filter(hr, sigma=1.2)
            lr_generator = zoom(
                lr_generator,
                [
                    1,
                    (1 / upscaling_factor),
                    (1 / upscaling_factor),
                    (1 / upscaling_factor),
                    1,
                ],
                prefilter=False,
                order=0,
            )
            lrgenin = lr_generator
            # xgenin is now lrgenin

            # ========================= train SRGAN ========================= #
            # update D
            errd, _ = session.run(
                [d_loss, d_optim],
                {t_target_image: hr, t_input_gen: lrgenin},
                options=run_opts,
            )  # added options
            # update G
            errg, errmse, errgan, errgd, _ = session.run(
                [g_loss, mse_loss, g_gan_loss, gd_loss, g_optim],
                {t_input_gen: lrgenin, t_target_image: hr},
            )  # deleted t_input_mask: xm
            print(
                "Epoch [%2d/%2d] [%4d/%4d]: d_loss: %.8f g_loss: %.8f (mse: %.6f gdl: %.6f adv: %.6f)"
                % (
                    j, epochs + val_restore, i, iterations_train, errd, errg,
                    errmse, errgd, errgan,
                )
            )
            with open(
                os.path.join(result_dir, "losses_" + modelName + ".txt"), "a+"
            ) as log:
                log.write(
                    "Epoch [%2d/%2d] [%4d/%4d]: d_loss: %.8f g_loss: %.8f (mse: %.6f gdl: %.6f adv: %.6f)"
                % (
                    j, epochs + val_restore, i, iterations_train, errd, errg,
                    errmse, errgd, errgan,
                )
                    + "\n"
                )
                log.close()
            k = k + 1
            # ========================= evaluate & save model ========================= #

            #        if k == 1 and i % 20 == 0:
            #            if j - val_restore == 0:
            hr_true_img = hr[0]
            if normfactor != 0:
                hr_true_img = (hr_true_img + 1) * normfactor  # denormalize
            #       img_true = nib.Nifti1Image(x_true_img, np.eye(4))
            if step % saveiters == 0:
                io.imsave(
                    os.path.join(
                        result_dir, str(f) + "_Epoch" + str(step) + "_hr_true.tif"
                    ),
                    hr_true_img,
                )
            lr_gen_img = lrgenin[0]
            if normfactor != 0:
                lr_gen_img = (lr_gen_img + 1) * normfactor  # denormalize
            #     img_gen = nib.Nifti1Image(x_gen_img, np.eye(4))
            if step % saveiters == 0:
                io.imsave(
                    os.path.join(
                        result_dir, str(f) + "_Epoch" + str(step) + "_lr_gen.tif"
                    ),
                    lr_gen_img,
                )

            lr_pred = session.run(gen_test.outputs, {t_input_gen: lrgenin})
            lr_pred_img = lr_pred[0]
            if normfactor != 0:
                lr_pred_img = (lr_pred_img + 1) * normfactor  # denormalize
            #   lr_img_pred = nib.Nifti1Image(x_pred_img, np.eye(4))
            if step % saveiters == 0:
                io.imsave(
                    os.path.join(
                        result_dir, str(f) + "_Epoch" + str(step) + "_lr_pred.tif"
                    ),
                    lr_pred_img,
                )

            max_gen = np.amax(lr_pred_img)
            max_real = np.amax(hr_true_img)
            if max_gen > max_real:
                val_max = max_gen
            else:
                val_max = max_real
            min_gen = np.amin(lr_pred_img)
            min_real = np.amin(hr_true_img)
            if min_gen < min_real:
                val_min = min_gen
            else:
                val_min = min_real
            val_psnr = psnr(hr_true_img, lr_pred_img, data_range=val_max - val_min)
            val_ssim = ssim(
                hr_true_img,
                lr_pred_img,
                data_range=val_max - val_min,
                multichannel=True,
            )
            print("val_psnr: " + str(val_psnr))
            print("val_ssim: " + str(val_ssim))

            array_psnr.append(val_psnr)
            array_ssim.append(val_ssim)
        saver.save(sess=session, save_path=checkpoint_dir, global_step=step)
        print("Saved step: [%2d]" % step)
        step = step + 1
    df = pd.DataFrame(data=[array_psnr, array_ssim]).T
    df.columns = ["PSNR", "SSIM"]
    df.to_csv(os.path.join(result_dir, "Train_PSNR_SSIM_results.csv"))
    with open(os.path.join(result_dir, "TrainingParameters.txt"), "a+") as f:
        f.write(
            "Generator Loss Rate: " + str(g_gan_loss) + "\n"
            + "Completed Epochs: " + str(step) + "\n"
            + "Decay rate: " + str(decay_rate) + "\n"
            + "Decay steps: " + str(decay_steps) + "\n"
        )
        f.close()
    return (array_psnr, array_ssim)


def evaluate(
    modelName,
    kernel,
    img_zdepth,
    img_height,
    img_width,
    img_path,
    checkpoint_dir,
    result_dir,
    batch_size=1,
    feature_size=64,
    upscaling_factor=2,
    residual_blocks=6,
    subpixel_NN=True,
    nn=False,
):
    testdata = Test_dataset(
        data_path=img_path,
        zdepth=img_zdepth,
        height=img_height,
        width=img_width,
        batch_size=batch_size,
        test_portion=0.2,
    )
    iterations = math.ceil((len(testdata.file_list)) / batch_size)

    print(len(testdata.file_list))
    print(iterations)
    totalpsnr = 0
    totalssim = 0
    array_psnr = np.empty(iterations)
    array_ssim = np.empty(iterations)

    # define model
    t_input_gen = tf.compat.v1.placeholder(
        "float32",
        [int(batch_size), img_zdepth / 2, img_height / 2, img_width / 2, 1],
        name="t_image_input_to_SRGAN_generator",
    )

    srgan_network = generator(
        input_gen=t_input_gen,
        kernel=kernel,
        num_blocks=residual_blocks,
        upscaling_factor=upscaling_factor,
        img_zdepth=testdata.zdepth,
        img_height=testdata.height,
        img_width=testdata.width,
        subpixel_NN=subpixel_NN,
        nn=nn,
        feature_size=feature_size,
        is_train=False,
        reuse=False,
    )

    # restore g
    sess = tf.Session(
        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    )

    saver = tf.train.Saver(
        tf.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="SRGAN_g")
    )
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

    for i in range(0, iterations):
        # extract volumes
        hr_total = testdata.dataToTensor(i)
        f, e = os.path.splitext(testdata.file_list[i])
        #    xt_mask = traindata.mask(654 + i)
        #    normfactor = (np.amax(hr_total[0])) / 2
        #    lr_generator = ((hr_total[0] - normfactor) / normfactor) #MIGHT NEED THIS
        res = 1 / upscaling_factor
        #    lr_generator = lr_generator[:, :, :, np.newaxis]
        #    lr_generator = gaussian_filter(lr_generator, sigma=1)
        lr_generator = gaussian_filter(hr_total[0], sigma=1.2)  # removed hr_total[0]
        lr_generator = zoom(lr_generator, [res, res, res, 1], prefilter=False, order=0)
        sr_generated = sess.run(
            srgan_network.outputs, {t_input_gen: lr_generator[np.newaxis, :]}
        )
        #    sr_generated = ((sr_generated + 1) * normfactor)
        volume_real = hr_total[0]
        #     volume_real = volume_real[:, :, :, np.newaxis]
        volume_generated = sr_generated[0]
        #     volume_mask = aggregate(xt_mask)
        # compute metrics
        max_gen = np.amax(volume_generated)
        max_real = np.amax(volume_real)
        if max_gen > max_real:
            val_max = max_gen
        else:
            val_max = max_real
        min_gen = np.amin(volume_generated)
        min_real = np.amin(volume_real)
        if min_gen < min_real:
            val_min = min_gen
        else:
            val_min = min_real
        val_psnr = psnr(volume_real, volume_generated, data_range=val_max - val_min)
        array_psnr[i] = val_psnr

        totalpsnr += val_psnr
        val_ssim = ssim(
            volume_real,
            volume_generated,
            data_range=val_max - val_min,
            multichannel=True,
        )
        array_ssim[i] = val_ssim
        totalssim += val_ssim
        print(val_psnr)
        print(val_ssim)
        # save volumes
        filename_gen = os.path.join(
            result_dir, str(f) + "_" + str(i) + "_" + str(modelName) + "gen.tif"
        )
        img_volume_gen = io.imsave(filename_gen, volume_generated)
        filename_real = os.path.join(
            result_dir, str(f) + "_" + str(i) + "_" + str(modelName) + "real.tif"
        )
        img_volume_real = io.imsave(filename_real, volume_real)

    with open(
        os.path.join(result_dir, modelName + "_EvaluationResults.txt"), "a+"
    ) as log:
        log.write(
            ("{}{}".format("PSNR: ", array_psnr))
            + "\n"
            + ("{}{}".format("SSIM: ", array_ssim))
            + "\n"
            + ("{}{}".format("Mean PSNR: ", array_psnr.mean()))
            + "\n"
            + ("{}{}".format("Mean SSIM: ", array_ssim.mean()))
            + "\n"
            + ("{}{}".format("Variance PSNR: ", array_psnr.var()))
            + "\n"
            + ("{}{}".format("Variance SSIM: ", array_ssim.var()))
            + "\n"
            + ("{}{}".format("Max PSNR: ", array_psnr.max()))
            + "\n"
            + ("{}{}".format("Min PSNR: ", array_psnr.min()))
            + "\n"
            + ("{}{}".format("Max SSIM: ", array_ssim.max()))
            + "\n"
            + ("{}{}".format("Min SSIM: ", array_ssim.min()))
            + "\n"
            + ("{}{}".format("Median PSNR: ", np.median(array_psnr)))
            + "\n"
            + ("{}{}".format("Median SSIM: ", np.median(array_ssim)))
            + "\n"
        )
        log.close()

    print("{}{}".format("PSNR: ", array_psnr))
    print("{}{}".format("SSIM: ", array_ssim))
    print("{}{}".format("Mean PSNR: ", array_psnr.mean()))
    print("{}{}".format("Mean SSIM: ", array_ssim.mean()))
    print("{}{}".format("Variance PSNR: ", array_psnr.var()))
    print("{}{}".format("Variance SSIM: ", array_ssim.var()))
    print("{}{}".format("Max PSNR: ", array_psnr.max()))
    print("{}{}".format("Min PSNR: ", array_psnr.min()))
    print("{}{}".format("Max SSIM: ", array_ssim.max()))
    print("{}{}".format("Min SSIM: ", array_ssim.min()))
    print("{}{}".format("Median PSNR: ", np.median(array_psnr)))
    print("{}{}".format("Median SSIM: ", np.median(array_ssim)))


"""
                lr_generator = gaussian_filter(hr, sigma=1.2)
                lr_generator = zoom(lr_generator, [1, (1 / upscaling_factor), (1 / upscaling_factor),
                                                 (1 / upscaling_factor), 1], prefilter=False, order=0)
                lrgenin = lr_generator

"""
