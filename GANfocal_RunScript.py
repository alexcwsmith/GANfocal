#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 09:47:55 2020

@author: smith
"""

import os
os.chdir("/d2/studies/ImageGAN/GANfocal/")
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import GANfocal_Model_NoBatchNormLayers as gfm

modelName = "March26_2021_DiscrimGAP_NoBatchNormDG"

img_path = "/d2/studies/ImageGAN/GANfocal/Data/March2021_TrainingData"
result_dir = "/d2/studies/ImageGAN/GANfocal/" + modelName + "/"
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
checkpoint_dir = "/d2/studies/ImageGAN/GANfocal/checkpoints/" + modelName + "/"
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
residual_blocks = 6
upscaling_factor = 2
subpixel_NN = True
evaluate = False
nn = False
batch_size = 1
restore = None

img_zdepth = 50
img_height = 256
img_width = 256
batch_size = 1
kernel = 3
feature_size = 64
epochs = 20
saveiters = 1
train_fraction = 0.8
normalize=False

psnr, ssim = gfm.train(modelName, img_zdepth, img_width, img_height, kernel=kernel, img_path=img_path, 
                       result_dir=result_dir, checkpoint_dir=checkpoint_dir, upscaling_factor=2, 
                       residual_blocks=residual_blocks, feature_size=feature_size, 
                       subpixel_NN=True, nn=False, restore=restore, batch_size=batch_size, 
                       epochs=epochs, saveiters=saveiters, train_fraction=train_fraction, normalize=normalize
)


gfm.evaluate(modelName, kernel, img_zdepth, img_height, img_width, img_path, checkpoint_dir, 
             result_dir, batch_size=batch_size, feature_size=feature_size, upscaling_factor=upscaling_factor, 
             residual_blocks=6, subpixel_NN=True, nn=False)


#Changelog:
#12/6 - introduced gen_lossrate variable and set to 20e-4 (was 10e-3, 10e-2 before that)
    
    
    
    