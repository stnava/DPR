#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/gist/stnava/061b2cb8f79c582822e539f238809bd8/xray_sr_randpatch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[65]:


#!/usr/bin/python3
import os.path
from os import path
import numpy as np
import random
# !pip uninstall antspyx
# get_ipython().system('pip install antspyx')
# get_ipython().system('pip install git+https://github.com/ANTsX/ANTsPyNet')
import ants


# In[66]:


import ants

import random
from datetime import datetime
# In[67]:


# from google.colab import drive
# drive.mount('/content/drive')


# **overview**<br>
# <br>
# we are reviewing a simple script for training a super-resolution network<br>
# that uses a straightforward approach and does not require a discriminator<br>
# network which is, generally speaking, much harder to optimize/control.<br>
# instead, we rely on total variation and perceptual losses in addition to<br>
# the standard reconstruction loss ( mean squared error ).  while this is<br>
# done in TF, it could be implemented transparently in pytorch or whatever.<br>
# the implementation will also work in 3D if weight transfer is done as<br>
# explained [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7279929/).

# **goals**<br>
# provide a conceptual and practical foundation for how to train a super-Resolution<br>
# network.  ideally, we can collectively identify a better network, parameter set,<br>
# evaluation criterion, perceptual loss or other feature that would improve<br>
# on what we have now.  the hyperplane and DGX resources support such efforts.

# **glossary**<br>
# <br>
# [*super-resolution*](https://scholar.google.com/scholar?q=super-resolution+image&hl=en&as_sdt=0%2C30&as_ylo=1965&as_yhi=1977): a typically nonlinear approach to upsampling data in order<br>
# to increase the effective or perceived image resolution.<br>
# <br>
# *deep backprojection network (DBPN)*: Haris et al Deep Back-Projection Networks For Super-Resolution<br>
# a residual network that does both up and downsampling and uses perceptual losses.<br>
# the number of backprojection layers relates to how many residual layers exist.<br>
# [references here](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C30&q=haris+deep+back+projection+network&btnG=)<br>
# <br>
# *perceptual loss*: - a loss function that maps raw image features to a<br>
# higher-dimensional feature space defined by a pre-trained network<br>
# [references](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C30&q=perceptual+losses+super+resolution&btnG=&oq=perceptual+losses+super)

# **input parameters**<br>
# <br>
# user defined parameters below - they include options for perceptual features<br>
# the number of back-projection layers, patch scaling options and weighting<br>
# terms for the loss function

# In[68]:


import sys
arglist = (sys.argv)
arglist=["name",5,"True",0.5,256,"True","True"]
nbp=int( arglist[1] )
ctmod = 10
patch_scale = False


# **Note**: loss function weights are important for performance and are typically<br>
# set empirically or by parameter search where quality is assessed against an independent metric

# In[69]:


# set up naming such that we know something about the stored network

# In[70]:



# standard imports for I/O, sampling and image manipulation

# In[71]:


import ants
import antspynet
import random
import glob as glob
import numpy as np
import tensorflow as tf
import ants
import tensorflow.keras as keras


# **dependencies**<br>
# 1. tensorflow 2.0 or higher<br>
# 2. [ANTsPy](https://github.com/ANTsX/ANTsPy)<br>
# 3. [ANTsPyNet](https://github.com/ANTsX/ANTsPyNet)

# we use freely available data for reference - one can get started with just<br>
# one of the tar.gz files - these should be unzipped and will result in a folder<br>
# called "images/" with a bunch of jpgs in it<br>
# get data from here https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737

# import the VGG19 network with pre-trained weights

# In[72]:


from tensorflow.keras.applications.vgg19 import VGG19
# our VGG19 assumes data is centered about 0 with a range of 255 the
# offset intensity controls this where range is offsetIntensity*2
offsetIntensity = 127.5


# load in relevant objects from keras so we can build our network


from keras.models import Model
from keras.layers import (Input, Add, Subtract,
                          PReLU, Concatenate,
                          UpSampling2D, UpSampling3D,
                          Conv2D, Conv2DTranspose,
                          Conv3D, Conv3DTranspose)


# define the DBPN network - this uses a model definition that is general to<br>
# both 2D and 3D. recommended parameters for different upsampling rates can<br>
# be found in the papers by Haris et al.  We make one significant change to<br>
# the original architecture by allowing standard interpolation for upsampling<br>
# instead of convolutional upsampling.  this is controlled by the interpolation<br>
# option.

# In[74]:


def dbpn(input_image_size,
                                                 number_of_outputs=1,
                                                 number_of_base_filters=64,
                                                 number_of_feature_filters=256,
                                                 number_of_back_projection_stages=7,
                                                 convolution_kernel_size=(12, 12),
                                                 strides=(8, 8),
                                                 last_convolution=(3, 3),
                                                 number_of_loss_functions=1,
                                                 interpolation = 'nearest'
                                                ):
    idim = len( input_image_size ) - 1
    if idim == 2:
        myconv = Conv2D
        myconv_transpose = Conv2DTranspose
        myupsampling = UpSampling2D
        shax = ( 1, 2 )
        firstConv = (3,3)
        firstStrides=(1,1)
        smashConv=(1,1)
    if idim == 3:
        myconv = Conv3D
        myconv_transpose = Conv3DTranspose
        myupsampling = UpSampling3D
        shax = ( 1, 2, 3 )
        firstConv = (3,3,3)
        firstStrides=(1,1,1)
        smashConv=(1,1,1)
    def up_block_2d(L, number_of_filters=64, kernel_size=(12, 12), strides=(8, 8),
                    include_dense_convolution_layer=True):
        if include_dense_convolution_layer == True:
            L = myconv(filters = number_of_filters,
                       use_bias=True,
                       kernel_size=smashConv,
                       strides=firstStrides,
                       padding='same')(L)
            L = PReLU(alpha_initializer='zero',
                      shared_axes=shax)(L)
        # Scale up
        if idim == 2:
            H0 = myupsampling( size = strides, interpolation=interpolation )(L)
        if idim == 3:
            H0 = myupsampling( size = strides )(L)
        H0 = myconv(filters=number_of_filters,
                    kernel_size=firstConv,
                    strides=firstStrides,
                    use_bias=True,
                    padding='same')(H0)
        H0 = PReLU(alpha_initializer='zero',
                   shared_axes=shax)(H0)
        # Scale down
        L0 = myconv(filters=number_of_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    kernel_initializer='glorot_uniform',
                    padding='same')(H0)
        L0 = PReLU(alpha_initializer='zero',
                   shared_axes=shax)(L0)
        # Residual
        E = Subtract()([L0, L])
        # Scale residual up
        if idim == 2:
            H1 = myupsampling( size = strides, interpolation=interpolation  )(E)
        if idim == 3:
            H1 = myupsampling( size = strides )(E)
        H1 = myconv(filters=number_of_filters,
                    kernel_size=firstConv,
                    strides=firstStrides,
                    use_bias=True,
                    padding='same')(H1)
        H1 = PReLU(alpha_initializer='zero',
                   shared_axes=shax)(H1)
        # Output feature map
        up_block = Add()([H0, H1])
        return up_block
    def down_block_2d(H, number_of_filters=64, kernel_size=(12, 12), strides=(8, 8),
                    include_dense_convolution_layer=True):
        if include_dense_convolution_layer == True:
            H = myconv(filters = number_of_filters,
                       use_bias=True,
                       kernel_size=smashConv,
                       strides=firstStrides,
                       padding='same')(H)
            H = PReLU(alpha_initializer='zero',
                      shared_axes=shax)(H)
        # Scale down
        L0 = myconv(filters=number_of_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    kernel_initializer='glorot_uniform',
                    padding='same')(H)
        L0 = PReLU(alpha_initializer='zero',
                   shared_axes=shax)(L0)
        # Scale up
        if idim == 2:
            H0 = myupsampling( size = strides, interpolation=interpolation )(L0)
        if idim == 3:
            H0 = myupsampling( size = strides )(L0)
        H0 = myconv(filters=number_of_filters,
                    kernel_size=firstConv,
                    strides=firstStrides,
                    use_bias=True,
                    padding='same')(H0)
        H0 = PReLU(alpha_initializer='zero',
                   shared_axes=shax)(H0)
        # Residual
        E = Subtract()([H0, H])
        # Scale residual down
        L1 = myconv(filters=number_of_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    kernel_initializer='glorot_uniform',
                    padding='same')(E)
        L1 = PReLU(alpha_initializer='zero',
                   shared_axes=shax)(L1)
        # Output feature map
        down_block = Add()([L0, L1])
        return down_block
    inputs = Input(shape=input_image_size)
    # Initial feature extraction
    model = myconv(filters=number_of_feature_filters,
                   kernel_size=firstConv,
                   strides=firstStrides,
                   padding='same',
                   kernel_initializer='glorot_uniform')(inputs)
    model = PReLU(alpha_initializer='zero',
                  shared_axes=shax)(model)
    # Feature smashing
    model = myconv(filters=number_of_base_filters,
                   kernel_size=smashConv,
                   strides=firstStrides,
                   padding='same',
                   kernel_initializer='glorot_uniform')(model)
    model = PReLU(alpha_initializer='zero',
                  shared_axes=shax)(model)
    # Back projection
    up_projection_blocks = []
    down_projection_blocks = []
    model = up_block_2d(model, number_of_filters=number_of_base_filters,
      kernel_size=convolution_kernel_size, strides=strides)
    up_projection_blocks.append(model)
    for i in range(number_of_back_projection_stages):
        if i == 0:
            model = down_block_2d(model, number_of_filters=number_of_base_filters,
              kernel_size=convolution_kernel_size, strides=strides)
            down_projection_blocks.append(model)
            model = up_block_2d(model, number_of_filters=number_of_base_filters,
              kernel_size=convolution_kernel_size, strides=strides)
            up_projection_blocks.append(model)
            model = Concatenate()(up_projection_blocks)
        else:
            model = down_block_2d(model, number_of_filters=number_of_base_filters,
              kernel_size=convolution_kernel_size, strides=strides,
              include_dense_convolution_layer=True)
            down_projection_blocks.append(model)
            model = Concatenate()(down_projection_blocks)
            model = up_block_2d(model, number_of_filters=number_of_base_filters,
              kernel_size=convolution_kernel_size, strides=strides,
              include_dense_convolution_layer=True)
            up_projection_blocks.append(model)
            model = Concatenate()(up_projection_blocks)
    outputs = myconv(filters=number_of_outputs,
                     kernel_size=last_convolution,
                     strides=firstStrides,
                     padding = 'same',
                     kernel_initializer = "glorot_uniform")(model)
    if number_of_loss_functions == 1:
        deep_back_projection_network_model = Model(inputs=inputs, outputs=outputs)
    else:
        outputList=[]
        for k in range(number_of_loss_functions):
            outputList.append(outputs)
        deep_back_projection_network_model = Model(inputs=inputs, outputs=outputList)
    return deep_back_projection_network_model




# set up strides and patch sizes - **these could also be explored empirically**
pszlo = 32
psz = pszlo * 2

# generate a random corner index for a patch

def get_random_base_ind( full_dims=(256,256,256), off = 10, patchWidth = 96 ):
    baseInd = [None,None,None]
    for k in range(3):
        baseInd[k]=random.sample( range( off, full_dims[k]-1-patchWidth ), 1 )[0]
    return baseInd


# extract a random patch
def get_random_patch( img, patchWidth=psz ):
    mystd = 0
    while mystd == 0:
        inds = get_random_base_ind( full_dims = img.shape )
        hinds = [None,None,None]
        for k in range(len(inds)):
            hinds[k] = inds[k] + patchWidth
        myimg = ants.crop_indices( img, inds, hinds )
        mystd = myimg.std()
    return myimg

def get_random_patch_pair( img, img2, patchWidth=psz ):
    mystd = mystd2 = 0
    ct = 0
    while mystd == 0 or mystd2 == 0:
        inds = get_random_base_ind( full_dims = img.shape )
        hinds = [None,None,None]
        for k in range(len(inds)):
            hinds[k] = inds[k] + patchWidth
        myimg = ants.crop_indices( img, inds, hinds )
        myimg2 = ants.crop_indices( img2, inds, hinds )
        mystd = myimg.std()
        mystd2 = myimg2.std()
        ct = ct + 1
        if ( ct > 20 ):
            return myimg, myimg2
    return myimg, myimg2


# set up qc features
# myvgg = tf.keras.l
# perceptual features - one can explore different layers and features
# these layers - or combinations of these - are commonly used in the literature
# as feature spaces for loss functions.  weighting terms relative to MSE are
# also given in several papers which can help guide parameter setting.
grader = antspynet.create_resnet_model_3d(
    [None,None,None,1],
    lowest_resolution = 32,
    number_of_classification_labels = 4,
    cardinality = 1,
    squeeze_and_excite = False )
# the folder and data below as available from antspyt1w get_data
graderfn = os.path.expanduser("~" + "/.antspyt1w/resnet_grader.h5" )
grader.load_weights( graderfn)
feature_extractor = tf.keras.Model( inputs=grader.inputs, outputs=grader.layers[6].output )


def my_loss_msq(y_true, y_pred  ):
    squared_difference = tf.square(y_true - y_pred)
    myax = [1,2,3,4]
    msqTerm = tf.reduce_mean(squared_difference, axis=myax)
    return ( msqTerm  )

# **model instantiation**: these are close to defaults for the 2x network.<br>
# empirical evidence suggests that making covolutions and strides evenly<br>
# divisible by each other reduces artifacts.  2*3=6.

# In[82]:

nfilt=64
nff = 256
convn = 6
ofn='./models/dsr3d_2up_' + str(nfilt) + '_' + str( nff ) + '_' + str(convn)+ '_' + str(os.environ['CUDA_VISIBLE_DEVICES'])+'_v0.0.h5'
mdl = dbpn( (None,None,None,1),
  number_of_outputs=1,
  number_of_base_filters=nfilt,
  number_of_feature_filters=nff,
  number_of_back_projection_stages=nbp,
  convolution_kernel_size=(convn, convn, convn),
  strides=(2, 2, 2),
  last_convolution=(3, 3, 3), number_of_loss_functions=1, interpolation='nearest')

# collect all the images you have locally

print("assemble images", flush=True )
imgfns = glob.glob( "/raid/data_BA/brains/HCP/T*w/*nii.gz") + glob.glob( "/raid/data_BA/brains/HCP/T*w/*nii.gz")
if len(imgfns) == 0:
    imgfns = glob.glob( "/Users/stnava/.antspyt1w/2*T1w*gz")
random.shuffle(imgfns)
# 90\% training data
n = round( len( imgfns ) * 0.9 )
imgfnsTrain = imgfns[0:n]      # just start small
imgfnsTest = imgfns[(n+1):len(imgfns)]    # just a few test for now
# get a reference image from the sample - we assume all are the same size
# here which is true in this case.  small changes would be needed for
# generalization.
img = ants.image_read( imgfnsTrain[0] )
print( imgfns[0] )


# In[91]:


img = ants.image_read( imgfnsTrain[0] )
img2 = ants.image_read( imgfnsTrain[1] )
patch1, patch2 = get_random_patch_pair( img, img, patchWidth=32 )
# ants.plot(patch1)
# ants.plot(patch2)


# get pre-trained network weights
if os.path.isfile(ofn):
    print( "load " + ofn )
    mdl = tf.keras.models.load_model( ofn, compile=False )


# resampling - use a custom antspynet layer for this (also in antsrnet)<br>
# key here is that these networks dont need to know the input spatial<br>
# dimensions in order to work effectively.  this is also a very tricky<br>
# part of TF: the coordinate system for resampling is different than, for<br>
# example, ITK.  not sure this is well-documented but it means that mixing<br>
# resampling done by TF and by ITK will not provide the same results and this<br>
# will impact training, inference, evaluation, etc.

# In[88]:


myinput = tf.keras.layers.Input( [None,None,None,1] )
mytarget = tf.keras.layers.Input( [None,None,None,1] )
output = antspynet.ResampleTensorToTargetTensorLayer3D('linear')([myinput, mytarget])
rmodel = tf.keras.Model(inputs=[myinput, mytarget], outputs=output)
outputnn = antspynet.ResampleTensorToTargetTensorLayer3D('nearest_neighbor')([myinput, mytarget])
rmodelnn = tf.keras.Model(inputs=[myinput, mytarget], outputs=outputnn)


# build some test data - high res, low res and bilinear

# In[93]:


# **data generation**<br>
# recent versions of tensorflow/keras allow data generators to be passed<br>
# directly to the fit function.  underneath, this does a fairly efficient split<br>
# between GPU and CPU usage and data transfer.  EG probably knows more about this.<br>
# this generator randomly chooses between linear and nearest neighbor downsampling.<br>
# the *patch_scale* option can also be seen here which impacts how the network<br>
# sees/learns from image intensity.

# In[94]:


def my_generator( nPatches , nImages = 16, istest=False,
    target_patch_size=psz,
    target_patch_size_low=pszlo,
    patch_scaler=True, verbose = False ):
    while True:
        for myn in range(nImages):
            patchesOrig = np.zeros(shape=(nPatches,target_patch_size,target_patch_size,target_patch_size,1))
            patchesResam = np.zeros(shape=(nPatches,target_patch_size_low,target_patch_size_low,target_patch_size_low,1))
            if not istest:
                imgfn = random.sample( imgfnsTrain, 1 )[0]
            else:
                imgfn = random.sample( imgfnsTest, 1 )[0]
            if verbose:
                print(imgfn)
            img = ants.image_read( imgfn ).iMath("Normalize")
            if img.components > 1:
                img = ants.split_channels(img)[0]
            img = ants.crop_image( img, ants.threshold_image( img, 0.05, 1 ) )
            ants.set_origin( img, ants.get_center_of_mass(img) )
            img = ants.iMath(img,"Normalize")
            spc = ants.get_spacing( img )
            newspc = []
            for jj in range(len(spc)):
                newspc.append(spc[jj]*2.0)
            interp_type = random.choice( [0,1] )
            # masker = ants.threshold_image(img,0.01,1)
            # rmasker = ants.threshold_image(rimg,0.01,1)
            # rimg = ants.crop_image( rimg * rmasker, masker )
            # img = ants.crop_image( img * masker, masker )
            for myb in range(nPatches):
                imgp = get_random_patch( img, target_patch_size )
                rimgp = ants.resample_image( imgp, newspc, use_voxels = False, interp_type=interp_type  )
                imgpmin = imgp.min()
                if patch_scaler:
                    imgp = imgp - imgpmin
                    rimgp = rimgp - imgpmin
                    imgpmax = imgp.max()
                    if imgpmax > 0 :
                        imgp = imgp / imgpmax
                        rimgp = rimgp / imgpmax
                patchesOrig[myb,:,:,:,0] = imgp.numpy()
                patchesResam[myb,:,:,:,0] = rimgp.numpy()
            patchesOrig = tf.cast( patchesOrig, "float32")
            patchesResam = tf.cast( patchesResam, "float32")
            yield (patchesResam, patchesOrig)

# instanstiate the generator function with a given sub-batch and total batch size<br>
# i dont entirely understand how this works (it's farily new) but it seems to<br>
# spit off sub-batches of the given size until it's exhausted the total number<br>
# of batches which would then count as an epoch.

# In[113]:

mybs = 1
mydatgen = my_generator( 2, mybs, istest=False ) # FIXME for a real training run
mydatgenTest = my_generator( 2, mybs, istest=True ) # FIXME for a real training run
patchesResamTeTf, patchesOrigTeTf = next( mydatgenTest )

def my_loss_6(y_true, y_pred,
  msqwt = tf.constant( 10.0 ),
  fw=tf.constant( 100.0), # this is a starter weight - might need to be optimized
  tvwt = tf.constant( 1.0e-8 ) ): # this is a starter weight - might need to be optimized
    squared_difference = tf.square(y_true - y_pred)
    myax = [1,2,3,4]
    msqTerm = tf.reduce_mean(squared_difference, axis=myax)
    temp1 = feature_extractor(y_true)
    temp2 = feature_extractor(y_pred)
    vggsquared_difference = tf.square(temp1-temp2)
    vggTerm = tf.reduce_mean(vggsquared_difference, axis=myax)
    tvTerm = tf.cast( 0.0, 'float32')
    loss = msqTerm * msqwt + vggTerm * fw
    # return( loss )
    mytv = 0.0
    # myr = y_true.shape.as_list()[0]
    for k in range( 4 ): # BUG not sure why myr fails .... might be old TF version
        sqzd = y_pred[k,:,:,:,:]
        mytv = mytv + tf.reduce_mean( tf.image.total_variation( sqzd ) ) * tvwt
    return( loss + mytv )

# my_loss_6( patchesPred, patchesOrigTeTf )

# set an optimizer - just standard Adam - may be sensitive to learning_rate
opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
mdl.compile(optimizer=opt, loss=my_loss_6)

# set up some parameters for tracking performance
bestValLoss=1e12
bestSSIM=0.0
bestQC0 = -1000
bestQC1 = -1000
print( "begin training", flush=True  )
derka
for myrs in range( 100000 ):
    tracker = mdl.fit( mydatgen,  epochs=2, steps_per_epoch=10, verbose=1,
        validation_data=(patchesResamTeTf,patchesOrigTeTf),
        workers = 1, use_multiprocessing=False )
    print( "ntrain: " + str(myrs) + " loss " + str( tracker.history['loss'][0] ) + ' val-loss ' + str(tracker.history['val_loss'][0]), flush=True  )
    if ( tracker.history['val_loss'][0] < bestValLoss ):
        print("MyIT " + str( myrs ) + " IS BEST!!", flush=True )
        bestValLoss = tracker.history['val_loss'][0]
        tf.keras.models.save_model( mdl, ofn )
