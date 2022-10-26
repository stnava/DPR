#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/gist/stnava/061b2cb8f79c582822e539f238809bd8/xray_sr_randpatch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[65]:


#!/usr/bin/python3
import os.path
from os import path
from pathlib import Path
import sys
if '__file__' in vars() or '__file__' in globals():
    path_root = Path(__file__)
else:
    __file__='./'
    path_root = Path(__file__)
sys.path.append(str(path_root))

import numpy as np
import random
# !pip uninstall antspyx
# get_ipython().system('pip install antspyx')
# get_ipython().system('pip install git+https://github.com/ANTsX/ANTsPyNet')
import ants

plaindb=False
try:
  import src.dbpn_arch
except ImportError:
  import dbpn_arch
  plaindb=True


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
patch_scale = True


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



# set up strides and patch sizes - **these could also be explored empirically**
pszlo = [72,72,12]
strider = [1,1,6] # this controls the amount of upsampling  -- 1 = none
psz=[]
namer=''
for k in range(len(strider)):
    psz.append( pszlo[k] * strider[k] )
    tailer=''
    if k < (len(strider)-1):
        tailer='x'
    namer=namer+str(strider[k])+tailer

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
            hinds[k] = inds[k] + patchWidth[k]
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
            hinds[k] = inds[k] + patchWidth[k]
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
feature_extractor_23 = tf.keras.Model( inputs=grader.inputs, outputs=grader.layers[23].output )
feature_extractor_44 = tf.keras.Model( inputs=grader.inputs, outputs=grader.layers[44].output )


def my_loss_msq(y_true, y_pred  ):
    squared_difference = tf.square(y_true - y_pred)
    myax = [1,2,3,4]
    msqTerm = tf.reduce_mean(squared_difference, axis=myax)
    return ( msqTerm  )

# **model instantiation**: these are close to defaults for the 2x network.<br>
# empirical evidence suggests that making covolutions and strides evenly<br>
# divisible by each other reduces artifacts.  2*3=6.

nfilt=64
nff = 256
convn = 6
lastconv = 3
nbp=7
ofn='./models/dsr3d_'+namer+'up_' + str(nfilt) + '_' + str( nff ) + '_' + str(convn)+ '_' + str(lastconv)+ '_' + str(os.environ['CUDA_VISIBLE_DEVICES'])+'_v0.0.h5'
if plaindb:
    mdl = dbpn_arch.dbpn( (None,None,None,1),
      number_of_outputs=1,
      number_of_base_filters=nfilt,
      number_of_feature_filters=nff,
      number_of_back_projection_stages=nbp,
      convolution_kernel_size=(convn, convn, convn),
      strides=(strider[0], strider[1], strider[2]),
      last_convolution=(lastconv, lastconv, lastconv), number_of_loss_functions=1, interpolation='nearest')
else:
    mdl = src.dbpn_arch.dbpn( (None,None,None,1),
      number_of_outputs=1,
      number_of_base_filters=nfilt,
      number_of_feature_filters=nff,
      number_of_back_projection_stages=nbp,
      convolution_kernel_size=(convn, convn, convn),
      strides=(strider[0], strider[1], strider[2]),
      last_convolution=(lastconv, lastconv, lastconv), number_of_loss_functions=1, interpolation='nearest')

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


# **data generation**<br>
# recent versions of tensorflow/keras allow data generators to be passed<br>
# directly to the fit function.  underneath, this does a fairly efficient split<br>
# between GPU and CPU usage and data transfer.<br>
# this generator randomly chooses between linear and nearest neighbor downsampling.<br>
# the *patch_scale* option can also be seen here which impacts how the network<br>
# sees/learns from image intensity.
def my_generator_dsr( nPatches , nImages = 16, istest=False,
    target_patch_size=psz,
    target_patch_size_low=pszlo,
    patch_scaler=patch_scale, verbose = False ):
    while True:
        for myn in range(nImages):
            patchesOrig = np.zeros(shape=(nPatches,target_patch_size[0],target_patch_size[1],target_patch_size[2],1))
            patchesResam = np.zeros(shape=(nPatches,target_patch_size_low[0],target_patch_size_low[1],target_patch_size_low[2],1))
            if not istest:
                imgfn = random.sample( imgfnsTrain, 1 )[0]
            else:
                patchesUp = np.zeros(shape=patchesOrig.shape)
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
                newspc.append(spc[jj]*strider[jj])
            interp_type = random.choice( [0,1] )
            for myb in range(nPatches):
                imgp = get_random_patch( img, target_patch_size )
                imgpmin = imgp.min()
                if patch_scaler:
                    imgp = imgp - imgpmin
                    imgpmax = imgp.max()
                    if imgpmax > 0 :
                        imgp = imgp / imgpmax
                rimgp = ants.resample_image( imgp, newspc, use_voxels = False, interp_type=interp_type  )
                if istest:
                    rimgbi = ants.resample_image( rimgp, spc, use_voxels = False, interp_type=0  )
                patchesOrig[myb,:,:,:,0] = imgp.numpy()
                patchesResam[myb,:,:,:,0] = rimgp.numpy()
                if istest:
                    patchesUp[myb,:,:,:,0] = rimgbi.numpy()
            patchesOrig = tf.cast( patchesOrig, "float32")
            patchesResam = tf.cast( patchesResam, "float32")
            if istest:
                patchesUp = tf.cast( patchesUp, "float32")
                yield (patchesResam, patchesOrig,patchesUp)
            yield (patchesResam, patchesOrig)


def my_generator_dpr( nPatches , nImages = 16, istest=False,
    target_patch_size=psz,
    target_patch_size_low=psz,
    patch_scaler=patch_scale, verbose = False ):
    while True:
        for myn in range(nImages):
            patchesOrig = np.zeros(shape=(nPatches,target_patch_size[0],target_patch_size[1],target_patch_size[2],1))
            patchesResam = np.zeros(shape=(nPatches,target_patch_size[0],target_patch_size[1],target_patch_size[2],1))
            if not istest:
                imgfn = random.sample( imgfnsTrain, 1 )[0]
            else:
                patchesUp = np.zeros(shape=patchesOrig.shape)
                imgfn = random.sample( imgfnsTest, 1 )[0]
            if verbose:
                print(imgfn)
            img = ants.image_read( imgfn ).iMath("Normalize")
            if img.components > 1:
                img = ants.split_channels(img)[0]
            rRotGenerator = ants.contrib.RandomRotate3D( ( -25, 25 ), reference=img )
            tx0 = rRotGenerator.transform()
            tx0inv = ants.invert_ants_transform(tx0)
            rimg = tx0.apply_to_image( img )
            rimg = tx0inv.apply_to_image( rimg )
            interp_type = random.choice( [0,1] )
            for myb in range(nPatches):
                imgp1, imgp2 = get_random_patch_pair( img, rimg, patchWidth=target_patch_size )
                imgpmin = imgp1.min()
                if patch_scaler:
                    imgp1 = imgp1 - imgpmin
                    imgp2 = imgp2 - imgpmin
                    imgpmax = imgp1.max()
                    if imgpmax > 0 :
                        imgp1 = imgp1 / imgpmax
                        imgp2 = imgp2 / imgpmax
                patchesOrig[myb,:,:,:,0] = imgp1.numpy()
                patchesResam[myb,:,:,:,0] = imgp2.numpy()
            patchesOrig = tf.cast( patchesOrig, "float32")
            patchesResam = tf.cast( patchesResam, "float32")
            if istest:
                yield (patchesResam, patchesOrig,patchesResam)
            yield (patchesResam, patchesOrig)







# instanstiate the generator function with a given sub-batch and total batch size<br>
# i dont entirely understand how this works (it's farily new) but it seems to<br>
# spit off sub-batches of the given size until it's exhausted the total number<br>
# of batches which would then count as an epoch.

# In[113]:

mybs = 1
if strider[2] > 1:
    my_generator = my_generator_dsr
else:
    my_generator = my_generator_dpr

mydatgen = my_generator( 1, mybs, istest=False , verbose=False) # FIXME for a real training run
mydatgenTest = my_generator( 1, mybs, istest=True, verbose=True) # FIXME for a real training run

patchesResamTeTf, patchesOrigTeTf, patchesUpTeTf = next( mydatgenTest )

# for cpu testing - too costly to run on gpu - will take cpu penalty here
mydatgenTest = my_generator( 1, mybs, istest=True, verbose=True) # FIXME for a real training run
patchesResamTeTfB, patchesOrigTeTfB, patchesUpTeTfB = next( mydatgenTest )
for k in range( 11 ):
    mydatgenTest = my_generator( 1, mybs, istest=True, verbose=True) # FIXME for a real training run
    temp0, temp1, temp2 = next( mydatgenTest )
    patchesResamTeTfB = tf.concat( [patchesResamTeTfB,temp0],axis=0)
    patchesOrigTeTfB = tf.concat( [patchesOrigTeTfB,temp1],axis=0)
    patchesUpTeTfB = tf.concat( [patchesUpTeTfB,temp2],axis=0)

def my_loss_6(y_true, y_pred,
  msqwt = tf.constant( 10.0 ),
  fw=tf.constant( 2000.0), # this is a starter weight - might need to be optimized
  tvwt = tf.constant( 5.0e-8 ) ): # this is a starter weight - might need to be optimized
    squared_difference = tf.square(y_true - y_pred)
    myax = [1,2,3,4]
    msqTerm = tf.reduce_mean(squared_difference, axis=myax)
    temp1 = feature_extractor_44(y_true)
    temp2 = feature_extractor_44(y_pred)
    vggsquared_difference = tf.square(temp1-temp2)
    vggTerm = tf.reduce_mean(vggsquared_difference, axis=myax)
    loss = msqTerm * msqwt + vggTerm * fw
    mytv = tf.cast( 0.0, 'float32')
    for k in range( mybs ): # BUG not sure why myr fails .... might be old TF version
        sqzd = y_pred[k,:,:,:,:]
        mytv = mytv + tf.reduce_mean( tf.image.total_variation( sqzd ) ) * tvwt
    return( loss + mytv )

# my_loss_6( patchesPred, patchesOrigTeTf )

# set an optimizer - just standard Adam - may be sensitive to learning_rate
opt = tf.keras.optimizers.Adam(learning_rate=5e-5)
mdl.compile(optimizer=opt, loss=my_loss_6)

# set up some parameters for tracking performance
bestValLoss=1e12
bestSSIM=0.0
bestQC0 = -1000
bestQC1 = -1000
print( "begin training", flush=True  )
for myrs in range( 100000 ):
    tracker = mdl.fit( mydatgen,  epochs=2, steps_per_epoch=4, verbose=1,
        validation_data=(patchesResamTeTf,patchesOrigTeTf),
        workers = 1, use_multiprocessing=False )
    print( "ntrain: " + str(myrs) + " loss " + str( tracker.history['loss'][0] ) + ' val-loss ' + str(tracker.history['val_loss'][0]), flush=True  )
    if myrs % 20 == 0:
        with tf.device("/cpu:0"):
            tester = mdl.evaluate( patchesResamTeTfB, patchesOrigTeTfB )
            if ( tester < bestValLoss ):
                print("MyIT " + str( myrs ) + " IS BEST!! " + str( tester ) , flush=True )
                bestValLoss = tester
                tf.keras.models.save_model( mdl, ofn )
            pp = mdl.predict( patchesResamTeTfB, batch_size = 1 )
            myssimSR = tf.image.psnr( pp * 220, patchesOrigTeTfB* 220, max_val=255 )
            myssimSR = tf.reduce_mean( myssimSR ).numpy()
            myssimBI = tf.image.psnr( patchesUpTeTfB * 220, patchesOrigTeTfB* 220, max_val=255 )
            myssimBI = tf.reduce_mean( myssimBI ).numpy()
            print( "PSNR Lin: " + str( myssimBI ) + " SR: " + str( myssimSR ), flush=True  )




#
patchesPred = mdl( patchesResamTeTf )
squared_difference = tf.square(patchesPred - patchesOrigTeTf)
msqTerm = tf.reduce_mean(squared_difference )
vggTerm = tf.reduce_mean(tf.square(feature_extractor(patchesOrigTeTf)-feature_extractor(patchesPred)))
vggTerm = tf.reduce_mean(tf.square(feature_extractor_23(patchesOrigTeTf)-feature_extractor_23(patchesPred)))
vggTerm = tf.reduce_mean(tf.square(feature_extractor_44(patchesOrigTeTf)-feature_extractor_44(patchesPred)))
# qcTerm = tf.reduce_mean( tf.square( qcmodel( patchesPred/127.5 ) - qcmodel( patchesHiTe/127.5 ) ), axis=[0])
tvTerm = tf.reduce_mean( tf.image.total_variation( tf.squeeze(patchesPred[0,:,:,:,:] ) ))
print( msqTerm * 10 )
print( vggTerm * 150  )
print( tvTerm  * 1e-7 )
my_loss_6( patchesPred, patchesOrigTeTf )
