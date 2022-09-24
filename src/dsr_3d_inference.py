#!/usr/bin/python3
import os.path
from os import path
import numpy as np
import ants
import random
from datetime import datetime
import sys
import antspynet
import glob as glob
import numpy as np
import tensorflow as tf
import ants
import tensorflow.keras as keras

def get_random_base_ind( full_dims=(256,256,256), off = 10, patchWidth = 96 ):
    baseInd = [None,None,None]
    for k in range(3):
        baseInd[k]=random.sample( range( off, full_dims[k]-1-patchWidth ), 1 )[0]
    return baseInd


# extract a random patch
def get_random_patch( img, patchWidth ):
    mystd = 0
    while mystd == 0:
        inds = get_random_base_ind( full_dims = img.shape )
        hinds = [None,None,None]
        for k in range(len(inds)):
            hinds[k] = inds[k] + patchWidth
        myimg = ants.crop_indices( img, inds, hinds )
        mystd = myimg.std()
    return myimg

def get_random_patch_pair( img, img2, patchWidth ):
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

ofn = 'models/dsr3d_2up_64_256_6_3_v0.0.h5'
mdl = tf.keras.models.load_model( ofn, compile=False )

print("assemble images", flush=True )
imgfns = glob.glob( "/raid/data_BA/brains/HCP/T*w/*nii.gz") + glob.glob( "/raid/data_BA/brains/HCP/T*w/*nii.gz")
if len(imgfns) == 0:
    imgfns = glob.glob( "/Users/stnava/.antspyt1w/2*T1w*gz")
random.shuffle(imgfns)
print( imgfns[0] )
img = ants.image_read( imgfns[0] )
img1, img2 = get_random_patch_pair( img, img, patchWidth=96 )
ants.plot(img1)
spc = ants.get_spacing( img1 )
newspc = []
for j in range( img1.dimension ):
    newspc.append( spc[j] * 2. )

img1lo = ants.resample_image( img1, newspc )
img1up = ants.resample_image( img1lo, spc )

print('begin sr : psnr orig = ' + str( antspynet.psnr(img1,img1up) ) )
sr = antspynet.apply_super_resolution_model_to_image(
    img1lo,
    mdl,
    target_range=[0,1], regression_order=None )
# some metrics on the output
gmsdSR = antspynet.gmsd(img1,sr)
gmsdBi = antspynet.gmsd(img1,img1up)
ssimSR = antspynet.ssim(img1,sr)
ssimBi = antspynet.ssim(img1,img1up)
psnrSR = antspynet.psnr(img1,sr)
psnrBi = antspynet.psnr(img1,img1up)
print("PSNR Test: " + str( psnrBi ) + " vs SR: " + str( psnrSR ), flush=True  )
print("GMSD Test: " + str( gmsdBi ) + " vs SR: " + str( gmsdSR ), flush=True  )
print("ssim Test: " + str( ssimBi ) + " vs SR: " + str( ssimSR ), flush=True  )
ants.plot( img1 )
ants.plot( img1up )
ants.plot( sr )

# now apply to "real" case
srhi = antspynet.apply_super_resolution_model_to_image(
    img1,
    mdl,
    target_range=[0,1], regression_order=None )

ants.image_write( img1, '/tmp/or.nii.gz' )
ants.image_write( srhi, '/tmp/srhi.nii.gz' )
