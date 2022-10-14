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
import glob

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

mdlfns = glob.glob( "models/*sr*3d*h5" )
mdlfns = glob.glob( "models/dsr3d_2up_64_256_6_3_v0.0.h5" )
mdlfns=['models/dpr3d_64_256_6_3_v0.0.h5']
mdlfns = glob.glob( "models/dpr*3d*h5" )
mdlfns.sort()
print(mdlfns)
print("assemble images", flush=True )
imgfns = glob.glob( "/raid/data_BA/brains/HCP/T*w/*nii.gz") + glob.glob( "/raid/data_BA/brains/HCP/T*w/*nii.gz")
if len(imgfns) == 0:
    imgfns = glob.glob( "/Users/stnava/.antspyt1w/28386-00000000-T1w-01.nii.gz")
    imgfns.sort()

random.shuffle(imgfns)
print( imgfns[0] )
img = ants.image_read( imgfns[0] ).iMath("Normalize").iMath("TruncateIntensity",0.02,0.98)
seg = ants.threshold_image( img, 'Otsu', 3)

image1_list = list()
image1_list.append(img)
input_images = list()
input_images.append(image1_list)

rimg = antspynet.randomly_transform_image_data(img,
    input_images, sd_affine=0.1,
    transform_type = "rigid" )
k=4
img1, img2 = get_random_patch_pair( img, rimg['simulated_images'][k][0], patchWidth=64 )
ants.plot(img2)
antspynet.psnr(img1,img2)
spc = ants.get_spacing( img1 )
newspc = []
for j in range( img1.dimension ):
    newspc.append( spc[j] * 1 )

img1lo = ants.resample_image( img1, newspc )
img1up = ants.resample_image( img1lo, spc )
img1up = img2
ants.image_write( img1, '/tmp/or.nii.gz' )
ants.image_write( img2, '/tmp/orr.nii.gz' )
print('begin sr : psnr orig = ' + str( antspynet.psnr(img1,img1up) ) )
blw = 1
ct = 1
for k in range(len(mdlfns)):
    mdl = tf.keras.models.load_model( mdlfns[k], compile=False )
    ranger = [0,1]
    sr = antspynet.apply_super_resolution_model_to_image(
        img2,
        mdl,
        target_range=ranger, regression_order=1 )
    # ants.plot(sr)
    # sr = sr * blw + ants.iMath(sr,"Sharpen") * (1.0-blw)
    sr = sr * blw + img2 * (1.0-blw)
    sharp = ants.iMath(img2,"Sharpen")
    tempfn = '/tmp/dpr' + str(ct) + '.nii.gz'
    print( tempfn )
    ants.image_write( sr, tempfn )
    sharp = ants.iMath(img2,"Sharpen")
    tempfn = '/tmp/sharp.nii.gz'
    ants.image_write( sharp, tempfn )
    ct = ct + 1
    # some metrics on the output
    print("*****************************************")
    print(mdlfns[k])
    gmsdSR = antspynet.gmsd(img2,sr)
    gmsdBi = antspynet.gmsd(img1,img1up)
    ssimSR = antspynet.ssim(img2,sr)
    ssimBi = antspynet.ssim(img1,img1up)
    psnrSR = antspynet.psnr(img2,sr)
    psnrBi = antspynet.psnr(img1,img1up)
    print("PSNR Test: " + str( psnrBi ) + " vs SR: " + str( psnrSR ), flush=True  )
    print("GMSD Test: " + str( gmsdBi ) + " vs SR: " + str( gmsdSR ), flush=True  )
    print("ssim Test: " + str( ssimBi ) + " vs SR: " + str( ssimSR ), flush=True  )
    # ants.plot( img1 )
    # ants.plot( img1up )
    # ants.plot( sr )

if False:
    # now apply to "real" case
    srhi = antspynet.apply_super_resolution_model_to_image(
        img1,
        mdl,
        target_range=[0,1], regression_order=None )
    ants.image_write( srhi, '/tmp/srhi.nii.gz' )
