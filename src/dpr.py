import os
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"
import re
import numpy as np
import tensorflow as tf
import glob
import ants
import antspynet
import antspyt1w
import shutil

# get some data
image = ants.image_read( "data/blocki.nii.gz")
p=(8,8,8)
image = ants.iMath(image,"Normalize").pad_image(  pad_width=[p,p,p] )
seg = ants.threshold_image( image, 'Otsu', 3 )
mdlfn='models/CSRS.R.TV.D.Res6.h5'

# make some transforms
rRotGenerator = ants.contrib.RandomRotate3D( ( -2, 2 ), reference=image )
tx0 = rRotGenerator.transform()
tx0inv = ants.invert_ants_transform(tx0)

# do SR first
srmdl = tf.keras.models.load_model( mdlfn )
ssr = antspyt1w.label_and_img_to_sr(
            image,
            seg,
            srmdl,
            return_intensity=True )
rimg = tx0.apply_to_image( ssr['super_resolution'] )
rimgrSR1 = tx0inv.apply_to_image( rimg )
dprimgSR1 = ants.resample_image_to_target( rimgrSR1, image, interp_type='nearestNeighbor' )
def dp( x ):
    return ants.pad_image( x, pad_width=(-8,-8,-8 ) )

dprsnr1 = antspynet.psnr( dp(image), dp(dprimgSR1 ))
dprssim1 = antspynet.ssim(dp(image), dp(dprimgSR1 ))


# do SR last
rimg = tx0.apply_to_image( image )
rimgr = tx0inv.apply_to_image( rimg )
seg2 = ants.threshold_image( rimgr, 'Otsu', 3 )
ssr2 = antspyt1w.label_and_img_to_sr(
            rimgr,
            seg2,
            srmdl,
            return_intensity=True )
rimgrSR2 = ssr2['super_resolution']
dprimgSR2 = ants.resample_image_to_target( rimgrSR2, image, interp_type='nearestNeighbor' )
dprsnr2 = antspynet.psnr(dp(image), dp(dprimgSR2 ))
dprssim2 = antspynet.ssim(dp(image), dp(dprimgSR2 ))

# dont do any SR
dprsnr3 = antspynet.psnr(dp(image), dp(rimgr ))
dprssim3 = antspynet.ssim(dp(image), dp(rimgr ))

ants.image_write( image, '/tmp/ground_truth.nii.gz')
ants.image_write( rimgr, '/tmp/rimgr.nii.gz')
ants.image_write( dprimgSR1, '/tmp/dprimgSR1.nii.gz')
ants.image_write( dprimgSR2, '/tmp/dprimgSR2.nii.gz')

print( str(dprsnr1) + " " + str(dprsnr2) + " " + str(dprsnr3) )
print( str(dprssim1) + " " + str(dprssim2) + " " + str(dprssim3) )
