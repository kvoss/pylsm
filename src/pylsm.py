#!/usr/bin/env python

# based on an example from:
# http://www.gimp.org/docs/python/index.html
#
# http://developer.gimp.org/plug-ins.html
from gimpfu import *

import numpy as np
import matplotlib.pyplot as plt

import lsmethod as lsm

import logging
logging.basicConfig(filename='/home/kvoss/pylsm.log', level=logging.DEBUG)

def segment_image(timg, tdraw, upsilon, mu, eta, t_f):
    #logging.info(upsilon, mu, eta, t_f)
    #x1, y1, x2, y2 = tdraw.mask_bounds
    pr = tdraw.get_pixel_rgn(0, 0, tdraw.width, tdraw.height, False, False)
    original_img = np.fromstring(pr[:,:], dtype=np.uint8)
    if tdraw.bpp > 1:
        img = original_img.reshape(tdraw.height, tdraw.width, tdraw.bpp)
    else:
        img = original_img.reshape(tdraw.height, tdraw.width)
    #logging.info(np.shape(img))
    #plt.imsave('babelek.png', img)
    #logging.info('img saved..')

    #mymask = img[:,:,0]
    #thrd = np.mean(mymask)
    #mymask = mymask > thrd

    lsm.t_f = float(t_f)
    sln    = lsm.segment_img(img)
    mymask = lsm.extract_cytoplasm(sln)
    mymask = np.array(mymask, dtype=np.uint8)
    mymask *= 255
    #print mymask[4:14,6]
    #print np.max(mymask)

    #print type(mymask.tostring()), np.size(mymask.tostring())
    #ch = gimp.Channel(timg, 'lsm selection', tdraw.width, tdraw.height, 50, 2)
    ch = pdb.gimp_selection_save(timg)
    chpr = ch.get_pixel_rgn(0, 0, ch.width, ch.height, True, False)
    #logging.info(str(chpr[:,:]))
    #logging.info(str(np.shape(chpr)))
    chpr[:,:] = mymask.tostring()
    pdb.gimp_selection_load(ch)
    timg.remove_channel(ch)
    pdb.gimp_displays_flush()

    #plt.imshow(mymask)
    #plt.show()
    
def run_plugin():
    # zebrac pixele: drawable.get_pixel_rgn(   ,True, True)
    # uruchomic na uzyskanych danych lsm
    # zaznaczyc maske poprzez img.selection.set_pixel(

    # progress = 0
    # max_progress = 100
    # gimp.progress_init("selecting region")
    # gimp.progress_update(1)

    pass

register(
        "python-fu-lsmcyto",
        "Selects cytoplasm",
        "Segment cytoplasm using level set method",
        "Krzysztof Voss",
        "Krzysztof Voss",
        "2011-2012",
        "<Image>/Select/Cytoplams/Level Set Method",
        "RGB*, GRAY*",
        [
            (PF_FLOAT, "upsilon", "arc smoothing", 0.1),
            (PF_FLOAT, "mu", "LS regularity", 0.1),
            (PF_FLOAT, "eta", "cytoplasm marker", 0.0),
            (PF_FLOAT, "t_f", "duration", 0.5)
        ],
        [],
        segment_image
        )

main()

