#!/usr/bin/env python

# based on an example from:
# http://www.gimp.org/docs/python/index.html
#
# http://developer.gimp.org/plug-ins.html

from gimpfu import *
import gimp
import numpy as np

#import lsmethod.segment as segment
#import lsmethod.load_file as load_file

def mylog(txt):
    f = open('dupa.8', 'w+')
    f.write(txt)
    f.close()


def lsmcyto(image, drawable, upsilon, mu, eta, t_f, time_step, mask_size):
    """The actual plugin implementation
    """
    # sets selection acordingly to calculated curve

    width = drawable.width
    height = drawable.height

    #img = gimp.Image(width, height, RGB)
    layer_one = gimp.Layer(image, "MyLayer", width, height, GRAY_IMAGE, 10.0, NORMAL_MODE)
    image.add_layer(layer_one, 1)

    pdb.gimp_edit_fill(layer_one, BACKGROUND_FILL)

    pr = drawable.get_pixel_rgn(0,0, width, height)
    print np.array(pr[:,:])

    mylog("[!!] layer added")

    #solns = segment(rgn)
    #print "hello"
    #print width, height

    #import numpy as np
    #sel = (np.abs(solns[-1]) < threshold).reshape(w,h)
    #s = gimp.Selection(sel)
    #gimp.set_selection(drawable)



register(
    "python-fu-lsmcyto",
    N_("Selects cytoplasm"),
    "Adds a layer of recognized cytoplasm.",
    "Krzysztof Voss",
    "Krzysztof Voss",
    "2011,2012",
    N_("_Cytoplasm..."),
    "RGB*, GRAY*",
    [
        (PF_IMAGE, "image",       "Input image", None),
        (PF_DRAWABLE, "drawable", "Input drawable", None),

        (PF_FLOAT, "upsilon", "arc smoothing", 0.1),
        (PF_FLOAT, "mu", "LS regularity", 0.1),
        (PF_FLOAT, "eta", "cytoplasm marker", 0.0),
        (PF_FLOAT, "t_f", "duration", 1.5),
        (PF_FLOAT, "time_step", "delta t", 0.1),
        (PF_INT, "mask_size", "mask size", 5)
    ],
    [],
    lsmcyto, menu="<Image>/Filters/Python-Fu"
    )

main()

