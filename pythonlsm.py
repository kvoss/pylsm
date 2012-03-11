#!/usr/bin/env python

# based on an example from:
# http://www.gimp.org/docs/python/index.html
#
# http://developer.gimp.org/plug-ins.html

from gimpfu import *

#import lsmethod.segment as lsm
#import lsmethod.load_file as load_file

def python_lsm(timg, tdrawable, upsilon, mu, eta, t_f, time_step, mask_size):
    width = tdrawable.width
    height = tdrawable.height

    rgn = tdrawable.get_pixel_rgn(0, 0, width, height)
    print width, height

    #solns = segment(pic)
    #layer_one = gimp.Layer(timg, "X Dots", width, height, RGB_IMAGE,
                           100, NORMAL_MODE)
    #timg.add_layer(layer_one, 0)
    return


register(
        "pythonlsm",
        "Selects uniform region",
        "Selects uniform region",
        "Krzysztof Voss",
        "Krzysztof Voss",
        "2011-2012",
		"<Image>/Filters/Python-Fu/Segment/LSMMethod",
        "RGB*, GRAY*",
        [
            (PF_FLOAT, "upsilon", "arc smoothing", 0.1),
            (PF_FLOAT, "mu", "LS regularity", 0.1),
            (PF_FLOAT, "eta", "cytoplasm marker", 0.0),
            (PF_FLOAT, "t_f", "duration", 1.5),
            (PF_FLOAT, "time_step", "delta t", 0.1),
            (PF_INT, "mask_size", "mask size", 5)
        ],
        [],
        python_lsm)

main()

