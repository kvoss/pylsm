#!/usr/bin/env python

# based on an example from:
# http://www.gimp.org/docs/python/index.html
#
# http://developer.gimp.org/plug-ins.html

from gimpfu import *

#import lsmethod.segment as lsm
#import lsmethod.load_file as load_file

def lsmcyto(image, drawable, upsilon, mu, eta, t_f, time_step, mask_size):
    pass

def asklsd():
    width = tdrawable.width
    height = tdrawable.height

    rgn = tdrawable.get_pixel_rgn(0, 0, width, height)
    print width, height

    #solns = segment(pic)
    #layer_one = gimp.Layer(timg, "X Dots", width, height, RGB_IMAGE,
                           100, NORMAL_MODE)
    #timg.add_layer(layer_one, 0)
    pass

register(
    "python-fu-lsmcyto",
    N_("Adds a layer of recognized cytoplasm"),
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
    pythonlsm,
    menu="<Image>/Filters/Segment",
    domain=("gimp20-python", gimp.locale_directory)
    )
#register(
#        "pythonlsm",
#        "Selects uniform region",
#        "Selects uniform region",
#        "Krzysztof Voss",
#        "GNU",
#        "2011-2012",
#		"My Plugin",
#        "*",
#        [
#            (PF_IMAGE, "timg", "Input image"),
#            (PF_DRAWABLE, "tdrawable", "Input drawable"),
#        (PF_STRING, "name",       _("_Layer name"), _("Clouds")),
#        (PF_COLOUR, "colour",     _("_Fog color"),  (240, 180, 70)),
#        (PF_SLIDER, "turbulence", _("_Turbulence"), 1.0, (0, 10, 0.1)),
#        (PF_SLIDER, "opacity",    _("Op_acity"),    100, (0, 100, 1)),
#        ],
#        [],
#        pythonlsm, menu="<Image>/Filters/Python-Fu/LASKDJ")

main()

