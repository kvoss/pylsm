#!/usr/bin/env python

# based on an example from:
# http://www.gimp.org/docs/python/index.html

import math
from gimpfu import *

import lsmethod.segment as lsm
import lsmethod.load_file as load_file

def python_lsm(timg, tdrawable,
                ):
    width = tdrawable.width
    height = tdrawable.height

    img = gimp.Image(width, height, RGB)
    img.disable_undo()

    fn = 'dupa.png'
    pic = load_file(fn)

    solns = segment(pic)

    layer_one = gimp.Layer(img, "X Dots", width, height, RGB_IMAGE,
                           100, NORMAL_MODE)
    img.add_layer(layer_one, 0)
    pdb.gimp_edit_fill(layer_one, BACKGROUND_FILL)

    pdb.plug_in_noisify(img, layer_one, 0, 0.7, 0.7, 0.7, 0.7)

    img.flatten()

    gimp.delete(layer_one)
    gimp.delete(img)




def python_clothify(timg, tdrawable, bx=9, by=9,
                    azimuth=135, elevation=45, depth=3):
    width = tdrawable.width
    height = tdrawable.height

    img = gimp.Image(width, height, RGB)
    img.disable_undo()

    layer_one = gimp.Layer(img, "X Dots", width, height, RGB_IMAGE,
                           100, NORMAL_MODE)
    img.add_layer(layer_one, 0)
    pdb.gimp_edit_fill(layer_one, BACKGROUND_FILL)

    pdb.plug_in_noisify(img, layer_one, 0, 0.7, 0.7, 0.7, 0.7)

    layer_two = layer_one.copy()
    layer_two.mode = MULTIPLY_MODE
    layer_two.name = "Y Dots"
    img.add_layer(layer_two, 0)

    pdb.plug_in_gauss_rle(img, layer_one, bx, 1, 0)
    pdb.plug_in_gauss_rle(img, layer_two, by, 0, 1)

    img.flatten()

    bump_layer = img.active_layer

    pdb.plug_in_c_astretch(img, bump_layer)
    pdb.plug_in_noisify(img, bump_layer, 0, 0.2, 0.2, 0.2, 0.2)
    pdb.plug_in_bump_map(img, tdrawable, bump_layer, azimuth,
                         elevation, depth, 0, 0, 0, 0, True, False, 0)

    gimp.delete(img)

register(
        "python_fu_lsm",
        "Selects uniform region",
        "Selects uniform region",
        "Krzysztof Voss",
        "Krzysztof Voss",
        "2011-2012",
        "<Image>/Filters/Biology/_LSMSegmentation...",
        "RGB*, GRAY*",
        [
                (PF_FLOAT, "time_step", "delta t", 0.1),
                (PF_FLOAT, "time_end", "duration", 2.0),
                (PF_INT, "mask_size", "mask size", 5)
        ],
        [],
        python_lsm)

main()




