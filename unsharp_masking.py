#!/usr/bin/python

# Author : Sudnya Padalikar
# Date : 01/05/2014
# A script to apply unsharp masking to an image (as explained on http://en.wikipedia.org/wiki/Unsharp_masking )

import argparse
from PIL import Image
from pylab import *
from numpy import *
from scipy.ndimage import filters
import scipy.misc

def unsharp_masking(filename):
    # L converts color to grayscale
    original = array(Image.open(filename))#.convert('L'))
    blurred = {}
    negative = {}
    for i in range(10):
        # 8 bit representation ensures colors are in default format (removing this line causes strange shades in image)
        blurred[i] = zeros(original.shape, 'uint8')
        negative[i] = zeros(original.shape, 'uint8')
        # handles R,G,B
        for j in range(3):
            blurred[i][:,:,j]  = filters.gaussian_filter(original[:,:,j],i)
            negative[i][:,:,j] = 255-original[:,:,j]
        # using basic additive function to combine for now
        imshow(blurred[i]+negative[i])
        show()


def main():
    parser = argparse.ArgumentParser(description="Process commandline inputs")
    parser.add_argument('-i', help="input file path (including file name, extension)", type=str)
    args = parser.parse_args()
    unsharp_masking(args.i)


if __name__ == '__main__':
    main()


