#!/usr/bin/python

# Author : Sudnya Padalikar
# Date : 01/05/2014
# A script to apply gaussian blur to an image (as explained in the book "Programming Computer Vision with Python" by Jan Erik Solem)

import argparse
from PIL import Image
from pylab import *
from numpy import *
from scipy.ndimage import filters
import scipy.misc

def gaussian_blur(filename):
    # L converts color to grayscale
    original = array(Image.open(filename))#.convert('L'))

    blurred = {}
    for i in range(10):
        blurred[i] = zeros(original.shape)
        # handles R,G,B
        for j in range(3):
            blurred[i][:,:,j] =  filters.gaussian_filter(original[:,:,j],i)
        # 8 bit representation ensures colors are in default format (removing this line causes strange shades in image)
        blurred[i] = uint8(blurred[i])
        imshow(blurred[i])
        show()


def main():
    parser = argparse.ArgumentParser(description="Process commandline inputs")
    parser.add_argument('-i', help="input file path (including file name, extension)", type=str)
    args = parser.parse_args()
    gaussian_blur(args.i)


if __name__ == '__main__':
    main()

