"""
Make updated body shape from updated segmentation
"""

import os
import numpy as np
import cv2
from PIL import Image
import sys


(cv_major, _, _) = cv2.__version__.split(".")
if cv_major != '4' and cv_major != '3':
    print('doesnot support opencv version')
    sys.exit()


# @TODO this is too simple and pixel based algorithm
def body_detection(image, seg_mask):
    # binary thresholding by blue ?
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0, 0, 120])
    upper_blue = np.array([180, 38, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(image, image, mask=mask)

    # binary threshold by green ?
    b, g, r = cv2.split(result)
    filter = g.copy()
    ret, mask = cv2.threshold(filter, 10, 255, 1)

    # at least original segmentation is FG
    mask[seg_mask] = 1

    return mask


def make_body_mask(img_file, seg_file):

    # Load images
    img_dir = "data/test/image/"
    img = cv2.imread(img_dir + img_file) # image/***.png
    # segm = Image.open(seg_pth)
    # the png file should be 1-ch but it is 3 ch ^^;
    
    #image parse new folder
    seg_dir = "data/test/image-parse-new/"
    #seg_file = "/home/vinod/D2B/tryon_api/data/test/image-parse-new/000010_0.png"
    gray = cv2.imread(seg_dir + seg_file, cv2.IMREAD_GRAYSCALE)
    _, seg_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    body_mask = body_detection(img, seg_mask)
    body_mask = body_mask + seg_mask
    body_mask[seg_mask] = 1
    
    # write image mask
    mask_path = "data/test/image-mask/" + img_file.replace(".jpg", ".png")
    cv2.imwrite(mask_path, body_mask)
    return True    
    
