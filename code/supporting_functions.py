import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import shutil
import re
import requests
import json
from numba import njit
import time

def plot(img):
    plt.imshow(img)
    plt.show()


def scale_image(img):
    # we get an image that might be horisontal or vertical, and we want to rotate it so that it is always vertical
    rot90 = False # if we rotate the image 90 degrees, we keep track of it by sending it back to the caller

    # get the dimensions of the image
    height, width, channels = img.shape

    # if the image is horisontal, rotate it
    if width > height:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        rot90 = True

    new_width = 1000  
    new_height = 1400
    return cv2.resize(img, (new_width, new_height)), rot90


def gray_scale_image(image, threshold_value = 180):
    # returns a gray_scale version of the image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    _, bimg = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY) # better if we treshold it?
    return bimg


@njit
def get_y1_y2(img):
    y1, y2 = 0, 0
    x1, x2 = 200, 800
    for y in range(100, 1300):
        y_slice_1, y_slice_2 = img[y : y + 1, x1 : x1 + 3], img[y : y + 1, x2: x2 + 3]
        y_color_1, y_color_2 = np.sum(y_slice_1) / 3, np.sum(y_slice_2) / 3  # Calculate average without np.mean

        if not y1 and y_color_1 < 200:
            y1 = y
        if not y2 and y_color_2 < 200:
            y2 = y

        if y1 and y2:
            break

        if y > y1 + 25:
            y1 = 0

        if y > y2 + 25:
            y2 = 0

    return y1, y2

@njit
def find_xs(img):
    xs = np.zeros(1, dtype=np.int64)
    for x in range(20, 980, 1):  # for each x
        x_slice = img[200: 1200, x: x + 2]
        avg_col = np.sum(x_slice) / (1000 * 2)  # Calculate average without np.mean
        if avg_col < 450 and x > xs[-1] + 10:
            xs = np.append(xs, x + 2)  # offset to remove border as much as possible w/o remove time
    
    return xs


def orient_scorecard(img):
    '''
    Takes an image of a scanned scorecard (no surroundings)
    1) make sure the scorecard is oriented so the horisontal lines are completely horisontal
    2) flip the scorecard so it's oriented correctly (event on top and so on)
    Returns oriented scorecard
    '''
    
    # 1)
    org = img.copy()

    y1, y2 = get_y1_y2(img)
    x1, x2 = 200, 800
    
    # Calculate the angle between the line formed by (x1, y1) and (x2, y2) and the horizontal axis
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

    # Rotate the image by the calculated angle
    rows, cols, _ = org.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    org = cv2.warpAffine(org, rotation_matrix, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=[255,255,255])
    
    # 2)
    xs = find_xs(org)
    
    if len(xs)<6:
        return 0, False
    
    diffs = []
    for i in range(len(xs)-1):
        diffs.append(xs[i+1]-xs[i])
    
    if diffs[-1] == min(diffs):
        return cv2.rotate(org, cv2.ROTATE_180 ), True # keep track of the rotation
    return org, False # keep track of the rotation


@njit
def find_lines(img):
    # returns y of each horizontal black line, from top to bottom,
    # and x of each vertical black line, left to right
    ys = np.zeros(1, dtype=np.int64)
    for y in range(20, 1380):  # for each y
        y_slice = img[y: y + 5, 150: 850]
        avg_col = np.sum(y_slice) / (5 * 700)  # Calculate average without np.mean
        if avg_col < 250 and y > ys[-1] + 10:
            ys = np.append(ys, y)

    xs = np.zeros(1, dtype=np.int64)
    for x in range(20, 980, 1):  # for each x
        x_slice = img[200: 1200, x: x + 2]
        avg_col = np.sum(x_slice) / (1000 * 2)  # Calculate average without np.mean
        if avg_col < 450 and x > xs[-1] + 10:
            xs = np.append(xs, x + 2)  # offset to remove border as much as possible w/o remove time

    return ys[1:], xs[1:]


# Get info boxes

@njit
def find_ID_top_left_corner(img, x0=150, y0=125):
    ys = np.zeros(1, dtype=np.int64)
    for y in range(20, 1380):  # for each y
        y_slice = img[y: y + 5, 150: 850]
        avg_col = np.sum(y_slice) / (5 * 700)  # Calculate average without np.mean
        if avg_col < 435 and y > ys[-1] + 10:
            ys = np.append(ys, y)

    xs = np.zeros(1, dtype=np.int64)
    for x in range(20, 980, 1):  # for each y
        x_slice = img[450: 1000, x: x + 2]
        avg_col = np.sum(x_slice) / (550 * 2)  # Calculate average without np.mean
        if avg_col < 500 and x > xs[-1] + 10:
            xs = np.append(xs, x + 2)  # offset to remove border as much as possible w/o remove time

    return xs[1], ys[1]


def info_boxs_from_top_left_coordinate(img, x, y):
    boxes = []
    
    box_coords = [
        # [x0, x1, y0, y1]
        [x+170,x+650,y-110,y-48], # event
        [x+50,x+400,y-50,y-7], # round
        [x+10,x+126,y+10,y+65], # ID
    ]
    
    for coords in box_coords:
        x0,x1,y0,y1 = coords[0],coords[1],coords[2],coords[3]
        boxes.append(img[y0:y1,x0:x1])
    
    return boxes

def scorecard_to_boxes(scorecard, ys, xs):
    '''
    scorecard oriented and 1000x1400
    ys - one y for each horisontal line on the scorecard
    '''
    
    x = xs[2] # + 510
    y_lines = [3,4,6,7,8,10]
    
    boxes = [] #fill with images of times
    for yi in y_lines:
        boxes.append(scorecard[ys[yi] : ys[yi] + 110, x : x + 490])
    return boxes

def resize_info_boxes(info_boxes,boxes):
    new_height = boxes[0].shape[0]
    for i in range(len(info_boxes)):
        width = info_boxes[i].shape[1]
        info_boxes[i] = cv2.resize(info_boxes[i], (width, new_height))
    return info_boxes

def stack_images_with_border(info_boxes,boxes, L = 125):
    # Create a white border
    dimensions = boxes[0].shape
    
    border = np.ones((dimensions[0], L, 3), dtype=np.uint8) * 255

    # Stack the images horizontally with borders
#     print(info_boxes[0].shape,boxes[0].shape)
    result = np.hstack([info_boxes[0],border,info_boxes[1],border,info_boxes[2],border,boxes[0], 
                        border, boxes[1], border, boxes[2], border,
                        boxes[3], border, boxes[4], border, boxes[5]])
    return result

def scorecard_to_horisontal(scorecard_path, compress_original = True, L = 125):
    # returns a horisontal image of the important parts of the scorecard
    # if the scorecard is blank, it returns 0
    # if compress is True, the original image is rotated according to the rotation of the scorecard (which makes double checking more convenient) and reduces the size to 500x700 (half the size)

    img = cv2.imread(scorecard_path)
    img = gray_scale_image(img)
    img, rot90 = scale_image(img)
    height, width, channels = img.shape
    
    # check if backside (average color is white)
    if np.sum(np.mean(img, axis=(0, 1))) > 715:
        return 0, 0
    
    img, rot180 = orient_scorecard(img)
    if isinstance(img,int):
        return 0, 0
    
    if compress_original:
        org = img.copy()
        # rotate the original image according to rot90 and rot180
        if rot90:
            org = cv2.rotate(org, cv2.ROTATE_90_CLOCKWISE)
        if rot180:
            org = cv2.rotate(org, cv2.ROTATE_180)
        
        # check the size of the original image
        height, width, channels = org.shape
        if height > 700 or width > 500:
            # resize the original image
            org = cv2.resize(org, (500, 700), interpolation=cv2.INTER_CUBIC * 0.8)

        # save the original image again
        cv2.imwrite(scorecard_path, org)
    
    ys, xs = find_lines(img)
    
    if len(ys) != 12:
        return 0, 0
    
    boxes = scorecard_to_boxes(img, ys, xs)

    for b in boxes:
        if len(b)==0:
            return 0, 0
    
    x, y = find_ID_top_left_corner(img)
    info_boxes = info_boxs_from_top_left_coordinate(img, x, y)
    info_boxes = resize_info_boxes(info_boxes, boxes)
    himg = stack_images_with_border(info_boxes, boxes, L)
    
    if np.sum(np.mean(himg[:,:500], axis=(0, 1))) > 700: # if event is blank (white), in case wrongly oriented or something wrong
        return 0, 0

    # resize to 110 x 5000 so we know the size of the image
    himg = cv2.resize(himg, (5000, 110))

    # we also want to return x for the right side of the results boxes, and y for the top of the results boxes
    if rot180:
        # reverse xs and ys
        xs = np.flip(xs)
        ys = np.flip(ys)
    
    # we want to scale the coordinates to the original image
    xs = xs / 1000 * width
    ys = ys / 1400 * height

    top_right_coords = {
        "x": xs[-1],
        "ys": [ys[3], ys[4], ys[6], ys[7], ys[8], ys[10]]
    }

    return himg, top_right_coords

def stack_ID_images(ID_images):
    return np.vstack(ID_images)

def stack_n_images(n, line_identifier = "<!>"):
    # stacks images n times
    img = np.ones((210, 325, 3), dtype=np.uint8) * 255
    img = cv2.putText(img, line_identifier, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5, cv2.LINE_AA)
    imgs = []
    for i in range(n):
        imgs.append(img)
    return np.vstack(imgs)


def get_all_image_paths(root_path):
    jpg_images = []

    # Get all .jpg files in subfolders
    for foldername, subfolders, filenames in os.walk(root_path):
        jpg_images.extend(glob.glob(os.path.join(foldername, '*.jpg')))

    return jpg_images