import cv2
import numpy as np
import random
import mtutils as mt
import math


def sub_imgs(img, name):
    return [[img[:,:,i], name + '_' + str(i)] for i in range(3)]

def rand_pos_2D(data):
    H, W = data.shape[:2]
    x = random.randint(0, W)
    y = random.randint(0, H)
    return x, y

def rand_shake_2D(data, x, y, var):
    H, W = data.shape[:2]
    new_x = x + random.randint(-var, var)
    new_y = y + random.randint(-var, var)
    new_x = min(max(0, new_x), W)
    new_y = min(max(0, new_y), H)
    return new_x, new_y

def make_gaussian_points(data, x, y):
    H, W = data.shape[:2]
    sigma = random.randint(5, 30)
    size = max(H, W)
    T = random.randint(30, 130)

    Xs, Ys = np.meshgrid(np.arange(size * 2), np.arange(size * 2))
    dis_mat = ((Xs - size) ** 2 + (Ys - size) ** 2) ** 0.5
    cos_mask = (np.cos(dis_mat / T * 2 * np.pi) + 1) / 2

    img = mt.gaussian_mask_2d(size * 2, sigma)
    bbox = [size - x, size - y, size - x + W, size - y + H]
    crop = mt.crop_data_around_boxes(img * cos_mask, bbox)
    return crop

def change_color(img, mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H = hsv[:,:,0] * 1.0
    H += mask * random.randint(-70, 120)
    # H = (H.astype('int') % 256).astype('uint8')
    H = np.clip(H, 0, 255).astype('uint8')

    S = hsv[:,:,1] * 1.0
    S += mask * random.randint(-80, 80)
    # S = (S.astype('int') % 256).astype('uint8')
    S = np.clip(S, 0, 255).astype('uint8')

    V = hsv[:,:,2] * 1.0
    V += mask * random.randint(-30, 30)
    # S = (S.astype('int') % 256).astype('uint8')
    V = np.clip(V, 0, 255).astype('uint8')
    
    hsv[:,:,0] = H
    hsv[:,:,1] = S
    hsv[:,:,2] = V

    bac = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return bac

def add_rainbow(img):
    mask = np.zeros(img.shape[:2])
    base_pos = rand_pos_2D(img)
    gauss_img = make_gaussian_points(img, *base_pos)
    mask += gauss_img
    res = change_color(img, gauss_img)
    for _ in range(random.randint(3, 9)):
        new_pos = rand_shake_2D(img, *base_pos, random.randint(3, 55))
        gauss_img = make_gaussian_points(img, *new_pos)
        mask += gauss_img
        res = change_color(res, gauss_img)
    bbox = mt.get_xyxy_from_mask(mask > 0.05)
    return res, bbox
