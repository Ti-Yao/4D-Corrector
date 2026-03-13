import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import os
from unet3plus_4D import *
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import time
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from skimage import exposure
from rasterio import features, Affine
from skimage.measure import label 
from shapely.geometry import Polygon, box, Point, shape, MultiPolygon

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def load_font(size):
    # Try Linux font
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except:
        pass
    # Try Windows font
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except:
        pass
    # Fallback (non scalable)
    return ImageFont.load_default()

def load_nii(nii_path):
    file = nib.load(nii_path)
    data = file.get_fdata(caching='unchanged')
    return data

def save_mask(mask, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    nib_mask = nib.Nifti1Image(mask.astype(np.uint8), affine=np.eye(4))
    nib.save(nib_mask, save_path)

def find_crop_box(mask, crop_factor):
    # Check shape of the input is 2D
    if len(mask.shape) != 2:
        raise ValueError("Input mask must be a 2D array")
    
    y = np.sum(mask, axis=1) # sum the masks across columns of array, returns a 1D array of row totals
    x = np.sum(mask, axis=0) # sum the masks across rows of array, returns a 1D array of column totals

    top = np.min(np.nonzero(y)) - 1 # Returns the indices of the elements in 1d row totals array that are non-zero, then finds the minimum value and subtracts 1 (i.e. top extent of mask)
    bottom = np.max(np.nonzero(y)) + 1 # Returns the indices of the elements in 1d row totals array that are non-zero, then finds the maximum value and adds 1 (i.e. bottom extent of mask)

    left = np.min(np.nonzero(x)) - 1 # Returns the indices of the elements in 1d column totals array that are non-zero, then finds the minimum value and subtracts 1 (i.e. left extent of mask)
    right = np.max(np.nonzero(x)) + 1 # Returns the indices of the elements in 1d column totals array that are non-zero, then finds the maximum value and adds 1 (i.e. right extent of mask)
    if abs(right - left) > abs(top - bottom):
        largest_side = abs(right - left) # Find the largest side of the bounding box
    else:
        largest_side = abs(top - bottom)
    x_mid = round((left + right) / 2) # Find the mid-point of the x-length of mask
    y_mid = round((top + bottom) / 2) # Find the mid-point of the y-length of mask
    half_largest_side = round(largest_side * crop_factor / 2) # Find half the largest side of the bounding box (crop factor scales the largest side to ensure whole heart and some surrounding is captured)
    x_max, x_min = round(x_mid + half_largest_side), round(x_mid - half_largest_side) # Find the maximum and minimum x-values of the bounding box
    y_max, y_min = round(y_mid + half_largest_side), round(y_mid - half_largest_side) # Find the maximum and minimum y-values of the bounding box
    if x_min < 0:
        x_max -= x_min # if x_min less than zero, expand the x_max value by the absolute value of x_min, to ensure bounding box is same size
        x_min = 0

    if y_min < 0:
        y_max -= y_min # if y_min less than zero, expand the y_max value by the absolute value of y_min, to ensure bounding box is same size
        y_min = 0

    return [x_min, y_min, x_max, y_max]

def clip_outliers(img, lower_percentile=1, upper_percentile=99):
    lower = np.percentile(img, lower_percentile)
    upper = np.percentile(img, upper_percentile)
    clipped_img = np.clip(img, lower, upper)
    return clipped_img


def standardize(x):
    x = x - np.mean(x)
    x = x / np.std(x)
    return x

def get_one_hot(targets, nb_classes):
    '''
    One hot encode segmentation mask
    '''
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def transpose_channels(image, input_channel_order, target_channel_order):
    """
    Given input and target channel orders, return the transpose order.

    Args:
        input_order (list or str): Current order of axes, e.g., ['H','W','D','T','C']
        target_order (list or str): Desired order of axes, e.g., ['D','T','H','W','C']

    Returns:
        list: Indices to use in np.transpose to convert input_order to target_order
    """
    return np.transpose(image, [input_channel_order.index(axis) for axis in target_channel_order])


def make_video(true_image, true_mask, pred_image, pred_mask, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    H, W, position, timesteps = true_image.shape

    grid_rows = int(np.sqrt(position) + 0.5)
    grid_cols = (position + grid_rows - 1) // grid_rows

    # -------- FIXED WIDTH --------
    canvas_w = 1200
    gap = 20
    top_margin = 40
    # -----------------------------

    tile_w = (canvas_w - gap) // (grid_cols * 2)
    tile_h = int(tile_w * H / W)  # preserve aspect ratio

    canvas_h = grid_rows * tile_h + top_margin

    frames = []
    font = load_font(16)

    true_min, true_max = true_image.min(), true_image.max()
    pred_min, pred_max = pred_image.min(), pred_image.max()

    def gray_to_rgb(img, vmin, vmax):
        img = (img - vmin) / (vmax - vmin + 1e-8)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return np.stack([img]*3, axis=-1)

    def overlay_mask(base, mask, color):
        base = base.astype(np.float32)
        mask = (mask > 0).astype(np.float32)[..., None]
        color = np.array(color, dtype=np.float32)

        alpha = 0.3
        blended = base * (1 - alpha * mask) + color * (alpha * mask)

        return np.clip(blended, 0, 255).astype(np.uint8)

    for t in range(timesteps):

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for pos in range(position):

            row = pos // grid_cols
            col = pos % grid_cols

            y = row * tile_h + top_margin
            x_left = col * tile_w
            x_right = col * tile_w + grid_cols * tile_w + gap

            # TRUE
            img = gray_to_rgb(true_image[:, :, pos, t], true_min, true_max)
            img = overlay_mask(img, true_mask[:, :, pos, t, 1], [0,255,255])
            img = overlay_mask(img, true_mask[:, :, pos, t, 2], [255,0,255])

            img = Image.fromarray(img).resize((tile_w, tile_h), Image.NEAREST)
            canvas[y:y+tile_h, x_left:x_left+tile_w] = np.array(img)

            # PRED
            img = gray_to_rgb(pred_image[:, :, pos, t], pred_min, pred_max)
            img = overlay_mask(img, pred_mask[:, :, pos, t, 1], [0,255,255])
            img = overlay_mask(img, pred_mask[:, :, pos, t, 2], [255,0,255])

            img = Image.fromarray(img).resize((tile_w, tile_h), Image.NEAREST)
            canvas[y:y+tile_h, x_right:x_right+tile_w] = np.array(img)

        frame = Image.fromarray(canvas)
        draw = ImageDraw.Draw(frame)

        draw.text((canvas_w*0.25, 15), "2D Model", fill="white", anchor="mm", font=font)
        draw.text((canvas_w*0.75, 15), "4D Model", fill="white", anchor="mm", font=font)

        draw.text((canvas_w*0.5, 15),
                  f"Timestep = {t+1}/{timesteps}",
                  fill="white",
                  anchor="mm",
                  font=font)

        frames.append(frame)

    frames[0].save(
        f"{save_path}.gif",
        save_all=True,
        append_images=frames[1:],
        duration=int(1000/timesteps),
        loop=0
    )

def mask_to_polygons_layer(mask):
    '''
    convert mask to polygons
    '''
    all_polygons = []
    for poly, value in features.shapes(mask.astype(np.int16), mask=(mask >0), transform= Affine(1.0, 0, 0, 0, 1.0, 0)):
        all_polygons.append(shape(poly))

    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        if all_polygons.geom_type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons

def remove_unconnected_segmentations(masks):
    '''
    postprocessing step, clean up any segmentations outside the heart (the main connected layer)
    '''
    
    keep_seg_masks = []
    mid_slice_idx = round(masks.shape[2]/2)
    for channel in range(1,3):
        keep_seg_masks_time = []
        for time in range(masks.shape[3]):
            heart_mask = np.sum(np.sum(np.sum(masks[...,time,-2:],-1),0),0)
            max_heart_idx = np.argmax(heart_mask)
            y_sum = np.sum(masks[...,max_heart_idx,time,1:],-1)
            sum_polygon = mask_to_polygons_layer(y_sum)
            keep_seg_masks_slice = []
            for pos in range(masks.shape[2]):
                pred_polygons = mask_to_polygons_layer(masks[...,pos,time,channel])
                keep_pred_polygons = []
                for pred_poly in list(pred_polygons.geoms):
                    current_poly = pred_poly
                    criteria = current_poly.intersects(sum_polygon) 
                    if criteria :
                        keep_pred_polygons.append(pred_poly)
                if len(keep_pred_polygons) > 0:
                    keep_seg_masks_slice.append(features.rasterize(keep_pred_polygons, out_shape=y_sum.shape))
                else:
                    keep_seg_masks_slice.append(np.zeros_like(masks[...,pos,time,channel]))
            keep_seg_masks_slice = np.stack(keep_seg_masks_slice,-1)
            keep_seg_masks_time.append(keep_seg_masks_slice)
        keep_seg_masks_time = np.stack(keep_seg_masks_time,-1)
        keep_seg_masks.append(keep_seg_masks_time)

    keep_seg_masks = np.stack(keep_seg_masks,-1)
    bkg_mask_array = np.ones(keep_seg_masks.shape[:4]) - np.sum(keep_seg_masks[...,1:], axis = -1)
    masks = np.concatenate([bkg_mask_array[...,np.newaxis],keep_seg_masks], axis = -1)
    return masks