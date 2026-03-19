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
from scipy import ndimage
from skimage.measure import label   

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

    if true_mask.shape[-1] !=3 and true_mask.ndim != 5:
        true_mask = get_one_hot(true_mask.astype('uint8'), 3)
    if pred_mask.shape[-1] !=3 and pred_mask.ndim != 5:
        pred_mask = get_one_hot(pred_mask.astype('uint8'), 3)

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


def calculate_sax_metrics(mask_4d, voxel_size):
    mask_4d = get_one_hot(mask_4d.astype('uint8'), 3)
    
    myo_index = 1 # fixed
    endo_index = 2 # fixed

    masses = np.sum(mask_4d[...,myo_index], axis = (0,1,2)) * voxel_size
    volume = np.sum(mask_4d[...,endo_index], axis = (0,1,2)) * voxel_size  * 1.05

    dia_idx = np.argmax(volume)
    sys_idx = np.argmin(volume)
    mass = masses[dia_idx]
    edv = volume[dia_idx]
    esv = volume[sys_idx]
    sv = edv - esv
    ef = (sv) * 100/edv

    return volume, mass, esv, edv, sv, ef


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC


def postprocess(mask):
    """
    mask: (H,W,Z,T) predicted mask for one channel (non-binary)
    sum_mask: (H,W) reference mask (2D) for max heart area
    Returns cleaned mask with only values from components intersecting sum_mask
    """
    one_hot_mask = get_one_hot(mask.astype(np.uint8), 3) 
    sum_mask = np.sum(one_hot_mask[...,1:], axis = (-1, 2, 3))
    sum_mask = sum_mask > (np.quantile(sum_mask, 0.95)).astype(int)
    sum_mask = getLargestCC(sum_mask)
    H, W, Z, T = mask.shape
    keep_mask_time = []

    for t in range(T):
        keep_mask_slice = []

        for z in range(Z):
            slice_mask = mask[..., z, t]

            # create boolean mask of nonzero values
            nonzero = slice_mask != 0

            # label connected components on nonzero values
            labeled, n = ndimage.label(nonzero)

            # find labels that intersect the reference mask
            touching_labels = np.unique(labeled[sum_mask > 0])
            if touching_labels.size == 0:
                keep_mask_slice.append(np.zeros_like(slice_mask))
                continue

            # keep original values for touching components
            keep = np.isin(labeled, touching_labels)
            cleaned_slice = np.where(keep, slice_mask, 0)

            keep_mask_slice.append(cleaned_slice)

        keep_mask_slice = np.stack(keep_mask_slice, axis=-1)
        keep_mask_time.append(keep_mask_slice)

    keep_mask_time = np.stack(keep_mask_time, axis=-1)
    return keep_mask_time