from utils import *


############ load major revisions #########
data_path = 'data/test_4d'
patients = [pat.split('/')[-1].replace('.nii.gz','').replace('image___','') for pat in glob.glob(f'{data_path}/*') if 'image___' in pat    ]
##########################################


# load model
model_name = 'SEG4D-112'
segger_path = f'models/{model_name}.h5'
segger = build_unet3plus_4D(input_shape=(32, None, 128, 128, 1), num_classes=3)
segger.load_weights(segger_path) 

for patient in tqdm(patients): # loop through cases
    if not os.path.exists(f'results/compare_gifs/{model_name}/{patient}.gif'):
        image = load_nii(f'{data_path}/image___{patient}.nii.gz') # load image
        mask_2d = load_nii(f'{data_path}/masks___{patient}.nii.gz') # load mask

        [x_min, y_min, x_max, y_max] = find_crop_box(np.max(mask_2d, axis = (-1,-2)), crop_factor=1.5) # crop image
        cropped_image = image[y_min:y_max, x_min:x_max, :, :]
        cropped_mask = mask_2d[y_min:y_max, x_min:x_max, :, :]
        cropped_mask = get_one_hot(cropped_mask.astype('uint8'), 3)

        target_time = 32
        target_image = 128

        time_zoom = target_time / cropped_image.shape[3]
        image_zoom = target_image / cropped_image.shape[0]

        image_to_seg = zoom(cropped_image, (image_zoom, image_zoom, 1, time_zoom), order=1)
        image_to_seg = np.transpose(image_to_seg, (3, 2, 0, 1)) 
        image_to_seg = clip_outliers(image_to_seg, lower_percentile=1, upper_percentile=99)
        image_to_seg = standardize(image_to_seg)

        cropped_mask_4d = segger.predict(image_to_seg[np.newaxis, ...,np.newaxis])[-1] # predict the masks
        cropped_mask_4d = get_one_hot(np.argmax(cropped_mask_4d,axis = -1), 3)[0] # binarise each mask
        cropped_mask_4d = transpose_channels(cropped_mask_4d, ['T','D','H','W','C'], ['H','W','D','T','C'])

        cropped_mask_4d = zoom(cropped_mask_4d, (1/image_zoom, 1/image_zoom, 1, 1/time_zoom, 1), order=0) # size 4d mask back to cropped size
        make_video(cropped_image, cropped_mask, cropped_image, cropped_mask_4d, f'results/compare_gifs/{patient}')

        # resize to original image size        
        H, W, D, T = image.shape
        C = cropped_mask_4d.shape[-1]
        mask_4d = np.zeros((H, W, D, T, C), dtype=cropped_mask_4d.dtype)
        mask_4d[y_min:y_max, x_min:x_max, :, :, :] = cropped_mask_4d
        
        save_mask(np.argmax(mask_4d, -1), save_path=f'results/masks/masks___{patient}_4D.nii.gz')