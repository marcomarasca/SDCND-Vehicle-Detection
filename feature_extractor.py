import numpy as np
import cv2

from skimage.feature import hog
from tqdm import tqdm

color_space_map = {
    'HLS': cv2.COLOR_BGR2HLS,
    'HSV': cv2.COLOR_BGR2HSV,
    'YUV': cv2.COLOR_BGR2YUV,
    'YCrCb': cv2.COLOR_BGR2YCrCb,
}

def bin_spatial(img, size=(32, 32)):
    """
    Computes spatial features
    """
    features = cv2.resize(img, size).ravel()

    return features

def color_hist(img, nbins=32, bins_range=(0, 256)):
    """
    Computes color histogram features 
    """
    histograms = []
    
    # Compute the histogram of the color channels separately
    for channel in range(img.shape[2]):
        histograms.append(np.histogram(img[:,:,channel], bins=nbins, range=bins_range)[0])

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate(histograms)

    return hist_features

def extract_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    """
    Computes the HOG features and (optionally) the HOG features image
    """
    features = hog(img, orientations=orient, 
                        pixels_per_cell=(pix_per_cell, pix_per_cell),
                        cells_per_block=(cell_per_block, cell_per_block), 
                        transform_sqrt=True,
                        visualise=vis,
                        feature_vector=feature_vec,
                        block_norm= 'L2-Hys')

    hog_features = features[0]

    if vis:
        hog_image = features[1]
        return hog_features, hog_image
    else:
        return hog_features

def convert_color_space(img, color_space):
    
    if color_space is None or color_space == 'BGR':
        converted_img = np.copy(img)
    else:
        conversion = color_space_map.get(color_space)
        if conversion is None:
            print('[Warning]: Could not convert to {} color space, mapping is missing'.format(color_space))
            converted_img = np.copy(img)
        else:
            converted_img = cv2.cvtColor(img, conversion)

    return converted_img

def extract_features(imgs, 
                        color_space= 'YCrCb', 
                        spatial_size=(32, 32),
                        hist_bins=32, 
                        orient=12, 
                        pix_per_cell=8, 
                        cell_per_block=2):

    print('Extracting features on {} images ({}, {}, {}, {})'.format(
        len(imgs),
        'BGR' if color_space is None else color_space,
        'Spatial Features OFF' if spatial_size is None else 'Spatial Size: {}'.format(spatial_size),
        'Histogram Features OFF' if hist_bins is None else 'Histogram Bins {}'.format(hist_bins),
        'Orientations: {}, Pix Per Cell: {}, Cell Per Block: {}'.format(
            orient, pix_per_cell, cell_per_block
        )
    ))

    # List to append feature vectors to
    features = []
    for file in tqdm(imgs, unit=' images', desc='Extracting Features'):
        image_features = []
        image = cv2.imread(file)

        # Color space conversion
        feature_image = convert_color_space(image, color_space)

        # Spatial features
        if spatial_size is not None:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            image_features.append(spatial_features)

        # Color histogram features
        if hist_bins is not None:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            image_features.append(hist_features)

        # HOG features
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(extract_hog_features(feature_image[:,:,channel], 
                                orient, pix_per_cell, cell_per_block, 
                                vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)
        image_features.append(hog_features)

        features.append(np.concatenate(image_features))

    return features
