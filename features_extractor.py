import numpy as np
import cv2

from skimage.feature import hog
from tqdm import tqdm

class FeaturesExtractor:
    """
    This class is used to extract features (e.g. spatial, color histogram, HOG etc.) from images.
    """

    COLOR_SPACE_MAP = {
        'HLS': cv2.COLOR_BGR2HLS,
        'HSV': cv2.COLOR_BGR2HSV,
        'YUV': cv2.COLOR_BGR2YUV,
        'YCrCb': cv2.COLOR_BGR2YCrCb,
    }

    def __init__(self, color_space= 'YCrCb', spatial_size=16, hist_bins=32, orient=16, pix_per_cell=16, cell_per_block=2):
        """
        Initializes the features extractor with the given parameters

        Parameters
            color_space: The color space transform to apply to the images (from BGR), if None or 'BGR' no transformation is applied, default 'YCrCb'.
            spatial_size: The size for the spatial features, if None spatial features are not used, default 16.
            hist_bins: The number of color historgram bins to use, if None color features are not used, default 32.
            orient: The number of orientations for the HOG features, default 16.
            pix_per_cell: The number of pixels per cell for the HOG features, default 16.
            cell_per_block: The cells per block for the HOG features, default 2.
        """
        self.color_space = color_space
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block

    def bin_spatial(self, img):
        """
        Computes spatial features

        Parameters
            img: The input image
        Return
            Feature vector with spatial tranformation
        """
        features = cv2.resize(img, (self.spatial_size, self.spatial_size)).ravel()

        return features

    def _color_hist(self, channel):
        return np.histogram(channel, bins=self.hist_bins, range=(0, 256))[0]

    def color_hist(self, img, process_pool = None):
        """
        Computes color histogram features

        Parameters
            img: The input image
            process_pool: Optional multiprocessing Pool to use for extracting the histograms from the individual channels
        Return
            Feature vector with individual channels histograms concatenated
        """
        # Compute the histogram of the color channels separately
        if process_pool is None:
            histograms = []
            for channel in range(img.shape[2]):
                histograms.append(self._color_hist(img[:,:,channel]))
        else:
            histograms = process_pool.map(self._color_hist, [img[:,:,0], img[:,:,1], img[:,:,2]])

        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate(histograms)

        return hist_features

    def extract_hog_features(self, img, vis=False, feature_vec=True):
        """
        Computes the HOG features and (optionally) the HOG features image
        
        Parameters
            img: The input image (single channel)
            vis: If true returns an image with a visualization of the HOG features
            feature_vec: Return the data as a feature vector by calling .ravel() on the result  
                         just before returning.  
        Return
            Feature vector with HOG features, if vis is true a tuple (features, hog_image)
        """
        features = hog(img, orientations=self.orient, 
                            pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                            cells_per_block=(self.cell_per_block, self.cell_per_block), 
                            transform_sqrt=True,
                            visualise=vis,
                            feature_vector=feature_vec,
                            block_norm= 'L2-Hys')
        # If vis is True this is a tuple (hog_features, hog_image)
        return features

    def convert_color_space(self, img):
        """
        Converts the image to the extractor color space

        Parameters
            img: The input image
        Return
            Converted image, or a copy if the color space is not supplied or 'BGR'
        """
        
        if self.color_space is None or self.color_space == 'BGR':
            conversion = None
        else:
            conversion = FeaturesExtractor.COLOR_SPACE_MAP.get(self.color_space)

        if conversion is None:
            converted_img = np.copy(img)
        else:
            converted_img = cv2.cvtColor(img, conversion)

        return converted_img

    def extract_img_features(self, img, process_pool = None):
        """
        Extracts the features from the image (BGR expected) according to the extractor parameters.

        Parameters
            img: A BGR image
            process_pool: Optional multiprocessing Pool to use for extracting from the individual channels in parallel
                          the color histograms and the HOG features
        Return
            The extracted features vector
        """

        image_features = []
        
        # Color space conversion
        feature_image = self.convert_color_space(img)

        # Spatial features
        if self.spatial_size is not None:
            spatial_features = self.bin_spatial(feature_image)
            image_features.append(spatial_features)

        # Color histogram features
        if self.hist_bins is not None:
            hist_features = self.color_hist(feature_image, process_pool=process_pool)
            image_features.append(hist_features)

        # HOG features
        if process_pool is None:
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(self.extract_hog_features(feature_image[:,:,channel]))
        else:
            hog_features = process_pool.map(self.extract_hog_features, [feature_image[:,:,0], feature_image[:,:,1], feature_image[:,:,2]])
            
        hog_features = np.ravel(hog_features)
        
        image_features.append(hog_features)

        return np.concatenate(image_features)

    def extract_features(self, imgs, process_pool = None):
        """
        Extracts the features from the given list of images according to the extractor parameters.

        Parameters
            imgs: A list of images
            process_pool: Optional multiprocessing Pool to use for extracting from the individual channels in parallel
                          the color histograms and the HOG features
        Return
            An array of extracted features vectors
        """

        print('Extracting features on {} images ({}, {}, {}, {})'.format(
            len(imgs),
            'BGR' if self.color_space is None else self.color_space,
            'Spatial Features OFF' if self.spatial_size is None else 'Spatial Size: {}'.format(self.spatial_size),
            'Histogram Features OFF' if self.hist_bins is None else 'Histogram Bins {}'.format(self.hist_bins),
            'Orientations: {}, Pix Per Cell: {}, Cell Per Block: {}'.format(
                self.orient, self.pix_per_cell, self.cell_per_block
            )
        ))

        # List to append feature vectors to
        features = []

        for file in tqdm(imgs, unit=' images', desc='Extracting Features'):
           
            img = cv2.imread(file)

            img_features = self.extract_img_features(img, process_pool=process_pool)

            features.append(img_features)

        return features
