import numpy as np
import cv2
import os

from scipy.ndimage.measurements import label
from features_extractor import FeaturesExtractor
from data_loader import *

class VehicleDetector:

    def __init__(self, model_file = os.path.join('models', 'model.p'), 
                       cells_per_step = 1,
                       min_confidence = 0.3, 
                       heat_threshold = 5, 
                       interpolation = (cv2.INTER_AREA, cv2.INTER_LINEAR)):
        self._load_model(model_file = model_file)
        self.cells_per_step = cells_per_step  # Instead of overlap, define how many cells to step
        self.min_confidence = min_confidence
        self.heat_threshold = heat_threshold
        self.interpolation = interpolation # The interpolation algorithms used for shrinking/zooming
        self.layers = [
            # y_min, y_max, scale
            (400, 512, 0.8),
            (400, 528, 1),
            (400, 528, 1.5),
            (400, 592, 2)
        ]

    def _load_model(self, model_file):
        model_params = load_model(model_file = model_file)
        self.model = model_params['model']
        self.scaler = model_params['scaler']
        self.window = model_params['window']
        
        self.pix_per_cell = model_params['pix_per_cell']
        self.cell_per_block = model_params['cell_per_block']

        self.features_extractor = FeaturesExtractor(
            color_space = model_params['color_space'],
            spatial_size = model_params['spatial_size'],
            hist_bins = model_params['hist_bins'],
            orient = model_params['orient'],
            pix_per_cell = self.pix_per_cell,
            cell_per_block = self.cell_per_block
        )

    def window_confidence(self, window_img, layer_hog_features, x_pos, y_pos, blocks_per_window, process_pool = None):
        
        features = []
        
        # Spatial features
        if self.features_extractor.spatial_size is not None:
            spatial_features = self.features_extractor.bin_spatial(window_img)
            features.append(spatial_features)

        # Color historgram features
        if self.features_extractor.hist_bins is not None:
            hist_features = self.features_extractor.color_hist(window_img, process_pool=process_pool)
            features.append(hist_features)

        hog_features = []

        # HOG features
        for ch_hog_features in layer_hog_features:
            hog_features.append(ch_hog_features[y_pos:y_pos + blocks_per_window, x_pos:x_pos + blocks_per_window].ravel())
        
        hog_features = np.ravel(hog_features)

        features.append(hog_features)

        features = np.concatenate(features)

        # Scale features
        features = self.scaler.transform(features.reshape(1, -1))    

        # Computes prediction confidence
        
        confidence = self.model.decision_function(features)

        return confidence
    
    def windows_search(self, img, y_min = None, y_max = None, scale = 1.0, process_pool = None):

        if y_min is None:
            y_min = 0
        
        if y_max is None:
            y_max = img.shape[0]

        # Selects the image layer
        img_layer = img[y_min:y_max, :, :]

        # Color space conversion
        img_layer = self.features_extractor.convert_color_space(img_layer)

        # Scales if necessary
        if scale != 1.0:
            # Uses the correct algorithm for resizing
            interpolation = self.interpolation[0] if scale < 1 else self.interpolation[1]
            resize_shape = (np.int(img_layer.shape[1]/scale), np.int(img_layer.shape[0]/scale))
            img_layer = cv2.resize(img_layer, resize_shape, interpolation = interpolation)

        # Define blocks and steps as above
        x_blocks = (img_layer.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        y_blocks = (img_layer.shape[0] // self.pix_per_cell) - self.cell_per_block + 1

        blocks_per_window = (self.window // self.pix_per_cell) - self.cell_per_block + 1
        
        x_steps = (x_blocks - blocks_per_window) // self.cells_per_step + 1
        y_steps = (y_blocks - blocks_per_window) // self.cells_per_step + 1

        # Compute individual channel HOG features for the entire image
        if process_pool is None:
            hog_features = []

            for channel in range(img_layer.shape[2]):
                ch_features = self.features_extractor.extract_hog_features(img_layer[:,:,channel], feature_vec = False)
                hog_features.append(ch_features)
        else:
            hog_features = process_pool.starmap(self.features_extractor.extract_hog_features, [
                (img_layer[:,:,0], False, False), 
                (img_layer[:,:,1], False, False), 
                (img_layer[:,:,2], False, False)
            ])
            
        windows = []

        for window_x in range(x_steps):
            for window_y in range(y_steps):
                x_pos = window_x * self.cells_per_step
                y_pos = window_y * self.cells_per_step

                # Position relative to the image layer
                window_min_x = x_pos * self.pix_per_cell
                window_min_y = y_pos * self.pix_per_cell

                # Extract the image window
                window_img = img_layer[window_min_y:window_min_y + self.window, window_min_x:window_min_x + self.window]
                
                if window_img.shape[0] < self.window or window_img.shape[1] < self.window:
                    window_img = cv2.resize(window_img, (self.window, self.window), interpolation = self.interpolation[0])

                confidence = self.window_confidence(window_img, hog_features, x_pos, y_pos, blocks_per_window, process_pool=process_pool)

                window_scale = np.int(self.window * scale)

                top_left = (np.int(window_min_x * scale), np.int(window_min_y * scale) + y_min)
                bottom_right = (top_left[0] + window_scale, top_left[1] + window_scale)

                windows.append(((top_left, bottom_right), confidence))
                
        return windows

    def heat(self, heatmap, windows):
        
        for window in filter(lambda window:window[1] > self.min_confidence, windows):
            bbox, _ = window
            heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1
            #heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1 * confidence
        
        return heatmap

    def heatmap(self, img, windows):
        
        # Base empty heatmap
        heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
        # Builds heat
        self.heat(heatmap, windows)
        # Threshold
        heatmap[heatmap <= self.heat_threshold] = 0
        # Clipping
        heatmap = np.clip(heatmap, 0, 255)
        
        return heatmap

    def bounding_boxes(self, heatmap):
        
        # Extract labels
        labels = label(heatmap)

        bounding_boxes = []
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzero_y = np.array(nonzero[0])
            nonzero_x = np.array(nonzero[1])

            # Define a bounding box based on min/max x and y
            top_left = (np.min(nonzero_x), np.min(nonzero_y))
            bottom_right = (np.max(nonzero_x), np.max(nonzero_y))

            bounding_boxes.append((top_left, bottom_right))

        return bounding_boxes
    
    def detect_vehicles(self, img, process_pool = None):
        
        windows = []

        # Extract the windows for each layer
        for y_min, y_max, scale in self.layers:
            
            layer_windows = self.windows_search(img, y_min = y_min, y_max = y_max, scale = scale, process_pool = process_pool)

            windows.extend(layer_windows)

        # Computes the heatmap
        heatmap = self.heatmap(img, windows)

        # Computes the detected cars
        bounding_boxes = self.bounding_boxes(heatmap)

        return bounding_boxes, heatmap, windows

