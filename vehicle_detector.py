import numpy as np
import cv2
import os

from scipy.ndimage.measurements import label
from features_extractor import FeaturesExtractor
from data_loader import load_model
from collections import deque

class WindowSearch:
    """
    Implementation of multi-scale windows search
    """

    INTER = (cv2.INTER_AREA, cv2.INTER_LINEAR)

    def __init__(self, model_file, search_layers):
        """
        Initializes the window search with the given model (for predictions) and the given list
        of search layers.

        Parameters
            model_file: The path to the pickle file produced while training (See model.js)
            search_layers: An array of parameters for each layer to search in, the format for each layer is as follows:
                           (y_min, y_max, x_min, x_max, scale, cells_per_step)
        """
        self._load_model(model_file)
        # [(y_min, y_max, x_min, x_max, scale, cells_per_step)]
        self.search_layers  = search_layers
    
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
        
        print(model_params)

    def _window_confidence(self, window_img, hog_features):
        """
        Computes the features for the given window image (reusing the provided HOG features) and predicts
        the presence of a vehicle

        Parameters
            window_img: The (window) image to process
            hog_features: The HOG features for the given window
        Return
            The confidence of the prediction, in terms of distance from the plane
        """
        features = []
        
        # Spatial features
        if self.features_extractor.spatial_size is not None:
            spatial_features = self.features_extractor.bin_spatial(window_img)
            features.append(spatial_features)

        # Color historgram features
        if self.features_extractor.hist_bins is not None:
            hist_features = self.features_extractor.color_hist(window_img)
            features.append(hist_features)

        features.append(hog_features)

        features = np.concatenate(features)

        # Scale features
        features = self.scaler.transform(features.reshape(1, -1))

        # Computes prediction confidence
        confidence = self.model.decision_function(features)[0]

        return confidence
    
    def _windows_search(self, img, y_min = None, y_max = None, x_min = None, x_max = None, scale = 1.0, cells_per_step = 2):
        """
        Performs a window search for a single layer defined by the given coordinates in the image

        Parameters
            img: The input image, assumes already converted to the correct color space
            y_min: Start y
            y_max: End y
            x_min: Start x
            x_max: End x
            scale: Scale factor
            cells_per_step: Defines the overlap of the windows (e.g. assuming an original window of 64, with 8x8 cells
                            a cells_per_step of 2 means 75% overlap)
        Return
            An array of tuples with the window (bounding boxes) and the prediction confidence 
        """
        if y_min is None:
            y_min = 0
        
        if y_max is None:
            y_max = img.shape[0]

        if x_min is None:
            x_min = 0
        
        if x_max is None:
            x_max = img.shape[1]

        # Selects the image layer
        img_layer = img[y_min:y_max, x_min:x_max, :]

        # Scales if necessary
        if scale != 1.0:
            # Uses the correct algorithm for resizing
            interpolation = WindowSearch.INTER[0] if scale > 1 else WindowSearch.INTER[1]
            resize_shape = (np.int(img_layer.shape[1]/scale), np.int(img_layer.shape[0]/scale))
            img_layer = cv2.resize(img_layer, resize_shape, interpolation = interpolation)

        # Define blocks and steps as above
        x_blocks = (img_layer.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        y_blocks = (img_layer.shape[0] // self.pix_per_cell) - self.cell_per_block + 1

        blocks_per_window = (self.window // self.pix_per_cell) - self.cell_per_block + 1
        
        x_steps = (x_blocks - blocks_per_window) // cells_per_step + 1
        y_steps = (y_blocks - blocks_per_window) // cells_per_step + 1

        # Compute individual channel HOG features for the entire image
        hog_features = list(map(lambda ch:self.features_extractor.extract_hog_features(img_layer[:,:,ch], feature_vec = False), range(img_layer.shape[2])))
            
        windows = []

        for window_x in range(x_steps):
            for window_y in range(y_steps):
                x_pos = window_x * cells_per_step
                y_pos = window_y * cells_per_step

                # Position relative to the image layer
                window_min_x = x_pos * self.pix_per_cell
                window_min_y = y_pos * self.pix_per_cell

                # Extract the image window
                window_img = img_layer[window_min_y:window_min_y + self.window, window_min_x:window_min_x + self.window]
                
                if window_img.shape[0] < self.window or window_img.shape[1] < self.window:
                    window_img = cv2.resize(window_img, (self.window, self.window), interpolation = WindowSearch.INTER[1])

                # Extracts the HOG features for the image window
                window_hog_features = list(map(lambda hog_ch:hog_ch[y_pos:y_pos + blocks_per_window, x_pos:x_pos + blocks_per_window].ravel(), hog_features))
                
                window_hog_features = np.ravel(window_hog_features)

                confidence = self._window_confidence(window_img, window_hog_features)

                window_scale = np.int(self.window * scale)

                top_left = (np.int(window_min_x * scale) + x_min, np.int(window_min_y * scale) + y_min)
                bottom_right = (top_left[0] + window_scale, top_left[1] + window_scale)

                windows.append(((top_left, bottom_right), confidence))
                
        return windows
    
    def search(self, img, process_pool = None):
        """
        Performs a search on the given image for each of the search layers initialized in the WindowSearch

        Parameters
            img: The input image, will be converted to the color space used during training
            process_pool: Optional process pool used to split computation of each search layer
        Return
            An array tuples (scale, cells_per_step, layer_windows) containing all the windows (bbox, confidence) for each
            search layer
        """
        # Color space conversion
        img = self.features_extractor.convert_color_space(img)

        windows = []

        # Extract the windows for each layer
        if process_pool is None:
            for y_min, y_max, x_min, x_max, scale, cells_per_step in self.search_layers:
                layer_windows = self._windows_search(img, y_min, y_max, x_min, x_max, scale, cells_per_step)
                windows.append((scale, cells_per_step, layer_windows))
        else:
            process_params = [(img, y_min, y_max, x_min, x_max, scale, cells_per_step) for (y_min, y_max, x_min, x_max, scale, cells_per_step) in self.search_layers]
            process_result = process_pool.starmap(self._windows_search, process_params)

            for (_, _, _, _, scale, cells_per_step), layer_windows in zip(self.search_layers, process_result):
                windows.append((scale, cells_per_step, layer_windows))
        
        return windows

class VehicleDetector:
    """
    Implementation of vehicle detection with multi-scale window search and averaging of the resulting heatmaps
    """
    def __init__(self, model_file     = os.path.join('models', 'model.p'),
                       min_confidence = 0.5,
                       heat_threshold = 3.5,
                       smooth_frames  = 8,
                       search_layers  = [(400, 496,  256, 1024,   1, 1),
                                         (400, 528,   64, 1216,   1, 2),
                                         (400, 592, None, None, 1.5, 2),
                                         (400, 656, None, None,   2, 1)]):
        """
        Detector initialization

        Parameters
            model_file: The path to the model file produced during training
            min_confidence: Threshold on the confidence of the predictions
            heat_threshold: Threshold on the heat generated in the final heatmap from all the windows, if
                            smooth_frames > 0 this threshold is used on the average of the last smooth_frames
            smooth_frames: If > 0 is used to collect the last smooth_frames heatmaps for averaging
            search_layers: An array of parameters for each layer to search in, the format for each layer is as follows:
                           (y_min, y_max, x_min, x_max, scale, cells_per_step)
        """
        self.windows_search = WindowSearch(model_file, search_layers)
        self.min_confidence = min_confidence
        self.heat_threshold = heat_threshold
        self.smooth_frames  = smooth_frames
        self.heatmap_buffer = deque(maxlen = smooth_frames)
        self.frames         = deque(maxlen = smooth_frames)

        # Min dimensions for a detected bounding box
        self.min_box_width      = self.windows_search.window * 0.8
        self.min_box_height     = self.windows_search.window * 0.5
        
        print('Min Confidence: {}, Heat Threshold: {}, Frame Smoothing: {}'.format(
            min_confidence,
            heat_threshold,
            'Disabled' if smooth_frames == 0 else smooth_frames
        ))

    def _heat(self, heatmap, windows, min_confidence):
        """
        Heats the given heatmap using the given windows iff the window confidence is >= than the given min_confidence
        """
        # scale, cells_per_step, bboxes
        for _, _, bboxes in windows:
            # bbox, confidence
            for bbox, _ in filter(lambda bbox:bbox[1] >= min_confidence, bboxes):
                heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1
        
        return heatmap

    def _heatmap(self, img, windows, min_confidence):
        """
        Creates an heatmap for the given image from the given list of windows
        """
        # Base empty heatmap
        heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)
        # Builds heat
        self._heat(heatmap, windows, min_confidence)
        
        return heatmap

    def _bounding_boxes(self, heatmap):
        """
        Extracts bounding boxes from the given heatmap (uses scipy.ndimage.measurements). This implementation drops
        bounding boxes that are too small
        """
        # Extract labels
        labels = label(heatmap)

        bounding_boxes = []

        # Min/Max bounding box fmor labels
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzero_x = np.array(nonzero[1])
            nonzero_y = np.array(nonzero[0])

            # Define a bounding box based on min/max x and y
            top_left = (np.min(nonzero_x), np.min(nonzero_y))
            bottom_right = (np.max(nonzero_x), np.max(nonzero_y))
            
            width = bottom_right[0] - top_left[0]
            height = bottom_right[1] - top_left[1]

            # Filter small boxes
            if width >= self.min_box_width and height >= self.min_box_height:
                bounding_boxes.append((top_left, bottom_right))
            
        return bounding_boxes
    
    def detect_vehicles(self, img, process_pool = None):
        """
        Detects vehicles in the given image

        Parameters
            img: The input image, BGR expected
            process_pool: Optional process pool for parallel processing (may speed up detection)

        Return
            The detected cars in terms of bounding boxes, the used heatmap and the windows searched in the image
        """
        
        windows = self.windows_search.search(img, process_pool = process_pool)
        
        # Computes the heatmap
        heatmap = self._heatmap(img, windows, self.min_confidence)

        self.heatmap_buffer.append(heatmap)

        if len(self.heatmap_buffer) > 1:
            # Takes the average over the last x frames
            heatmap = np.average(self.heatmap_buffer, axis = 0)

        # Threshold
        heatmap[heatmap < self.heat_threshold] = 0

        # Clipping
        heatmap = np.clip(heatmap, 0, 255)
       
        # Computes the detected cars
        bounding_boxes = self._bounding_boxes(heatmap)

        return bounding_boxes, heatmap, windows

