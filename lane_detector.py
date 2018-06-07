import numpy as np
import cv2
from collections import deque
import pickle
import os

class ImageProcessor:
    """
    Class used to process an image for the LaneDetector. Applies both color and gradient thresholding and produces a set of
    images (undistored, thresholded and warped) that can be used for debugging.
    """

    def __init__(self, calibration_data_file):

        # Camera calibration data
        calibration_data = self._load_calibration_data(file_path = calibration_data_file)
        self.mtx = calibration_data['mtx']
        self.dist = calibration_data['dist']

        # Gradient and color thresholding parameters
        self.sobel_kernel = 5
        self.grad_x_thresh = (15, 255) # Sobel x threshold
        self.grad_y_thresh = (25, 255) # Sobel y threshold
        self.grad_mag_thresh = (40, 255) # Sobel mag threshold
        self.grad_dir_thresh = (0.7, 1.3) # Sobel direction range
        self.grad_v_thresh = (180, 255) # HSV, V channel threshold to filter gradient

        self.r_thresh = (195, 255) # RGB, Red channel threshold
        self.s_thresh = (100, 255) # HSL, S channel threshold
        self.l_thresh = (195, 255) # HSL, L channel threshold
        self.b_thresh = (150, 255) # LAB, B channel threshold
        self.v_thresh = (140, 255) # HSV, V channel threshold

        # Perspective transformation parameters
        # slope = (y2 - y1) / (x2 - x1)
        # intercept = y1 - slope * x1
        # top left, top right = (570, 470), (722, 470)
        # bottom left, bottom right = (220, 720), (1110, 720)
        self.persp_src_left_line = (-0.7142857143, 877.142857146) # Slope and intercept for left line
        self.persp_src_right_line = (0.6443298969, 4.793814441) # Slope and intercept for right line
        self.persp_src_top_pct = 0.645 # Percentage from the top
        self.persp_src_bottom_pct = 0.02 # Percentage from bottom
        self.persp_dst_x_pct = 0.22 # Destination offset percent
        self.persp_src = None
        self.persp_dst = None

    
    def _load_calibration_data(self, file_path = os.path.join('camera_cal', 'calibration.p')):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def _warp_coordinates(self, img):

        if self.persp_src is None or self.persp_dst is None:

            cols = img.shape[1]
            rows = img.shape[0]

            src_top_offset = rows * self.persp_src_top_pct
            src_bottom_offset = rows * self.persp_src_bottom_pct
            left_slope, left_intercept = self.persp_src_left_line
            right_slope, right_intercept = self.persp_src_right_line

            top_left = [(src_top_offset - left_intercept) / left_slope, src_top_offset]
            top_right = [(src_top_offset - right_intercept) / right_slope, src_top_offset]
            bottom_left = [(rows - src_bottom_offset - left_intercept) / left_slope, rows - src_bottom_offset]
            bottom_right = [(rows - src_bottom_offset - right_intercept) / right_slope, rows - src_bottom_offset]

            #Top left, Top right, Bottom right, Bottom left        
            src = np.float32([top_left, top_right, bottom_right, bottom_left])

            dst_x_offset = cols * self.persp_dst_x_pct
    
            top_left = [dst_x_offset, 0]
            top_right = [cols - dst_x_offset, 0]
            bottom_left = [dst_x_offset, rows]
            bottom_right = [cols - dst_x_offset, rows]
            
            dst = np.float32([top_left, top_right, bottom_right, bottom_left])
                            
            self.persp_src = src
            self.persp_dst = dst
        
        return self.persp_src, self.persp_dst

    def _sobel(self, img, orient = 'x', sobel_kernel = 3):
        # Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
        else:
            sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
        
        return sobel

    def _apply_thresh(self, img, thresh = [0, 255]):
        result = np.zeros_like(img)
        result[(img >= thresh[0]) & (img <= thresh[1])] = 1
        return result

    def unwarp_image(self, img):

        img_shape = img.shape[1::-1]

        src, dst = self._warp_coordinates(img)

        warp_m = cv2.getPerspectiveTransform(dst, src)
        unwarped = cv2.warpPerspective(img, warp_m, img_shape)

        return unwarped

    def warp_image(self, img):

        img_shape = img.shape[1::-1]

        src, dst = self._warp_coordinates(img)

        warp_m = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, warp_m, img_shape)

        return warped

    def undistort_image(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def sobel_abs_thresh(self, sobel, thresh=[0,255]):
        # Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)
        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        binary_output = self._apply_thresh(scaled_sobel, thresh)

        return binary_output

    def sobel_mag_thresh(self, sobel_x, sobel_y, thresh=(0, 255)):
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobel_x**2 + sobel_y**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 

        binary_output = self._apply_thresh(gradmag, thresh)
        
        return binary_output

    def sobel_dir_thresh(self, sobel_x, sobel_y, thresh=(0, np.pi/2)):
        # Take the absolute value of the x and y gradients
        abs_sobel_x = np.absolute(sobel_x)
        abs_sobel_y = np.absolute(sobel_y)

        # Calculate the direction of the gradient 
        abs_grad_dir = np.arctan2(abs_sobel_y, abs_sobel_x)

        binary_output = self._apply_thresh(abs_grad_dir, thresh)
       
        return binary_output

    def gradient_thresh(self, img):

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v_ch = hsv_img[:,:,2]
        v_binary = self._apply_thresh(v_ch, self.grad_v_thresh)

        sobel_x = self._sobel(gray_img, sobel_kernel = self.sobel_kernel, orient = 'x')
        sobel_y = self._sobel(gray_img, sobel_kernel = self.sobel_kernel, orient = 'y')

        sobel_x_binary = self.sobel_abs_thresh(sobel_x, thresh = self.grad_x_thresh)
        sobel_y_binary = self.sobel_abs_thresh(sobel_y, thresh = self.grad_y_thresh)

        sobel_mag_binary = self.sobel_mag_thresh(sobel_x, sobel_y, thresh = self.grad_mag_thresh)
        sobel_dir_binary = self.sobel_dir_thresh(sobel_x, sobel_y, thresh = self.grad_dir_thresh)

        sobel_binary = np.zeros_like(sobel_x_binary)

        sobel_binary[(((sobel_x_binary == 1) & (sobel_y_binary == 1)) | (sobel_dir_binary == 1)) & (sobel_mag_binary == 1) & (v_binary == 1)] = 1
        
        return sobel_binary

    def color_thresh(self, img):

        hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        r_ch = img[:,:,2]
        r_binary = self._apply_thresh(r_ch, self.r_thresh)

        l_ch = hls_img[:,:,1]
        l_binary = self._apply_thresh(l_ch, self.l_thresh)

        s_ch = hls_img[:,:,2]
        s_binary = self._apply_thresh(s_ch, self.s_thresh)

        b_ch = lab_img[:,:,2]
        b_binary = self._apply_thresh(b_ch, self.b_thresh)

        v_ch = hsv_img[:,:,2]
        v_binary = self._apply_thresh(v_ch, self.v_thresh)

        result = np.zeros_like(s_binary)

        # B and V for yellow, R and L for white, S and V for both
        result[((b_binary == 1) & (v_binary == 1)) | ((r_binary == 1) & (l_binary == 1)) | ((s_binary == 1) & (v_binary == 1))] = 1

        return result
        
    def threshold_image(self, img):

        gradient_binary = self.gradient_thresh(img)
        color_binary = self.color_thresh(img)

        result = np.zeros_like(gradient_binary)
        result[(gradient_binary == 1) | (color_binary) == 1] = 255

        return result

    def process_image(self, img):
        """
        Process the given image appling undistorsion from the camera calibration data, thresholds the result and then
        warps the image for an bird-eye view of the road.
        """

        undistorted_img = self.undistort_image(img)

        thresholded_img = self.threshold_image(undistorted_img)

        warped_img = self.warp_image(thresholded_img)

        return undistorted_img, thresholded_img, warped_img


class LaneDetector:
    """
    The class is used to detect road lanes in processed (from img_processor) frames, using a sliding window 
    through convolutions to detect hot pixels. For each slice extracts the centroids found in the windows and
    fits a polynomial to compute the curvature and deviation from center. The same polynomial can be used to draw
    the lines in the frame. The final centroids returned by the pipeline are averaged among last X frames to smooth
    the result.
    """

    FAIL_CODES = {
        1: 'Lane distance out of range',
        2: 'Lane distance deviates from mean',
        3: 'Lane distance deviates from previous frame',
        4: 'Low left lane confidence',
        5: 'Low right lane confidence',
        9: 'Low lanes confidence'
    }

    def __init__(self, window_width = 30, window_height = 80, margin = 35, smooth_frames = 15, xm = 3.7/700, ym = 3/110):
        """
        Initializes the class with the given parameters for the windows. Note that if smooth_frames is zero no interpolation is
        performed between frames.

        Parameters
            window_width: The width of the sliding window
            window_height: The height of the sliding window
            margin: Left/right margin that is used by the sliding window in subsequent layers
            smooth_frames: The number of frames to use for smoothing the result of the detection
            xm: The number of meters per pixel on the horizontal axis
            ym: The number of meters per pixel on the vertical axis

        """
        # [(left, right, y)]
        self.centroids_buffer = deque(maxlen = smooth_frames)
        self.last_lanes_distance = None
        self.window_width = window_width
        self.window_height = window_height
        self.margin = margin
        self.first_window_height = .75 # The height for the first window (for the start of the lane at the bottom)
        self.min_points_fit = 4 # Number of point already found before trying to fit a line when no center is detected
        self.min_confidence = 0.16 # Min confidence to keep a detected lane
        self.dist_thresh = (510, 890) # Lanes distance threshold
        self.max_dist_diff = 60 # Max lanes distance difference between frames
        self.max_dist_mean_dev = 80 # Max lanes distance deviation from mean
        self.xm = xm
        self.ym = ym
        self.min_conv_signal = 1000 # Min conv signal to avoid noise
        self.max_window_signal = None # Cache for the max amount of signal in a window to compute confidence

    def compute_window_max_signal(self, window, width, height, max_value = 255):
        """
        Returns the maximum amount of signal in a window with the given dimension, given the value for each pixel
        """
        window_sum = np.sum(np.ones((height, width)) * max_value, axis = 0)
        conv_signal = np.convolve(window, window_sum)
        return np.max(conv_signal)

    def detect_lanes(self, img):
        """
        Detection pipeline: Starts out with detecting the bottom lanes using a bigger window for the convolution. The
        centroids found at this stage are used as base for the next layer (look around the margin). For each layer estimates
        the correctness of the detected centroids and tries to detect failures based on the confidence (given by the amount of
        signal in each window) and the distance between lanes (and the mean of the previous lanes if smoothing is enabled).

        Parameters
            img: The input image, must be a processed image from the ImageProcessor
        
        Returns
            lanes_centroids: The centroids for the detected lanes
            (left_fit, right_fit): The left and right polynomial coefficients from the lanes_centroids
            (left_curvature, right_curvature): The curvature in meters
            deviation: The deviation from the center of the lane
            fail_code: 0 if the lanes could be detected from this frame, otherwise a code that can be mapped in the FAIL_CODES dictionary
                       Note that if the detection was not successful the lanes_centroids and the fits are the one from the previous frame  
        """
        lanes_centroids = []
        centroids_confidence = []

        window = np.ones(self.window_width)

        if self.max_window_signal is None:
            self.max_window_signal = self.compute_window_max_signal(window, self.window_width, self.window_height)
   
        left_center, left_confidence, right_center, right_confidence, center_y = self.estimate_start_centroids(img, window)

        # Add what we found for the first layer
        lanes_centroids.append((left_center, right_center, center_y))
        centroids_confidence.append((left_confidence, right_confidence))

        # Go through each layer looking for max pixel locations
        for level in range(1, (int)(img.shape[0] / self.window_height)):

            left_center, left_confidence, right_center, right_confidence, center_y = self.estimate_centroids(img, window, level, left_center, right_center, lanes_centroids)
            
            lanes_centroids.append((left_center, right_center, center_y))
            centroids_confidence.append((left_confidence, right_confidence))

        lanes_centroids = np.array(lanes_centroids)
        centroids_confidence = np.array(centroids_confidence)

        fail_code = self.detect_failure(lanes_centroids, centroids_confidence)

        # If the lane detection failed and we have frames uses the last one
        if fail_code > 0 and len(self.centroids_buffer) > 0:
            lanes_centroids = self.centroids_buffer[-1]
        
        self.centroids_buffer.append(lanes_centroids)

        if len(self.centroids_buffer) > 0:
            self.last_lanes_distance = self.compute_mean_distance(lanes_centroids[:,0], lanes_centroids[:,1])
            # Average frames for smoothing
            lanes_centroids = np.average(self.centroids_buffer, axis = 0)
        
        left_fit, right_fit = self.lanes_fit(lanes_centroids)
        left_fit_scaled, right_fit_scaled = self.lanes_fit(lanes_centroids, ym = self.ym, xm = self.xm)

        curvature = self.compute_curvature(left_fit_scaled, right_fit_scaled, np.max(lanes_centroids[:,:2]) * self.ym)
        deviation = self.compute_deviation(left_fit_scaled, right_fit_scaled, img.shape[0] * self.ym, img.shape[1] * self.xm)

        return lanes_centroids, (left_fit, right_fit), curvature, deviation, fail_code

    def estimate_start_centroids(self, img, window):
        """
        Estimates the centroids at the bottom of the image, if some frames are buffered uses the previous frames
        to define a boundary.

        Parameters
            img: Input image, must be processed from the ImageProcessor
            window: The base window used in the convolutions within a frame
        """

        if len(self.centroids_buffer) > 0:
            # If a "good" start was found already, limit the search within the previous
            # frame start boundaries
            prev_centroids = np.array(self.centroids_buffer)
            prev_left_centroids = prev_centroids[:,:,0]
            prev_right_centroids = prev_centroids[:,:,1]
            left_min_index = int(max(np.min(prev_left_centroids) - self.margin, 0))
            left_max_index = int(min(np.max(prev_left_centroids) + self.margin, img.shape[1]))
            right_min_index = int(max(np.min(prev_right_centroids) - self.margin, 0))
            right_max_index = int(min(np.max(prev_right_centroids) + self.margin, img.shape[1]))
        else:
            left_min_index = 0
            left_max_index = int(img.shape[1] / 2)
            right_min_index = int(img.shape[1] / 2)
            right_max_index = img.shape[1]

        window_top = int(img.shape[0] * self.first_window_height)
        window_y = int(img.shape[0] - self.window_height / 2)
        
        left_sum = np.sum(img[window_top:, left_min_index:left_max_index], axis=0)
        left_signal = np.convolve(window, left_sum)
        left_center, left_confidence = self.get_conv_center(left_signal, left_min_index, max_signal = None)
        
        right_sum = np.sum(img[window_top:, right_min_index:right_max_index], axis=0)
        right_signal = np.convolve(window, right_sum)
        right_center, right_confidence = self.get_conv_center(right_signal, right_min_index, max_signal = None)

        return left_center, left_confidence, right_center, right_confidence, window_y

    def get_conv_center(self, conv_signal, offset, max_signal = None):
        """
        Computes the center from the given convolution signal assuming the given offset

        Parameters
            conv_signal: The result of the convolution of a window
            offset: The offset used for the convolution (so that the center is relative to the image and not the window)
            max_signal: The maximum amount of singal in the convolution, used to compute the confidence, if supplied a threshold
                        is applied for the minimum amount of signal to consider valid
        
        Returns
            center: The center x, None if not enough signal
            confidence: The ratio between the signal and the max amount of signal
        """

        max_conv_signal = np.max(conv_signal)

        if max_signal is None or max_conv_signal > self.min_conv_signal:
            center = np.argmax(conv_signal) + offset - (self.window_width / 2)
            confidence = 1.0 if max_signal is None else max_conv_signal / max_signal
        else:
            center = None
            confidence = 0.0
        
        return center, confidence

    def find_window_centroid(self, img, conv_signal, prev_center):
        """
        Finds the centroids in a window resulting in the given convolution assuming the given previous starting center

        Parameters
            img: The input image
            conv_signal: The result of the convolution of a window
            prev_center: The previous center to be used as reference
        
        Returns
            center: The center x, None if not enough signal
            confidence: The ratio between the signal and the max amount of signal in a window
        """

        offset = self.window_width / 2
        # Find the best center by using past center as a reference
        min_index = int(max(prev_center + offset - self.margin, 0))
        max_index = int(min(prev_center + offset + self.margin, img.shape[1]))

        conv_window = conv_signal[min_index:max_index]

        center, confidence = self.get_conv_center(conv_window, min_index, self.max_window_signal)

        return center, confidence

    def estimate_centroids(self, img, window, level, prev_l_center, prev_r_center, lanes_centroids):
        """
        Estimates the centroids for the window at the given level using the given previous centers as reference

        Parameters
            img: The input image
            level: The level for the convolution (e.g. img height/window height)
            lanes_centroids: The centroids found so far in the frame
    
        Returns
            left_center: x coordinate for the left center
            left_confidence: Confidence for the left center
            right_center: x coordinate for the right center
            right_confidence: Confidence for the right center
            center_y: y coordinate for both centers
        """
        window_top = int(img.shape[0] - (level + 1) * self.window_height)
        window_bottom = int(img.shape[0] - level * self.window_height)
        center_y = int(window_bottom - self.window_height / 2)

        # Convolve the window into the vertical slice of the image
        window_sum = np.sum(img[window_top:window_bottom, :], axis=0)

        conv_signal = np.convolve(window, window_sum)

        left_center, left_confidence = self.find_window_centroid(img, conv_signal, prev_l_center)
        right_center, right_confidence = self.find_window_centroid(img, conv_signal, prev_r_center)

        if left_center is None and right_center is None:
            # If no centers were detected but we have enough points
            # we can try to fit the lane already to get an estimated point
            if len(lanes_centroids) > self.min_points_fit:
                left_fit, right_fit = self.lanes_fit(np.array(lanes_centroids))
                left_center = self.fit_point(img, left_fit, center_y)
                right_center = self.fit_point(img, right_fit, center_y)
            else:
                left_center = prev_l_center
                right_center = prev_r_center
        # If either one is detected we can use the previous distance as an estimation
        elif left_center is None:
            left_center = right_center - (prev_r_center - prev_l_center)
        elif right_center is None:
            right_center = left_center + (prev_r_center - prev_l_center)

        return left_center, left_confidence, right_center, right_confidence, center_y

    def fit_point(self, img, fit, y):
        return np.clip(fit[0]*y**2 + fit[1]*y + fit[2], 0, img.shape[1])

    def detect_failure(self, lanes_centroids, centroids_confidence):
        """
        Tries to detect detection failure from the given centroids and confidence. Uses the mean lane distance from
        the given centroids compared to the previous mean of the previous frames.
        """

        left_confidence, right_confidence = np.mean(centroids_confidence, axis = 0)
    
        # Checks detection confidence
        confidence_fail = 0
        if left_confidence < self.min_confidence:
            confidence_fail += 4
        if right_confidence < self.min_confidence:
            confidence_fail += 5
        
        if confidence_fail > 0:
            return confidence_fail

        lanes_distance = self.compute_mean_distance(lanes_centroids[:,0], lanes_centroids[:,1])

        # Checks lane distance threshold
        if lanes_distance < self.dist_thresh[0] or lanes_distance > self.dist_thresh[1]:
            return 1

        # Checks the difference with the previous frame
        if self.last_lanes_distance is not None and abs(lanes_distance - self.last_lanes_distance) > self.max_dist_diff:
            return 3

        # Checks that the distance with the mean of the previous frames
        if len(self.centroids_buffer) > 0:
            mean_centroids = np.mean(self.centroids_buffer, axis = 0)
            mean_lanes_distance = self.compute_mean_distance(mean_centroids[:,0], mean_centroids[:,1])
            if np.absolute(lanes_distance - mean_lanes_distance) > self.max_dist_mean_dev:
                return 2

        return 0

    def compute_mean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2) / len(x1))

    def lane_fit(self, lanes_centroids, idx = 0, ym = 1, xm = 1):
        fit_y_vals = lanes_centroids[:,2] * ym
        fit_x_vals = lanes_centroids[:,idx] * xm

        fit = np.polyfit(fit_y_vals, fit_x_vals , 2)

        return fit

    def lanes_fit(self, lanes_centroids, ym = 1, xm = 1):

        left_fit = self.lane_fit(lanes_centroids, 0, ym, xm)
        right_fit = self.lane_fit(lanes_centroids, 1, ym, xm)

        return left_fit, right_fit

    def compute_curvature(self, left_fit, right_fit, y_eval):
        """
        Curvature computation, assumes a scaled left and right fit
        """
       
        left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1])**2)**1.5) / np.absolute(2 * left_fit[0])
        right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1])**2)**1.5) / np.absolute(2 * right_fit[0])

        return (left_curverad, right_curverad)

    def compute_deviation(self, left_fit, right_fit, y_eval, x_eval):
        """
        Deviation computation, assumes a scaled left and right fit
        """
        
        l_x = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
        r_x = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
        center = (l_x + r_x) / 2.0
        
        return center - x_eval / 2.0

    def draw_lanes(self, img, polyfit, blend = False, marker_width = 20, fill_color = None):
        """
        Draws the given polynomials (left, right) using the image as reference, if blend is True draws on top of the input image
        """
        left_fit, right_fit = polyfit

        y_vals = range(0, img.shape[0])

        left_x_vals = left_fit[0] * y_vals * y_vals + left_fit[1] * y_vals + left_fit[2]
        right_x_vals = right_fit[0] * y_vals * y_vals + right_fit[1] * y_vals + right_fit[2]

        if blend:
            out_img = img
        else:
            out_img = np.zeros_like(img)

        cv2.polylines(out_img, np.int_([list(zip(left_x_vals, y_vals))]), False, (255, 0, 0), marker_width)
        cv2.polylines(out_img, np.int_([list(zip(right_x_vals, y_vals))]), False, (0, 0, 255), marker_width)

        if fill_color is not None:
            offset = marker_width / 2
            inner_x = np.concatenate((left_x_vals + offset, right_x_vals[::-1] - offset), axis = 0)
            inner_y = np.concatenate((y_vals, y_vals[::-1]), axis = 0)
            cv2.fillPoly(out_img, np.int_([list(zip(inner_x, inner_y))]), color = fill_color)

        return out_img

    def window_mask(self, width, height, img_ref, x, y):
        output = np.zeros_like(img_ref)
        output[int(y - height/2):int(y + height/2),max(0,int(x-width/2)):min(int(x+width/2),img_ref.shape[1])] = 1
        return output

    def draw_windows(self, img, lanes_centroids, polyfit = None, blend = False):
        """
        Draws the windows around the given centroids using the image as reference, if blend is True draws on top of the input image
        """
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(img)
        r_points = np.zeros_like(img)

        # Go through each level and draw the windows 	
        for level in range(0, len(lanes_centroids)):
            # Window_mask is a function to draw window areas
            center_y = lanes_centroids[level][2]
            l_mask = self.window_mask(self.window_width, self.window_height, img, lanes_centroids[level][0], center_y)
            r_mask = self.window_mask(self.window_width, self.window_height, img, lanes_centroids[level][1], center_y)
            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | (l_mask == 1) ] = 255
            r_points[(r_points == 255) | (r_mask == 1) ] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8) # make window pixels green

        if blend:
            out_img = np.array(cv2.merge((img, img, img)), np.uint8)
            out_img = cv2.addWeighted(out_img, 1.0, template, 0.5, 0)
        else:
            out_img = template

        if polyfit is None:
            left_fit, right_fit = self.lanes_fit(lanes_centroids)
            polyfit = (left_fit, right_fit)

        out_img = self.draw_lanes(out_img, polyfit, blend = True, marker_width = 3)

        return out_img
        
