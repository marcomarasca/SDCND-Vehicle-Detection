import numpy as np
import cv2

def draw_bboxes(img, bboxes, color, thick):
    
    img_copy = np.copy(img)

    for bbox in bboxes:
        cv2.rectangle(img_copy, bbox[0], bbox[1], color, thick)
    
    return img_copy

def draw_windows(img, windows, min_confidence):
    
    # Filters out according to confidence
    windows_neg = filter(lambda window:window[1] < 0, windows)
    windows_pos = filter(lambda window:window[1] > min_confidence, windows)
    windows_dis = filter(lambda window:window[1] > 0 and window[1] < min_confidence, windows)
    
    # Draw the boxes
    img_copy = draw_bboxes(img, map(lambda window:window[0], windows_neg), (255, 0, 0), 2)
    img_copy = draw_bboxes(img_copy, map(lambda window:window[0], windows_pos), (0, 255, 0), 5)
    img_copy = draw_bboxes(img_copy, map(lambda window:window[0], windows_dis), (0, 0, 255), 2)
    
    return img_copy