import numpy as np
import cv2

def draw_bboxes(img, bboxes, color, thick, fill = False):
    
    img_copy = np.copy(img)

    for bbox in bboxes:
        cv2.rectangle(img_copy, bbox[0], bbox[1], color, thick)

    if fill:
        img_fill = np.zeros_like(img_copy)
        for bbox in np.array(bboxes):
            cv2.rectangle(img_fill, tuple(bbox[0] + thick), tuple(bbox[1] - thick), color, -1)
        img_copy = cv2.addWeighted(img_copy, 1, img_fill, 0.7, 0)
    
    return img_copy

def draw_windows(img, windows, min_confidence):
    
    # Filters out according to confidence
    windows_neg = filter(lambda window:window[1] < 0, windows)
    windows_pos = filter(lambda window:window[1] >= min_confidence, windows)
    windows_rej = filter(lambda window:window[1] > 0 and window[1] < min_confidence, windows)

    # Draw the boxes
    img_copy = draw_bboxes(img, map(lambda window:window[0], windows_neg), (255, 0, 0), 2)
    img_copy = draw_bboxes(img_copy, map(lambda window:window[0], windows_pos), (0, 255, 0), 5)
    img_copy = draw_bboxes(img_copy, map(lambda window:window[0], windows_rej), (0, 0, 255), 2)
    
    return img_copy