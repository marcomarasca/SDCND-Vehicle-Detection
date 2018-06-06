import numpy as np
import cv2

def draw_bboxes(img, bboxes, color, thick, fill = False):

    if len(bboxes) == 0:
        return img

    for bbox in bboxes:
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)

    if fill:
        img_fill = np.zeros_like(img)
        for bbox in np.array(bboxes):
            cv2.rectangle(img_fill, tuple(bbox[0] + thick), tuple(bbox[1] - thick), color, -1)
        img = cv2.addWeighted(img, 1, img_fill, 0.7, 0)
    
    return img

def draw_windows(img, windows, min_confidence, lines_thick = (1, 3, 2)):
    
    # Filters out according to confidence
    windows_neg = filter(lambda window:window[1] < 0, windows)
    windows_pos = filter(lambda window:window[1] >= min_confidence, windows)
    windows_rej = filter(lambda window:window[1] > 0 and window[1] < min_confidence, windows)

    # Draw the boxes
    img = draw_bboxes(img, list(map(lambda window:window[0], windows_neg)), (255, 0, 0), lines_thick[0])
    img = draw_bboxes(img, list(map(lambda window:window[0], windows_pos)), (0, 255, 0), lines_thick[1], fill = True)
    img = draw_bboxes(img, list(map(lambda window:window[0], windows_rej)), (0, 0, 255), lines_thick[2])
    
    return img