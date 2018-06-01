import os
import csv
import numpy as np
import cv2
import argparse

from tqdm import tqdm

def process_udacity_dataset(map_file = os.path.join('data', 'udacity', 'labels_crowdai.csv'),
                            img_folder = os.path.join('data', 'udacity', 'object-detection-crowdai'),
                            dest_folder = os.path.join('data', 'udacity'),
                            dest_size = 64,
                            format = 'png',
                            limit = None,
                            skip = 5):
    
    print('Loading Udacity CSV file from: {}'.format(map_file))

    vehicles_folder = os.path.join(dest_folder, 'vehicles')
    
    if not os.path.isdir(vehicles_folder):
        os.makedirs(vehicles_folder)

    notvehicles_folder = os.path.join(dest_folder, 'non-vehicles')

    if not os.path.isdir(notvehicles_folder):
        os.makedirs(notvehicles_folder)


    file_bboxes = {}

    with open(map_file, 'r') as csv_file:

        lines = [line for line in csv_file]

        if limit is not None:
            lines = lines[:limit]

        reader = csv.reader(lines)

        # Skip header
        _ = next(reader, None)

        for x_min, y_min, x_max, y_max, img_file, label, _ in filter(lambda row:row[5] in ['Car', 'Truck'], tqdm(reader, total = len(lines), unit = ' images', desc = 'Parsing Vehicles')):
            
            img = cv2.imread(os.path.join(img_folder, img_file))

            x_min = int(x_min)
            x_max = int(x_max)
            y_min = int(y_min)
            y_max = int(y_max)

            if y_max <= y_min or x_max <= x_min:
                print('Wrong bounding box for {}: ({}, {}), ({}, {})'.format(img_file, x_min, y_min, x_max, y_max))
                continue

            bboxes = file_bboxes.get(img_file)

            if bboxes is None:
                bboxes = []
                file_bboxes[img_file] = bboxes

            bboxes.append(((x_min, y_min), (x_max, y_max)))
            
    print('Processed images: {}'.format(len(file_bboxes)))

    for i, (img_file, bboxes) in enumerate(tqdm(file_bboxes.items(), unit=' images', desc='Saving images')):

        if i % skip == 0:

            img = cv2.imread(os.path.join(img_folder, img_file))
            
            _save_windows(img, bboxes, vehicles_folder, img_file, dest_size, format)

            notvehicle_boxes = _get_notvehicles_bboxes(bboxes, img.shape)
            
            _save_windows(img, notvehicle_boxes, notvehicles_folder, img_file, dest_size, format)

def _save_windows(img, bboxes, folder, img_file, dest_size, format):

    for i, bbox in enumerate(bboxes):
        img_window = img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]

        interpolation = cv2.INTER_CUBIC
        
        if img_window.shape[0] < dest_size or img_window.shape[1] < dest_size:
            interpolation = cv2.INTER_AREA

        img_window = cv2.resize(img_window, (dest_size, dest_size), interpolation = interpolation)
        dest_file = os.path.join(folder, '{}_{}.{}'.format(img_file.split('.')[0], i,  format))
        cv2.imwrite(dest_file, img_window)

def _get_notvehicles_bboxes(taken_bboxes, img_shape):
    window_size = np.random.choice([128, 256])
    stride = int(window_size * 0.8)
    y_gap = 120 # Skip the hood

    bboxes = []
    max_notvehicles = len(taken_bboxes)
    for x in range(0, img_shape[1], stride):
        # Skip x with 1/3 probability
        if np.random.choice([True, False], replace = False, p = [1/3, 2/3]):
            continue
        for y in range(img_shape[0] - y_gap, window_size, -stride):
            # Skip y with 1/3 probability
            if np.random.choice([True, False], replace = False, p = [1/3, 2/3]):
                continue
            bbox = ((x, y - window_size), (x + window_size, y))
            if _is_free_box(bbox, taken_bboxes):
                bboxes.append(bbox)
            if len(bboxes) >= max_notvehicles:
                return bboxes

    return bboxes

def _is_free_box(bbox, bboxes):

    for taken_bbox in bboxes:
        if _overlap(bbox, taken_bbox):
            return False

    return True

def _overlap(bbox1, bbox2):
    if bbox1[1][0] <= bbox2[0][0]: return False # bbox1 is left of bbox2
    if bbox1[0][0] >= bbox2[1][0]: return False # bbox1 is right of bbox2
    if bbox1[1][1] <= bbox2[0][1]: return False # bbox1 is above bbox2
    if bbox1[0][1] >= bbox2[1][1]: return False # bbox1 is below bbox2

    return True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SVC Training')

    parser.add_argument(
        '--map_file',
        type=str,
        default=os.path.join('data', 'udacity', 'labels_crowdai.csv'),
        help='Mapping CSV file'
    )
    parser.add_argument(
        '--img_folder',
        type=str,
        default=os.path.join('data', 'udacity', 'object-detection-crowdai'),
        help='Folder where the original images are stored'
    )
    parser.add_argument(
        '--dest_folder',
        type=str,
        default=os.path.join('data', 'udacity'),
        help='Folder where to save the cropped images'
    )
    parser.add_argument(
        '--dest_size',
        type=int,
        default=64,
        help='What size are the destination images'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='png',
        help='Which format to save the cropped images'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit to the given amount of images'
    )
    parser.add_argument(
        '--skip',
        type=int,
        default=5,
        help='Only process every x images (e.g. to avoid time series data)'
    )

    args = parser.parse_args()

    process_udacity_dataset(map_file=args.map_file, 
                            img_folder=args.img_folder, 
                            dest_folder=args.dest_folder, 
                            dest_size=args.dest_size, 
                            format=args.format, 
                            limit=args.limit,
                            skip=args.skip)
    