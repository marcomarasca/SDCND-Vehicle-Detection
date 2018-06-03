import numpy as np
import glob
import time
import cv2
import os
import signal
import argparse
import matplotlib as mpl
# For plotting without a screen
mpl.use('Agg')
import matplotlib.pyplot as plt

from multiprocessing import Pool
from vehicle_detector import VehicleDetector
from multiprocessing import Pool
from utils import draw_bboxes, draw_windows
from tqdm import tqdm

def process_img(vd, img_file, out_dir = 'output_images', process_pool = None):

    img = cv2.imread(img_file)
    
    t1 = time.time()
    bounding_boxes, heatmap, windows = vd.detect_vehicles(img, process_pool=process_pool)
    t2 = time.time()
    
    plt.figure(figsize = (20, 15))
    
    rows = np.ceil((len(windows) + 1) / 2)

    i = 1
    
    all_bboxes = []
    for y_min, y_max, scale, bboxes in windows:
        
        i += 1
        plt.subplot(rows, 2, i)
        plt.title('Window Search (Scale: {})'.format(scale))
        box_img = draw_windows(img, bboxes, min_confidence = vd.min_confidence)
        plt.imshow(cv2.cvtColor(box_img, cv2.COLOR_BGR2RGB))
        all_bboxes.extend(bboxes)
        
    plt.subplot(rows, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
    box_img = draw_windows(img, all_bboxes, min_confidence = vd.min_confidence)
    plt.subplot(rows, 2, i + 1)
    plt.title('Combined Windows (Min Confidence: {})'.format(vd.min_confidence))
    plt.imshow(cv2.cvtColor(box_img, cv2.COLOR_BGR2RGB))
        
    plt.tight_layout()
    img_prefix = os.path.split(img_file)[-1].split('.')[0]
    plt.savefig(os.path.join(out_dir,  img_prefix + '_window_search.png'))
    
    plt.figure(figsize = (20, 10))
    
    plt.subplot(221)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    plt.subplot(222)
    plt.title('Combined Windows (Min Confidence: {})'.format(vd.min_confidence))
    plt.imshow(cv2.cvtColor(box_img, cv2.COLOR_BGR2RGB))
    plt.subplot(223)
    plt.title('Heatmap')
    heatmap_o = vd.heatmap(img, windows, 0, 0)
    plt.imshow(heatmap_o, cmap = 'hot')
    
    plt.subplot(224)
    plt.title('Heatmap (Min confidence: {} , Threshold: {})'.format(vd.min_confidence, vd.heat_threshold))
    plt.imshow(heatmap, cmap = 'hot')
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(out_dir,  img_prefix + '_heatmap.png'))
    
    plt.figure(figsize = (20, 10))
    
    plt.subplot(221)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    plt.subplot(222)
    plt.title('Combined Windows (Min Confidence: {})'.format(vd.min_confidence))
    plt.imshow(cv2.cvtColor(box_img, cv2.COLOR_BGR2RGB))
    plt.subplot(223)
    plt.title('Heatmap (Min confidence: {} , Threshold: {})'.format(vd.min_confidence, vd.heat_threshold))
    plt.imshow(heatmap, cmap = 'hot')
    
    labeled_img = draw_bboxes(img, bounding_boxes, (100, 255, 0), 3)
    
    plt.subplot(224)
    plt.title('Labeled Image')
    plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,  img_prefix + '_labeled.png'))

    return t2 - t1


def worker_init():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Image Processing')

    parser.add_argument(
        '--dir',
        type=str,
        default='test_images',
        help='Images folder'
    )

    parser.add_argument(
        '--model_file',
        type=str,
        default=os.path.join('models', 'model.p'),
        help='Images folder'
    )

    parser.add_argument(
        '--cells_per_step',
        type=int,
        default=1,
        help='Number of cells per steps'
    )

    parser.add_argument(
        '--min_confidence',
        type=float,
        default=0.3,
        help='Min prediction confidence for bounding boxes'
    )

    parser.add_argument(
        '--threshold',
        type=int,
        default=5,
        help='Heatmap threshold'
    )

    parser.add_argument('--disable-parallel', dest='parallel', action='store_false', help='Disable parallel processing (may decrease feature extraction speed)')
    parser.set_defaults(parallel=True)

    args = parser.parse_args()

    imgs = glob.glob(os.path.join(args.dir, '*.jpg'))

    vd = VehicleDetector(model_file=args.model_file, 
                        cells_per_step=args.cells_per_step, 
                        min_confidence = args.min_confidence, 
                        heat_threshold = args.threshold)

    pool_size = os.cpu_count()

    if args.parallel is False or pool_size < 2:
        process_pool = None
    else:
        process_pool = Pool(pool_size, initializer = worker_init)

    print('Using {} cores'.format(1 if process_pool is None else pool_size))

    try:

        t = 0
        
        for img_file in tqdm(imgs, unit = ' images', desc = 'Image Processing'):

            img_t = process_img(vd, img_file, process_pool=process_pool)

            t += img_t
        
        print('Total/Average time: {:.3f}/{:.3f} s'.format(t, t/len(imgs)))

    except Exception as e:
        if process_pool is not None:
            process_pool.terminate()
        raise e
    