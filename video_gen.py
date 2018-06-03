import cv2
import os
import argparse
import signal
import time
import numpy as np

from multiprocessing import Pool
from vehicle_detector import VehicleDetector
from moviepy.video.io.VideoFileClip import VideoFileClip
from utils import draw_bboxes, draw_windows

class VideoProcessor:

    def __init__(self, vehicle_detector, debug = False):
        self.vehicle_detector = vehicle_detector
        self.debug = debug
        self.frame_count = 0
        self.processed_frames = None

    def process_video(self, video_file, file_out, process_pool = None):

        input_clip = VideoFileClip(video_file)

        if self.debug:
            self.process_video_debug(input_clip, file_out, process_pool)
        else:
            output_clip = input_clip.fl_image(lambda frame:self.process_frame(frame, process_pool))
            output_clip.write_videofile(file_out, audio = False)

    def process_frame(self, frame, process_pool):
        
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        bboxes, _, _ = self.vehicle_detector.detect_vehicles(img, process_pool = process_pool)
        
        frame_out = draw_bboxes(img, bboxes, (0, 255, 150), 3)
        
        frame_out = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)

        return frame_out

    def process_frame_debug(self, frame, process_pool):
        
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        bboxes, heatmap, windows = self.vehicle_detector.detect_vehicles(img, process_pool = process_pool)
        
        frames_out = []

        frame_out = draw_bboxes(img, bboxes, (0, 255, 150), 3)
        frames_out.append(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB))

        heatmap = np.dstack((heatmap, np.zeros_like(heatmap), np.zeros_like(heatmap)))
        frames_out.append(heatmap)

        heatmap_o = self.vehicle_detector.heatmap(img, windows, 0, 0)
        heatmap_o = np.dstack((heatmap_o, np.zeros_like(heatmap_o), np.zeros_like(heatmap_o)))
        frames_out.append(heatmap_o)

        all_windows = []
        
        for y_min, y_max, scale, layer_windows in windows:
            layer_img = draw_windows(img, layer_windows, min_confidence = self.vehicle_detector.min_confidence)
            frames_out.append(cv2.cvtColor(layer_img, cv2.COLOR_BGR2RGB))
            all_windows.extend(layer_windows)

        box_img = draw_windows(img, all_windows, min_confidence = self.vehicle_detector.min_confidence)
        frames_out.append(cv2.cvtColor(box_img, cv2.COLOR_BGR2RGB))
        
        return frames_out
    
    def process_video_debug(self, input_clip, file_out, process_pool):

        self.processed_frames = []

        stage_idx = 0

        output_clip = input_clip.fl_image(lambda frame:self.process_frame_stage(frame, stage_idx, process_pool))
        output_clip.write_videofile(file_out, audio = False)

        if len(self.processed_frames) > 0:
            out_file_split = os.path.split(file_out)[1].split('.')
            for _ in range(len(self.processed_frames[0]) - 1):
                self.frame_count = 0
                stage_idx += 1
                stage_file = '{}.{}'.format(out_file_split[0] + '_' + str(stage_idx), out_file_split[1])
                output_clip.write_videofile(stage_file, audio = False)

    def process_frame_stage(self, frame, stage_idx, process_pool):

        if stage_idx == 0:
            result = self.process_frame_debug(frame, process_pool)
            self.processed_frames.append(result)
        
        frame_out = self.processed_frames[self.frame_count][stage_idx] 
        
        self.frame_count += 1

        return frame_out

    

def worker_init():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Video Processing')

    parser.add_argument(
        'file',
        type=str,
        help='File to process'
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

    parser.add_argument('--debug', action='store_true', help='Creates mulitple videos for each processing step')
    parser.set_defaults(debug=False)

    args = parser.parse_args()

    vehicle_detector = VehicleDetector(model_file=args.model_file, 
                         cells_per_step=args.cells_per_step, 
                         min_confidence = args.min_confidence, 
                         heat_threshold = args.threshold)

    video_processor = VideoProcessor(vehicle_detector, args.debug)    

    pool_size = os.cpu_count()

    if args.parallel is False or pool_size < 2:
        process_pool = None
    else:
        process_pool = Pool(pool_size, initializer = worker_init)

    print('Using {} cores'.format(1 if process_pool is None else pool_size))
    
    date_time_str = time.strftime('%Y%m%d-%H%M%S')

    file_out = os.path.split(args.file)[1].split('.')
    file_out = '{}.{}'.format(file_out[0] + '_processed_' + date_time_str, file_out[1])

    try:
        video_processor.process_video(args.file, file_out, process_pool=process_pool)
    except Exception as e:
        if process_pool is not None:
            process_pool.terminate()
        raise e