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

    def process_video(self, video_file, file_out, t_start = None, t_end = None, process_pool = None):

        input_clip = VideoFileClip(video_file)

        if t_start is not None:
            input_clip = input_clip.subclip(t_start = t_start, t_end = t_end)

        if self.debug:
            self.processed_frames = []

            stage_idx = 0

            output_clip = input_clip.fl_image(lambda frame:self.process_frame_stage(frame, stage_idx, process_pool))
            output_clip.write_videofile(file_out, audio = False)

            if len(self.processed_frames) > 0:
                out_file_path = os.path.split(file_out)
                out_file_name = out_file_path[1].split('.')
                for _ in range(len(self.processed_frames[0]) - 1):
                    self.frame_count = 0
                    stage_idx += 1
                    stage_file = '{}.{}'.format(os.path.join(out_file_path[0], out_file_name[0]) + '_' + str(stage_idx), out_file_name[1])
                    output_clip.write_videofile(stage_file, audio = False)
        else:
            output_clip = input_clip.fl_image(lambda frame:self.process_frame(frame, process_pool))
            output_clip.write_videofile(file_out, audio = False)

    def process_frame(self, frame, process_pool):
        
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        bboxes, heatmap, windows = self.vehicle_detector.detect_vehicles(img, process_pool = process_pool)

        # Labelled image
        frame_out = draw_bboxes(img, bboxes, (0, 255, 150), 2, fill = True)
        self.write_text(frame_out, 'Detected Cars: {}'.format(len(bboxes)))
        frame_out = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
       
        if self.debug:

            result = []

            result.append(frame_out)
            
            # Unthresholded heatmap image
            heatmap_o = self.vehicle_detector._heatmap(img, windows, 0)
            heatmap_o = self.normalize_heatmap(heatmap_o)
            heatmap_o = np.dstack((heatmap_o, np.zeros_like(heatmap_o), np.zeros_like(heatmap_o)))

            result.append(heatmap_o)

            # Heatmap image
            heatmap = self.normalize_heatmap(heatmap)
            heatmap = np.dstack((np.zeros_like(heatmap), np.zeros_like(heatmap), heatmap))
            
            result.append(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))

            mixed_img = cv2.addWeighted(img, 1, heatmap, 0.8, 0)
            result.append(cv2.cvtColor(mixed_img, cv2.COLOR_BGR2RGB))

            all_windows = []
            
            # Windows search image
            for _, _, scale, layer_windows in windows:
                
                all_windows.extend(layer_windows)

                layer_img = draw_windows(img, layer_windows, min_confidence = self.vehicle_detector.min_confidence)
                
                w_tot = len(layer_windows)
                w_pos = len(list(filter(lambda bbox:bbox[1] >= self.vehicle_detector.min_confidence, layer_windows)))
                w_rej = len(list(filter(lambda bbox:bbox[1] > 0 and bbox[1] < self.vehicle_detector.min_confidence, layer_windows)))
                
                layer_text = 'Scale: {}, Windows (Total/Positive/Rejected): {}/{}/{}'.format(scale, w_tot, w_pos, w_rej)
                self.write_text(layer_img, layer_text)

                result.append(cv2.cvtColor(layer_img, cv2.COLOR_BGR2RGB))

            # Combined scales image
            box_img = draw_windows(img, all_windows, min_confidence = self.vehicle_detector.min_confidence)

            w_tot = len(all_windows)
            w_pos = len(list(filter(lambda bbox:bbox[1] >= self.vehicle_detector.min_confidence, all_windows)))
            w_rej = len(list(filter(lambda bbox:bbox[1] > 0 and bbox[1] < self.vehicle_detector.min_confidence, all_windows)))
            
            box_text = 'Min Confidence: {}, Windows (Total/Positive/Rejected): {}/{}/{}'.format(self.vehicle_detector.min_confidence, w_tot, w_pos, w_rej)
            self.write_text(box_img, box_text)

            result.append(cv2.cvtColor(box_img, cv2.COLOR_BGR2RGB))            

        else:
            result = frame_out

        return result

    def process_frame_stage(self, frame, stage_idx, process_pool):

        if stage_idx == 0:
            result = self.process_frame(frame, process_pool)
            self.processed_frames.append(result)
        
        frame_out = self.processed_frames[self.frame_count][stage_idx] 
        
        self.frame_count += 1

        return frame_out

    def normalize_heatmap(self, heatmap, a = 0, b = 255):

        min_v = np.min(heatmap)
        max_v = np.max(heatmap)

        heatmap = a + ((heatmap - min_v) * (b - a)) / (max_v - min_v)

        return heatmap.astype(np.uint8)

    def write_text(self, img, text, pos = (30,60), font = cv2.FONT_HERSHEY_DUPLEX, font_color = (0, 255, 0)):
        cv2.putText(img, text, pos, font, 1, font_color, 1)

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
        '--output_dir',
        type=str,
        default='output_videos',
        help='Where to store the processed video'
    )

    parser.add_argument(
        '--start',
        type=float,
        default=None,
        help='Start time for subclipping'
    )

    parser.add_argument(
        '--end',
        type=float,
        default=None,
        help='End time for subclipping'
    )

    parser.add_argument(
        '--model_file',
        type=str,
        default=os.path.join('models', 'model.p'),
        help='Model file path'
    )

    parser.add_argument(
        '--cells_per_step',
        type=int,
        default=2,
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
        default=10,
        help='Heatmap threshold'
    )

    parser.add_argument(
        '--smooth_frames',
        type=int,
        default=5,
        help='How many frames to use for smoothing the resulting heatmap'
    )

    parser.add_argument('--disable-parallel', dest='parallel', action='store_false', help='Disable parallel processing (may decrease feature extraction speed)')
    parser.set_defaults(parallel=True)

    parser.add_argument('--debug', action='store_true', help='Creates mulitple videos for each processing step')
    parser.set_defaults(debug=False)

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    vehicle_detector = VehicleDetector(model_file     = args.model_file, 
                                       cells_per_step = args.cells_per_step, 
                                       min_confidence = args.min_confidence, 
                                       heat_threshold = args.threshold,
                                       smooth_frames  = args.smooth_frames)

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
    file_out = os.path.join(args.output_dir, file_out)

    try:
        video_processor.process_video(args.file, file_out, t_start = args.start, t_end = args.end, process_pool=process_pool)
    except Exception as e:
        if process_pool is not None:
            process_pool.terminate()
        raise e