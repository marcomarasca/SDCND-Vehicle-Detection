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
from lane_detector import LaneDetector, ImageProcessor

class VideoProcessor:

    def __init__(self, model_file, calibration_file, min_confidence, heat_threshold, smooth_frames, detect_lanes = False, debug = False):
        self.vehicle_detector = VehicleDetector(model_file, min_confidence, heat_threshold, smooth_frames)
        self.lane_detector    = LaneDetector(smooth_frames = 5)
        self.image_processor  = ImageProcessor(calibration_file)
        self.detect_lanes     = detect_lanes
        self.debug            = debug
        self.frame_count      = 0
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

        if self.detect_lanes:
            # Uses the undistored image
            img, _, warped_img = self.image_processor.process_image(img)

        bboxes, heatmap, windows = self.vehicle_detector.detect_vehicles(img, process_pool = process_pool)

        frame_out = np.copy(img) if self.debug else img
        # Labelled image
        frame_out = draw_bboxes(frame_out, bboxes, (250, 150, 55), 1, fill = True)

        frame_out_text = 'Frame Smoothing: {}, Min Confidence: {}, Threshold: {}'.format(
            self.vehicle_detector.smooth_frames,
            self.vehicle_detector.min_confidence,
            self.vehicle_detector.heat_threshold
        )
        
        self.write_text(frame_out, frame_out_text)
        self.write_text(frame_out, 'Detected Cars: {}'.format(len(bboxes)), pos = (30, frame_out.shape[0] - 30), font_color = (0, 250, 150))
             
        if self.detect_lanes:
            _, polyfit, curvature, deviation, fail_code = self.lane_detector.detect_lanes(warped_img)
        
            fill_color = (0, 255, 0) if fail_code == 0 else (0, 255, 255)

            lane_img = self.lane_detector.draw_lanes(frame_out, polyfit, fill_color = fill_color)
            lane_img = self.image_processor.unwarp_image(lane_img)

            frame_out = cv2.addWeighted(frame_out, 1.0, lane_img, 1.0, 0)

            curvature_text = 'Left Curvature: {:.1f}, Right Curvature: {:.1f}'.format(curvature[0], curvature[1])
            offset_text = 'Center Offset: {:.2f} m'.format(deviation)

            self.write_text(frame_out, curvature_text, pos = (30, 60))
            self.write_text(frame_out, offset_text, pos = (30, 90))
            
        frame_out = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
       
        if self.debug:

            result = []

            self.write_frame_count(frame_out)
            result.append(frame_out)
            
            # Unthresholded heatmap image
            heatmap_o = self.vehicle_detector._heatmap(img, windows, 0)
            heatmap_o = self.normalize_heatmap(heatmap_o)
            heatmap_o = np.dstack((heatmap_o, np.zeros_like(heatmap_o), np.zeros_like(heatmap_o)))
            self.write_frame_count(heatmap_o)

            result.append(heatmap_o)

            # Heatmap image
            heatmap = self.normalize_heatmap(heatmap)
            heatmap = np.dstack((np.zeros_like(heatmap), np.zeros_like(heatmap), heatmap))
            self.write_frame_count(heatmap)
            
            result.append(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))

            heatmap_img = cv2.addWeighted(img, 1, heatmap, 0.8, 0)

            result.append(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB))

            all_windows = []
            
            # Windows search image
            for scale, cells_per_step, layer_windows in windows:
                
                all_windows.extend(layer_windows)

                layer_img = draw_windows(np.copy(img), layer_windows, min_confidence = self.vehicle_detector.min_confidence)
                
                w_tot = len(layer_windows)
                w_pos = len(list(filter(lambda bbox:bbox[1] >= self.vehicle_detector.min_confidence, layer_windows)))
                w_rej = len(list(filter(lambda bbox:bbox[1] > 0 and bbox[1] < self.vehicle_detector.min_confidence, layer_windows)))
                
                self.write_text(layer_img, 'Scale: {}, Cells per Steps: {}, Min Confidence: {}'.format(scale, cells_per_step, self.vehicle_detector.min_confidence))
                layer_text = 'Windows (Total/Positive/Rejected): {}/{}/{}'.format(w_tot, w_pos, w_rej)
                self.write_text(layer_img, layer_text, pos = (30, layer_img.shape[0] - 30))
                self.write_frame_count(layer_img)

                result.append(cv2.cvtColor(layer_img, cv2.COLOR_BGR2RGB))

            # Combined scales image
            box_img = draw_windows(np.copy(img), all_windows, min_confidence = self.vehicle_detector.min_confidence)

            w_tot = len(all_windows)
            w_pos = len(list(filter(lambda bbox:bbox[1] >= self.vehicle_detector.min_confidence, all_windows)))
            w_rej = len(list(filter(lambda bbox:bbox[1] > 0 and bbox[1] < self.vehicle_detector.min_confidence, all_windows)))
            
            self.write_text(box_img, 'Min Confidence: {}'.format(self.vehicle_detector.min_confidence))
            box_text = 'Windows (Total/Positive/Rejected): {}/{}/{}'.format(w_tot, w_pos, w_rej)
            self.write_text(box_img, box_text, pos = (30, layer_img.shape[0] - 30))
            self.write_frame_count(box_img)

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

    def write_frame_count(self, img):
        self.write_text(img, '{}'.format(self.frame_count), pos = (img.shape[1] - 75, 30))

    def write_text(self, img, text, pos = (30, 30), font = cv2.FONT_HERSHEY_DUPLEX, font_color = (255, 255, 255), font_size = 0.8):
        cv2.putText(img, text, pos, font, font_size, font_color, 1, cv2.LINE_AA)

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
        '--min_confidence',
        type=float,
        default=0.5,
        help='Min prediction confidence for bounding boxes'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=3.5,
        help='Heatmap threshold'
    )

    parser.add_argument(
        '--smooth_frames',
        type=int,
        default=8,
        help='How many frames to use for smoothing the resulting heatmap'
    )

    parser.add_argument(
        '--calibration_file',
        type=str,
        default=os.path.join('calibration.p'),
        help='Calibration data file'
    )

    parser.add_argument('--disable-parallel', dest='parallel', action='store_false', help='Disable parallel processing (may decrease feature extraction speed)')
    parser.set_defaults(parallel=True)

    parser.add_argument('--debug', action='store_true', help='Creates mulitple videos for each processing step')
    parser.set_defaults(debug=False)

    parser.add_argument('--lanes-detection', dest='detect_lanes', action='store_true', help='Detect lane lines')
    parser.set_defaults(detect_lanes=False)

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    video_processor = VideoProcessor(args.model_file,
                                     args.calibration_file, 
                                     args.min_confidence,
                                     args.threshold,
                                     args.smooth_frames, 
                                     args.detect_lanes, 
                                     args.debug)

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