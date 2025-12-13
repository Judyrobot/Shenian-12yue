#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import time
import math
import numpy as np
import socket
import struct
import logging
from typing import List, Tuple, Dict, Optional, Any, Union
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Constants:
    BEACON_KEYPOINT_INDEX = 0
    INVALID_VALUE = -1000000
    MAX_BEACON_ID = 5
    
    UNAUTHENTICATED_COLOR = (255, 0, 0)
    PREDICTED_COLOR = (128, 128, 128)
    AUTHENTICATED_COLOR = (0, 255, 0)
    NEW_AUTHENTICATED_COLOR = (0, 255, 255)
    NEW_AUTHENTICATED_DURATION = 30
    OUTLIER_COLOR = (0, 0, 255)


class KalmanFilter:
    def __init__(self, dt: float = 0.1, sigma_a: float = 0.05):
        self.dt = dt
        self.sigma_a = sigma_a
        self.x = np.zeros((4, 1))
        self.P = np.eye(4)
        self.F = np.array([[1, 0, dt, 0],
                          [0, 1, 0, dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
        
        self.H = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]])
        
        G = np.array([[0.5 * dt**2], [0.5 * dt**2], [dt], [dt]])
        self.Q = G @ G.T * (self.sigma_a ** 2)
        
        self.initialized = False

    def initialize(self, x: float, y: float) -> None:
        self.x = np.array([[x], [y], [0], [0]])
        self.P = np.diag([1.0, 1.0, 10.0, 10.0])
        self.initialized = True
        
    def predict(self) -> np.ndarray:
        if not self.initialized:
            return np.zeros((2, 1))
            
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]
    
    def update(self, z: np.ndarray, detection_score: float = 0.5) -> np.ndarray:
        if not self.initialized:
            self.initialize(z[0, 0], z[1, 0])
            return self.x[:2]
        
        base_noise = 1.0
        noise_scale = base_noise / (detection_score + 0.1)
        self.R = np.array([[noise_scale, 0], [0, noise_scale]])
        
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return self.x[:2]


class BeaconTracker:
    def __init__(self, max_missing: int = 3, dt: float = 0.1):
        self.max_missing = max_missing
        self.dt = dt
        
        self.kalman_filters: Dict[int, KalmanFilter] = {
            i: KalmanFilter(dt) for i in range(Constants.MAX_BEACON_ID + 1)
        }
        
        self.missing_count: Dict[int, int] = {
            i: 0 for i in range(Constants.MAX_BEACON_ID + 1)
        }
        
        self.recent_measurements: Dict[int, List[Tuple[int, int]]] = {
            i: [] for i in range(Constants.MAX_BEACON_ID + 1)
        }
        self.max_recent_measurements = 5

    def update(self, beacon_id: int, x: int, y: int, detection_score: float = 0.5) -> None:
        if not (0 <= beacon_id <= Constants.MAX_BEACON_ID):
            return
            
        measurement = np.array([[x], [y]], dtype=np.float32)
        self.kalman_filters[beacon_id].update(measurement, detection_score=detection_score)
        
        self.missing_count[beacon_id] = 0
        
        self.recent_measurements[beacon_id].append((x, y))
        if len(self.recent_measurements[beacon_id]) > self.max_recent_measurements:
            self.recent_measurements[beacon_id].pop(0)
            
    def predict(self, beacon_id: int) -> Optional[Tuple[int, int]]:
        if not (0 <= beacon_id <= Constants.MAX_BEACON_ID):
            return None
            
        kf = self.kalman_filters[beacon_id]
        if not kf.initialized or self.missing_count[beacon_id] > self.max_missing:
            return None
            
        predicted_pos = kf.predict()
        x, y = int(round(predicted_pos[0, 0])), int(round(predicted_pos[1, 0]))
        
        return (x, y)

    def mark_missing(self, beacon_id: int) -> None:
        if 0 <= beacon_id <= Constants.MAX_BEACON_ID:
            self.missing_count[beacon_id] += 1
            
    def reset(self) -> None:
        self.kalman_filters = {
            i: KalmanFilter(self.dt) for i in range(Constants.MAX_BEACON_ID + 1)
        }
        self.missing_count = {
            i: 0 for i in range(Constants.MAX_BEACON_ID + 1)
        }
        self.recent_measurements = {
            i: [] for i in range(Constants.MAX_BEACON_ID + 1)
        }
        
    def get_velocity(self, beacon_id: int) -> Tuple[float, float]:
        if not (0 <= beacon_id <= Constants.MAX_BEACON_ID) or not self.kalman_filters[beacon_id].initialized:
            return (0.0, 0.0)
            
        vx = self.kalman_filters[beacon_id].x[2, 0]
        vy = self.kalman_filters[beacon_id].x[3, 0]
        return (vx, vy)


class UDPCommunicator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.udp_socket: Optional[socket.socket] = None
        self.udp_server_address = (config['ip'], config['port'])
        self.init_udp()
        
    def init_udp(self) -> None:
        try:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            logger.info(f"UDP client initialized, sending to {self.config['ip']}:{self.config['port']}")
        except Exception as e:
            logger.error(f"UDP initialization failed: {str(e)}")
            self.udp_socket = None

    def send_data(self, x: float, y: float, z: float,
                 yaw: float, pitch: float, roll: float,
                 avg_center_x: float = Constants.INVALID_VALUE, 
                 avg_center_y: float = Constants.INVALID_VALUE) -> bool:
        if not self.udp_socket:
            return False
            
        try:
            if x == Constants.INVALID_VALUE or y == Constants.INVALID_VALUE or z == Constants.INVALID_VALUE:
                if avg_center_x != Constants.INVALID_VALUE and avg_center_y != Constants.INVALID_VALUE:
                    data = [
                        int(round(avg_center_x)),
                        int(round(avg_center_y)),
                        Constants.INVALID_VALUE,
                        Constants.INVALID_VALUE,
                        Constants.INVALID_VALUE,
                        Constants.INVALID_VALUE
                    ]
                else:
                    return False
            else:
                data = [
                    int(round(x)),
                    int(round(y)),
                    int(round(z)),
                    int(round(yaw * 100)),
                    int(round(pitch * 100)),
                    int(round(roll * 100))
                ]
                
            head1 = b'#'
            head2 = b'a'
            message = struct.pack("<2c6i", head1, head2, *data)
            
            self.udp_socket.sendto(message, self.udp_server_address)
            return True
        except Exception as e:
            logger.error(f"UDP send failed: {str(e)}")
            return False
            
    def close(self) -> None:
        if self.udp_socket:
            self.udp_socket.close()
            logger.info("UDP connection closed")


class VideoProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.display = config.get('display', True)
        self.output_video_path = config.get('output_video_path', 'output_with_fps.avi')
        
        self.save_frames = config.get('save_frames', True)
        self.frames_save_path = config.get('frames_save_path', 'processed_frames')
        if self.save_frames and not os.path.exists(self.frames_save_path):
            os.makedirs(self.frames_save_path)
        
        self.frame_width = 0
        self.frame_height = 0
        self.out_width = 0
        self.out_height = 0
        self.fps = 30
        
        self.split_interval_minutes = config.get('split_interval_minutes', 20)
        self.split_interval_seconds = self.split_interval_minutes * 60
        self.start_time = None
        self.current_video_path = None
        self.frame_count = 0
        
        self.video_output_dir = config.get('video_output_dir', 'output_videos')
        if not os.path.exists(self.video_output_dir):
            os.makedirs(self.video_output_dir)
        
    def create_new_video_writer(self) -> bool:
        current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.current_video_path = os.path.join(
            self.video_output_dir, 
            f"video_{current_time}.avi"
        )
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(
            self.current_video_path,
            fourcc,
            self.fps,
            (self.out_width, self.out_height)
        )
        
        if not self.writer.isOpened():
            logger.error(f"Failed to initialize video writer, check path: {self.current_video_path}")
            return False
            
        self.start_time = time.time()
        self.frame_count = 0
        logger.info(f"Started recording new video: {self.current_video_path}")
        return True
    
    def check_video_split(self) -> bool:
        if self.start_time is None:
            return False
            
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.split_interval_seconds:
            logger.info(f"Video recording time reached {self.split_interval_minutes} minutes, creating new video file")
            if self.writer:
                self.writer.release()
            return self.create_new_video_writer()
        return False
    
    def initialize(self, video_source: Union[str, int]) -> bool:
        self.video_source = video_source
        
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {video_source}")
            return False
            
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        
        if isinstance(video_source, int):
            custom_width = self.config.get('custom_video_width')
            custom_height = self.config.get('custom_video_height')
            
            if custom_width and custom_height:
                logger.info(f"Attempting to set camera resolution to {custom_width}x{custom_height}")
                
                original_width = self.frame_width
                original_height = self.frame_height
                
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, custom_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, custom_height)
                
                new_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                new_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if new_width == custom_width and new_height == custom_height:
                    logger.info(f"Successfully set camera resolution to {new_width}x{new_height}")
                    self.frame_width = new_width
                    self.frame_height = new_height
                else:
                    logger.warning(f"Failed to set camera resolution to {custom_width}x{custom_height}")
                    logger.warning(f"Camera supports resolution {new_width}x{new_height}")
                    self.frame_width = original_width
                    self.frame_height = original_height
        
        if self.config.get('downscale', True):
            scale = 0.75
            self.out_width = int(self.frame_width * scale)
            self.out_height = int(self.frame_height * scale)
        else:
            self.out_width = self.frame_width
            self.out_height = self.frame_height
            
        return self.create_new_video_writer()
        
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.cap:
            return False, None
        return self.cap.read()
        
    def process_and_display(self, frame: np.ndarray, overlay_func, *args, **kwargs) -> bool:
        overlay_func(frame, *args, **kwargs)
        
        if self.display:
            cv2.imshow("High FPS Pose Estimation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
                
        self.check_video_split()
        
        if self.writer is not None:
            if frame.shape[1] != self.out_width or frame.shape[0] != self.out_height:
                frame = cv2.resize(frame, (self.out_width, self.out_height))
            self.writer.write(frame)
            self.frame_count += 1
            
        if self.save_frames and 'frame_idx' in kwargs:
            frame_filename = os.path.join(self.frames_save_path, f"frame_{kwargs['frame_idx']:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
                
        return True
        
    def get_supported_resolutions(self) -> List[Tuple[int, int]]:
        if not self.cap or not isinstance(self.video_source, int):
            return []
            
        supported_resolutions = []
        tested_resolutions = [
            (3840, 2160), (2560, 1440), (1920, 1080), (1280, 720),
            (1024, 768), (800, 600), (640, 480), (480, 360), (320, 240)
        ]
        
        current_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        current_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        logger.info("Detecting supported camera resolutions...")
        
        for width, height in tested_resolutions:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            ret, _ = self.cap.read()
            if ret:
                set_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                set_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if set_width == width and set_height == height:
                    resolution = (width, height)
                    if resolution not in supported_resolutions:
                        supported_resolutions.append(resolution)
                        logger.info(f"  Supported: {width}x{height}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, current_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, current_height)
        
        self.cap.read()
        
        logger.info(f"Detected {len(supported_resolutions)} supported resolutions")
        return supported_resolutions
    
    def release(self) -> None:
        if self.cap:
            self.cap.release()
            
        if self.writer:
            self.writer.release()
            
        cv2.destroyAllWindows()


def map_detections_to_beacons(centroids: np.ndarray, light_num: int,
                             img_shape: Tuple[int, int]) -> Dict[int, Tuple[float, float]]:
    return old_map_detections_to_beacons(centroids, light_num, img_shape)


def old_map_detections_to_beacons(centroids: np.ndarray, light_num: int,
                                img_shape: Tuple[int, int]) -> Dict[int, Tuple[float, float]]:
    if light_num < 3:
        return {}
        
    centroids = centroids.astype(np.float32)
    num_labels = centroids.shape[0] - 1
    if num_labels < 3:
        return {}
        
    centers = centroids[1:, :].astype(np.float32)
    if len(centers) < 3:
        return {}
        
    arg_rowsort = np.argsort(centers[:, 1], axis=0)
    arg_colsort = np.argsort(centers[:, 0], axis=0)
    
    uv_map: Dict[int, Tuple[float, float]] = {}
    
    if light_num == 6:
        uv_map[1] = tuple(centers[arg_colsort[-1]])
        uv_map[4] = tuple(centers[arg_colsort[0]])
        
        bottom_two = centers[arg_rowsort[-2:]]
        if bottom_two[0, 0] > bottom_two[1, 0]:
            uv_map[2], uv_map[3] = tuple(bottom_two[0]), tuple(bottom_two[1])
        else:
            uv_map[2], uv_map[3] = tuple(bottom_two[1]), tuple(bottom_two[0])
        
        top_two = centers[arg_rowsort[:2]]
        if top_two[0, 0] < top_two[1, 0]:
            uv_map[5], uv_map[0] = tuple(top_two[0]), tuple(top_two[1])
        else:
            uv_map[5], uv_map[0] = tuple(top_two[1]), tuple(top_two[0])
            
    elif light_num == 5:
        if len(centers) >= 2:
            uv_map[1] = tuple(centers[arg_colsort[-1]])
            uv_map[4] = tuple(centers[arg_colsort[0]])
        
        if len(centers) >= 4:
            bottom_two = centers[arg_rowsort[-2:]]
            if len(bottom_two) >= 2:
                if bottom_two[0, 0] > bottom_two[1, 0]:
                    uv_map[2], uv_map[3] = tuple(bottom_two[0]), tuple(bottom_two[1])
                else:
                    uv_map[2], uv_map[3] = tuple(bottom_two[1]), tuple(bottom_two[0])
        
        if len(centers) >= 1:
            top_one = centers[arg_rowsort[0]]
            uv_map[5] = tuple(top_one)
    
    else:
        if len(centers) >= 2:
            uv_map[1] = tuple(centers[arg_colsort[-1]])
            uv_map[4] = tuple(centers[arg_colsort[0]])
        
        if len(centers) >= 4:
            bottom_two = centers[arg_rowsort[-2:]] if len(arg_rowsort)>=2 else []
            if len(bottom_two)>=2:
                if bottom_two[0, 0] > bottom_two[1, 0]:
                    uv_map[2], uv_map[3] = tuple(bottom_two[0]), tuple(bottom_two[1])
                else:
                    uv_map[2], uv_map[3] = tuple(bottom_two[1]), tuple(bottom_two[0])
        
        if len(centers) >= 2:
            top_two = centers[arg_rowsort[:2]] if len(arg_rowsort)>=2 else []
            if len(top_two)>=2:
                if top_two[0, 0] < top_two[1, 0]:
                    uv_map[5], uv_map[0] = tuple(top_two[0]), tuple(top_two[1])
                else:
                    uv_map[5], uv_map[0] = tuple(top_two[1]), tuple(top_two[0])
    
    h, w = img_shape
    valid_map: Dict[int, Tuple[float, float]] = {}
    for bid in uv_map:
        x, y = uv_map[bid]
        if 0 <= x < w and 0 <= y < h:
            valid_map[bid] = (x, y)
            
    return valid_map



def geometric_map_detections_to_beacons(centroids: np.ndarray, light_num: int,
                                       img_shape: Tuple[int, int],
                                       beacon_world_coords: np.ndarray,
                                       angle_threshold: float = 20.0,
                                       distance_threshold: float = 0.2) -> Dict[int, Tuple[float, float]]:
    if light_num < 4:
        logger.debug(f"Insufficient beacons detected ({light_num}), using old method")
        return old_map_detections_to_beacons(centroids, light_num, img_shape)
        
    centroids = centroids.astype(np.float32)
    num_labels = centroids.shape[0] - 1
    if num_labels < 4:
        logger.debug(f"Insufficient valid beacons ({num_labels}), using old method")
        return old_map_detections_to_beacons(centroids, light_num, img_shape)
        
    centers = centroids[1:, :].astype(np.float32)
    if len(centers) < 4:
        logger.debug(f"Insufficient center points ({len(centers)}), using old method")
        return old_map_detections_to_beacons(centroids, light_num, img_shape)
    
    logger.debug(f"Using circular geometry-based beacon matching, detected {len(centers)} beacons")
    
    centroid = np.mean(centers, axis=0)
    logger.debug(f"Beacon centroid: {centroid}")
    
    points_info = []
    for i, (x, y) in enumerate(centers):
        angle_rad = math.atan2(y - centroid[1], x - centroid[0])
        angle_deg = np.rad2deg(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
        
        distance = math.hypot(x - centroid[0], y - centroid[1])
        
        points_info.append({
            'index': i,
            'coords': (x, y),
            'angle_rad': angle_rad,
            'angle_deg': angle_deg,
            'distance': distance
        })
    
    symmetric_pairs = []
    used = set()
    
    for i in range(len(points_info)):
        if i in used:
            continue
            
        for j in range(i + 1, len(points_info)):
            if j in used:
                continue
                
            angle_diff = abs(points_info[i]['angle_deg'] - points_info[j]['angle_deg'])
            if abs(angle_diff - 180) < angle_threshold:
                distance_ratio = min(points_info[i]['distance'], points_info[j]['distance']) / \
                               max(points_info[i]['distance'], points_info[j]['distance'])
                
                if distance_ratio > (1 - distance_threshold):
                    symmetric_pairs.append((i, j))
                    used.add(i)
                    used.add(j)
                    logger.debug(f"Found symmetric pair: {i} <-> {j}, angle diff: {angle_diff:.1f}°, distance ratio: {distance_ratio:.3f}")
                    break
    
    if not symmetric_pairs:
        logger.debug("No symmetric pairs found, using old method")
        return old_map_detections_to_beacons(centroids, light_num, img_shape)
    
    theoretical_angles = {}
    for bid in range(6):
        x, y, _ = beacon_world_coords[bid]
        angle_rad = math.atan2(y, x)
        angle_deg = np.rad2deg(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
        theoretical_angles[bid] = angle_deg
    
    logger.debug("Theoretical angle template:")
    for bid, angle in sorted(theoretical_angles.items()):
        logger.debug(f"  B{bid}: {angle:.1f}°")
    
    best_match = None
    min_error = float('inf')
    candidate_mappings = []
    
    possible_pairs = [
        (0, 3),
        (1, 4),
        (2, 5)
    ]
    
    for detected_pair in symmetric_pairs:
        i, j = detected_pair
        
        for (bid1, bid2) in possible_pairs:
            for direction in [1, -1]:
                mapping = {}
                error = 0.0
                used_bids = set()
                
                if direction == 1:
                    mapping[detected_pair[0]] = bid1
                    mapping[detected_pair[1]] = bid2
                else:
                    mapping[detected_pair[0]] = bid2
                    mapping[detected_pair[1]] = bid1
                
                used_bids.add(bid1)
                used_bids.add(bid2)
                
                for k in range(len(points_info)):
                    if k in mapping or k in used:
                        continue
                        
                    min_angle_diff = float('inf')
                    best_bid = -1
                    
                    for bid in range(6):
                        if bid in used_bids:
                            continue
                            
                        angle_diff = abs(points_info[k]['angle_deg'] - theoretical_angles[bid])
                        angle_diff = min(angle_diff, 360 - angle_diff)
                        
                        if angle_diff < min_angle_diff:
                            min_angle_diff = angle_diff
                            best_bid = bid
                    
                    if best_bid != -1:
                        mapping[k] = best_bid
                        used_bids.add(best_bid)
                        error += min_angle_diff
                
                avg_error = error / len(mapping) if mapping else float('inf')
                
                candidate_mappings.append({
                    'mapping': mapping,
                    'error': avg_error,
                    'pair': (bid1, bid2),
                    'direction': direction
                })
                
                if avg_error < min_error:
                    min_error = avg_error
                    best_match = mapping
    
    if best_match is None or len(best_match) < 4:
        logger.debug(f"Insufficient matches found ({len(best_match) if best_match else 0}), using old method")
        return old_map_detections_to_beacons(centroids, light_num, img_shape)
    
    logger.debug(f"Best match error: {min_error:.2f}°, matched points: {len(best_match)}")
    
    uv_map = {}
    for idx, bid in best_match.items():
        uv_map[bid] = tuple(centers[idx])
    
    h, w = img_shape
    valid_map: Dict[int, Tuple[float, float]] = {}
    for bid in uv_map:
        x, y = uv_map[bid]
        if 0 <= x < w and 0 <= y < h:
            valid_map[bid] = (x, y)
    
    logger.debug(f"Geometric matching successful, mapped {len(valid_map)} beacons")
    return valid_map


def dcm2angle(R: np.ndarray) -> Tuple[float, float, float]:
    r11 = R[2, 0]
    r12 = R[2, 2]
    r21 = -R[2, 1]
    r31 = R[0, 1]
    r32 = R[1, 1]
    
    r1 = math.atan2(r11, r12)
    r2 = math.asin(r21)
    r3 = math.atan2(r31, r32)
    
    return r1, r2, r3


def Rvec2Euler(rvec: np.ndarray) -> Tuple[float, float, float]:
    rmat, _ = cv2.Rodrigues(rvec)
    
    r1, r2, r3 = dcm2angle(rmat)
    
    return (np.rad2deg(r1), np.rad2deg(r2), np.rad2deg(r3))


class HighFpsPoseEstimator:
    def __init__(self, config: Dict[str, Any]):
        self.config = self._validate_config(config)
        
        self.model = YOLO(self.config['model_path'])
        self.model.fuse()
        
        self.camera_matrix = np.array(self.config['camera_matrix'], dtype=np.float64)
        self.dist_coeffs = np.array(self.config['dist_coeffs'], dtype=np.float64)
        self.beacon_world = np.array(self.config['beacon_world_coords'], dtype=np.float64)
        
        self.use_geometric_matching = self.config.get('use_geometric_matching', True)
        self.geometric_angle_threshold = self.config.get('geometric_angle_threshold', 20.0)
        self.geometric_distance_threshold = self.config.get('geometric_distance_threshold', 0.2)
        
        self.beacon_tracker = BeaconTracker(
            max_missing=self.config.get('tracker_max_missing', 5),
            dt=self.config.get('kalman_dt', 0.1)
        )
        self.track_beacon_map: Dict[int, int] = {}
        
        self.frame_idx = 0
        self.total_frames = 0
        self.total_time = 0.0
        self.fps_list: List[float] = []
        self.last_fps = 0.0
        
        self.pose_output_path = self.config.get('pose_output_path', "pose_results.txt")
        self.output_file = self._open_output_file()
        
        self.video_processor = VideoProcessor(config)
        
        self.udp_communicator = UDPCommunicator(config['udp_config'])
        
        self.min_auth_count = self.config.get('min_auth_count', 3)
        self.min_init_count = self.config.get('min_init_count', 2)
        self.min_init_conf = self.config.get('min_init_conf', 0.3)
        self.symmetry_threshold = self.config.get('symmetry_threshold', 20)
        self.init_data: Dict[int, Dict[str, Any]] = {}
        self.authenticated_count = 0
        self.authenticated_map: Dict[int, int] = {}
        self.authentication_done = False
        self.auth_required_frames = self.config.get('auth_required_frames', 2)
        
        self.new_authenticated: Dict[int, int] = {}
        
        self.full_reauth_interval = self.config.get('full_reauth_interval', 300)
        self.last_full_reauth = 0
        self.previous_authenticated_map = {}
        
        self.reset_timeout = self.config.get('reset_timeout', 5.0)
        self.last_valid_detection_time = time.time()
        self.consecutive_valid_frames = 0
        
        self.output_container: List[Tuple[float, float, float, float, float, float]] = []
        self.max_cache_size = self.config.get('max_cache_size', 8)
        self.temp_pose_buffer = []
        self.temp_buffer_size = 5
        
        self.reprojection_threshold = self.config.get('reprojection_threshold', 8.0)
        self.initial_reprojection_threshold = self.config.get('initial_reprojection_threshold', 100.0)
        self.consecutive_invalid_frames = 0
        
        self.ransac_max_iterations = self.config.get('ransac_max_iterations', 100)
        self.ransac_threshold = self.config.get('ransac_threshold', 5.0)
        self.ransac_min_inliers = self.config.get('ransac_min_inliers', 3)
        
        self.last_pose = np.array([Constants.INVALID_VALUE] * 6)
        
        self.revalidation_interval = config.get('revalidation_interval', 20)
        self.geometric_tolerance = config.get('geometric_tolerance', 15.0)
        self.min_consistent_points = config.get('min_consistent_points', 4)
        self.last_revalidation = 0
        
        self.outlier_detection_threshold = config.get('outlier_detection_threshold', 1.5)
        self.outlier_consecutive_frames = config.get('outlier_consecutive_frames', 3)
        self.outlier_counter: Dict[int, int] = {i: 0 for i in range(Constants.MAX_BEACON_ID + 1)}

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        required_keys = ['model_path', 'camera_matrix', 'dist_coeffs', 'beacon_world_coords', 'udp_config']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Configuration missing required parameter: {key}")
                
        if np.array(config['camera_matrix']).shape != (3, 3):
            raise ValueError("camera_matrix must be a 3x3 matrix")
            
        if len(config['dist_coeffs']) not in [4, 5]:
            raise ValueError("dist_coeffs must contain 4 or 5 elements")
            
        if len(config['beacon_world_coords']) != 6:
            raise ValueError("beacon_world_coords must contain 6 beacon coordinates")
            
        return config
        
    def _open_output_file(self) -> Any:
        try:
            output_file = open(self.pose_output_path, 'w')
            output_file.write("frame_idx,x,y,z,yaw,pitch,roll,reprojection_error,process_time,avg_center_x,avg_center_y\n")
            return output_file
        except Exception as e:
            logger.error(f"Failed to open output file {self.pose_output_path}: {str(e)}")
            raise

    def update_container(self, x: float, y: float, z: float,
                        yaw: float, pitch: float, roll: float,
                        error: float) -> bool:
        current_threshold = self.reprojection_threshold if self.authentication_done else self.initial_reprojection_threshold
        is_valid = (error <= current_threshold and
                   x != Constants.INVALID_VALUE and y != Constants.INVALID_VALUE and z != Constants.INVALID_VALUE)
                   
        if is_valid:
            self.output_container.append((x, y, z, yaw, pitch, roll))
            self.consecutive_invalid_frames = 0
            self.last_valid_detection_time = time.time()
            self.consecutive_valid_frames += 1
            
            if not self.authentication_done:
                self.temp_pose_buffer.append((x, y, z, yaw, pitch, roll))
                if len(self.temp_pose_buffer) > self.temp_buffer_size:
                    self.temp_pose_buffer.pop(0)
        else:
            self.consecutive_invalid_frames += 1
            self.consecutive_valid_frames = 0
            
        if len(self.output_container) > self.max_cache_size:
            self.output_container.pop(0)
            
        return len(self.output_container) > 0

    def extract_output(self) -> Tuple[float, float, float, float, float, float]:
        if not self.authentication_done and len(self.temp_pose_buffer) > 0:
            return self.temp_pose_buffer[-1]
        
        for pose in reversed(self.output_container):
            x, y, z, yaw, pitch, roll = pose
            if x != Constants.INVALID_VALUE and y != Constants.INVALID_VALUE and z != Constants.INVALID_VALUE:
                return (x, y, z, yaw, pitch, roll)
        
        return (Constants.INVALID_VALUE, Constants.INVALID_VALUE, Constants.INVALID_VALUE,
                Constants.INVALID_VALUE, Constants.INVALID_VALUE, Constants.INVALID_VALUE)

    def _check_geometric_constraints(self, candidate_bid: int, candidate_coords: Tuple[float, float]) -> bool:
        if len(self.authenticated_map) < 2:
            return True
            
        auth_bids = list(self.authenticated_map.values())
        auth_image_coords = {}
        auth_world_coords = {}
        
        for track_id, bid in self.authenticated_map.items():
            if track_id in self.init_data and len(self.init_data[track_id]['coords']) > 0:
                avg_x, avg_y = np.mean(self.init_data[track_id]['coords'], axis=0)
                auth_image_coords[bid] = (avg_x, avg_y)
                auth_world_coords[bid] = self.beacon_world[bid]
        
        candidate_world = self.beacon_world[candidate_bid]
        
        valid_count = 0
        for auth_bid, (auth_x, auth_y) in auth_image_coords.items():
            world_dist = np.linalg.norm(auth_world_coords[auth_bid] - candidate_world)
            image_dist = math.hypot(auth_x - candidate_coords[0], auth_y - candidate_coords[1])
            ratio = image_dist / (world_dist + 1e-8)
            
            ratios = []
            for a, b in [(i, j) for i in auth_bids for j in auth_bids if i < j]:
                if a in auth_image_coords and b in auth_image_coords:
                    wd = np.linalg.norm(auth_world_coords[a] - auth_world_coords[b])
                    id_ = math.hypot(auth_image_coords[a][0] - auth_image_coords[b][0], 
                                    auth_image_coords[a][1] - auth_image_coords[b][1])
                    if wd > 0:
                        ratios.append(id_ / wd)
            
            if ratios:
                avg_ratio = np.mean(ratios)
                if 0.7 * avg_ratio <= ratio <= 1.3 * avg_ratio:
                    valid_count += 1
        
        return valid_count >= max(1, len(auth_bids) // 2)

    def authenticate_beacons(self, incremental: bool = False) -> int:
        candidates = []
        for track_id, data in self.init_data.items():
            if incremental and track_id in self.authenticated_map:
                continue
                
            count = data['count']
            avg_conf = data['total_conf'] / count if count > 0 else 0.0
            
            if count >= self.min_init_count and avg_conf >= self.min_init_conf:
                candidates.append({
                    'track_id': track_id,
                    'avg_conf': avg_conf,
                    'coords': np.array(data['coords'])
                })
                
        if incremental and len(candidates) < 1:
            return 0
            
        if not incremental and len(candidates) < self.min_auth_count:
            logger.info(f"Dynamic authentication: Insufficient candidate beacons ({len(candidates)}/{self.min_auth_count})")
            return 0
            
        all_coords = np.concatenate([c['coords'] for c in candidates], axis=0)
        center = np.mean(all_coords, axis=0)
        
        paired = set()
        valid_pairs = []
        unpaired = []
        
        for i, c1 in enumerate(candidates):
            if i in paired:
                continue
                
            avg_x1, avg_y1 = np.mean(c1['coords'], axis=0)
            
            sym_x = 2 * center[0] - avg_x1
            sym_y = 2 * center[1] - avg_y1
            
            matched = False
            for j, c2 in enumerate(candidates):
                if j == i or j in paired:
                    continue
                    
                avg_x2, avg_y2 = np.mean(c2['coords'], axis=0)
                dist = math.hypot(avg_x2 - sym_x, avg_y2 - sym_y)
                
                if dist < self.symmetry_threshold:
                    paired.add(i)
                    paired.add(j)
                    valid_pairs.append((c1, c2))
                    matched = True
                    break
                    
            if not matched:
                unpaired.append(c1)
                
        all_valid_candidates = []
        for p in valid_pairs:
            all_valid_candidates.extend(p)
        
        unpaired_sorted = sorted(unpaired, key=lambda x: x['avg_conf'], reverse=True)
        all_valid_candidates.extend(unpaired_sorted[:6 - len(all_valid_candidates)])
        
        avg_coords = []
        for c in all_valid_candidates[:6]:
            avg_x, avg_y = np.mean(c['coords'], axis=0)
            avg_coords.append((c['track_id'], avg_x, avg_y))
        
        centroids = np.array([(x, y) for (_, x, y) in avg_coords], dtype=np.float32)
        centroids_with_bg = np.zeros((len(centroids) + 1, 2), dtype=np.float32)
        centroids_with_bg[1:] = centroids
        
        img_shape = (self.video_processor.out_height, self.video_processor.out_width)
        
        if self.use_geometric_matching:
            geo_map = geometric_map_detections_to_beacons(
                centroids_with_bg, 
                len(centroids), 
                img_shape,
                self.beacon_world,
                angle_threshold=self.geometric_angle_threshold,
                distance_threshold=self.geometric_distance_threshold
            )
        else:
            geo_map = old_map_detections_to_beacons(centroids_with_bg, len(centroids), img_shape)
        
        bound_ids = set(self.authenticated_map.values()) if incremental else set()
        new_authenticated = 0
        
        for (track_id, x, y) in avg_coords:
            for bid, (bx, by) in geo_map.items():
                if bid not in bound_ids and math.hypot(x - bx, y - by) < 10:
                    if incremental:
                        if not self._check_geometric_constraints(bid, (x, y)):
                            continue
                    
                    self.authenticated_map[track_id] = bid
                    bound_ids.add(bid)
                    new_authenticated += 1
                    self.new_authenticated[bid] = self.frame_idx
                    break
                    
        self.authenticated_count = len(self.authenticated_map)
        logger.info(f"Beacon authentication completed! Authenticated {self.authenticated_count}/6 beacons, added {new_authenticated} new")
        
        if self.authenticated_count >= self.min_auth_count and not self.authentication_done:
            self.output_container.extend(self.temp_pose_buffer)
            self.temp_pose_buffer = []
            self.authentication_done = True
            
        return new_authenticated

    def reset_authentication(self) -> None:
        if self.authentication_done:
            logger.info(f"Long time without valid beacon detection, triggering reset (exceeded {self.reset_timeout} seconds)")
            self.authentication_done = False
            self.authenticated_count = 0
            self.authenticated_map = {}
            self.new_authenticated = {}
            
            current_frame = self.frame_idx
            self.init_data = {
                k: v for k, v in self.init_data.items()
                if current_frame - v['last_seen'] < 10
            }
            
            self.track_beacon_map = {}
            self.beacon_tracker.reset()
            self.last_pose = np.array([Constants.INVALID_VALUE] * 6)
            self.output_container = []
            self.temp_pose_buffer = []
            self.outlier_counter = {i: 0 for i in range(Constants.MAX_BEACON_ID + 1)}
            
            self.consecutive_valid_frames = self.auth_required_frames - 1

    def reset_full_authentication(self) -> None:
        logger.info(f"Performing full re-authentication (every {self.full_reauth_interval} frames)")
        
        self.authentication_done = False
        self.authenticated_count = 0
        self.authenticated_map = {}
        self.new_authenticated = {}
        
        current_frame = self.frame_idx
        self.init_data = {
            k: v for k, v in self.init_data.items()
            if current_frame - v['last_seen'] < 5
        }
        
        self.track_beacon_map = {}
        self.beacon_tracker.reset()
        self.last_pose = np.array([Constants.INVALID_VALUE] * 6)
        self.output_container = []
        self.temp_pose_buffer = []
        self.outlier_counter = {i: 0 for i in range(Constants.MAX_BEACON_ID + 1)}
        
        self.consecutive_valid_frames = self.auth_required_frames - 1
        self.last_full_reauth = self.frame_idx

    def evaluate_distribution_quality(self, auth_map: Dict[int, int]) -> float:
        if len(auth_map) < 3:
            return 0.0
            
        coords = []
        beacon_ids = []
        for track_id, bid in auth_map.items():
            if track_id in self.init_data and len(self.init_data[track_id]['coords']) > 0:
                avg_x, avg_y = np.mean(self.init_data[track_id]['coords'], axis=0)
                coords.append((avg_x, avg_y))
                beacon_ids.append(bid)
                
        if len(coords) < 3:
            return 0.0
            
        center = np.mean(coords, axis=0)
        
        symmetry_score = 0.0
        checked = set()
        
        for i, bid1 in enumerate(beacon_ids):
            if bid1 in checked:
                continue
                
            x1, y1 = coords[i]
            sym_x = 2 * center[0] - x1
            sym_y = 2 * center[1] - y1
            
            min_dist = float('inf')
            for j, bid2 in enumerate(beacon_ids):
                if i != j and bid2 not in checked:
                    x2, y2 = coords[j]
                    dist = math.hypot(x2 - sym_x, y2 - sym_y)
                    if dist < min_dist:
                        min_dist = dist
            
            if min_dist < self.symmetry_threshold:
                symmetry_score += 1
                checked.add(bid1)
        
        symmetry_score /= len(beacon_ids)
        
        distances = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
                distances.append(dist)
        
        uniformity_score = np.std(distances) / (np.mean(distances) + 1e-6) if distances else 0.0
            
        count_score = len(beacon_ids) / 6.0
        
        total_score = 0.6 * count_score + 0.3 * symmetry_score + 0.1 * (1 - uniformity_score)
        
        return total_score

    def compare_authentication_results(self, new_map: Dict[int, int]) -> Dict[int, int]:
        if not self.previous_authenticated_map:
            return new_map
            
        new_score = self.evaluate_distribution_quality(new_map)
        old_score = self.evaluate_distribution_quality(self.previous_authenticated_map)
        
        logger.info(f"Authentication result comparison - New score: {new_score:.3f}, Old score: {old_score:.3f}")
        
        return new_map if new_score > old_score else self.previous_authenticated_map

    def detect_outliers(self, current_points: Dict[int, Tuple[float, float]]) -> List[int]:
        if len(current_points) < 4:
            return []
            
        beacon_ids = list(current_points.keys())
        distances = []
        
        for i in range(len(beacon_ids)):
            for j in range(i+1, len(beacon_ids)):
                x1, y1 = current_points[beacon_ids[i]]
                x2, y2 = current_points[beacon_ids[j]]
                dist = math.hypot(x1 - x2, y1 - y2)
                distances.append(dist)
                
        if not distances:
            return []
            
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        world_points = [self.beacon_world[bid] for bid in beacon_ids]
        image_points = [current_points[bid] for bid in beacon_ids]
        
        img_pts_np = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
        world_pts_np = np.array(world_points, dtype=np.float32).reshape(-1, 1, 3)
        
        try:
            success, rvec, tvec = cv2.solvePnP(
                world_pts_np, img_pts_np, self.camera_matrix.T,
                self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return []
                
            reproj_pts, _ = cv2.projectPoints(
                world_pts_np, rvec, tvec,
                self.camera_matrix.T, self.dist_coeffs
            )
            
            errors = np.linalg.norm(img_pts_np - reproj_pts, axis=2).flatten()
        except:
            errors = [float('inf')] * len(beacon_ids)
            
        outliers = []
        for i, bid in enumerate(beacon_ids):
            x, y = current_points[bid]
            other_points = [current_points[other_bid] for other_bid in beacon_ids if other_bid != bid]
            avg_dist = np.mean([math.hypot(x - ox, y - oy) for ox, oy in other_points])
            
            reproj_error = errors[i] if i < len(errors) else float('inf')
            
            is_outlier = (avg_dist > mean_dist + self.outlier_detection_threshold * std_dist or 
                         reproj_error > self.geometric_tolerance)
                         
            if is_outlier:
                self.outlier_counter[bid] += 1
                if self.outlier_counter[bid] >= self.outlier_consecutive_frames:
                    outliers.append(bid)
                    logger.debug(f"Detected outlier beacon B{bid}, avg distance: {avg_dist:.2f}, reprojection error: {reproj_error:.2f}")
            else:
                self.outlier_counter[bid] = 0
                
        return outliers

    def ransac_filter(self, img_pts: List[Tuple[float, float]], 
                     world_pts: List[np.ndarray]) -> Tuple[List[Tuple[float, float]], List[np.ndarray]]:
        num_points = len(img_pts)
        if num_points <= 3:
            return img_pts, world_pts
            
        best_inliers = []
        best_inlier_count = 0
        
        img_pts_np = np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2)
        world_pts_np = np.array(world_pts, dtype=np.float32).reshape(-1, 1, 3)
        
        for _ in range(self.ransac_max_iterations):
            indices = np.random.choice(num_points, 3, replace=False)
            sample_img = img_pts_np[indices]
            sample_world = world_pts_np[indices]
            
            try:
                success, rvec, tvec = cv2.solvePnP(
                    sample_world, sample_img, self.camera_matrix.T,
                    self.dist_coeffs, flags=cv2.SOLVEPNP_P3P
                )
                
                if not success:
                    continue
                    
                reproj_pts, _ = cv2.projectPoints(
                    world_pts_np, rvec, tvec, 
                    self.camera_matrix.T, self.dist_coeffs
                )
                
                errors = np.linalg.norm(img_pts_np - reproj_pts, axis=2).flatten()
                
                inliers = np.where(errors < self.ransac_threshold)[0]
                
                if len(inliers) > best_inlier_count:
                    best_inlier_count = len(inliers)
                    best_inliers = inliers
                    
            except Exception:
                continue
        
        if best_inlier_count >= self.ransac_min_inliers:
            filtered_img_pts = [img_pts[i] for i in best_inliers]
            filtered_world_pts = [world_pts[i] for i in best_inliers]
            return filtered_img_pts, filtered_world_pts
        else:
            return img_pts, world_pts

    def solve_pnp_fast(self, img_pts: List[Tuple[float, float]],
                      world_pts: List[np.ndarray]) -> Optional[Dict[str, Any]]:
        num_points = len(img_pts)
        if num_points < 3:
            return None
            
        try:
            img_pts_np = np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2)
            world_pts_np = np.array(world_pts, dtype=np.float32).reshape(-1, 1, 3)
            
            camera_matrix = self.camera_matrix.T
            
            if num_points == 3:
                retval, rvecs, tvecs = cv2.solveP3P(
                    world_pts_np, img_pts_np, camera_matrix,
                    self.dist_coeffs, flags=cv2.SOLVEPNP_P3P
                )
                
                if not retval:
                    return None
                    
                min_error = float('inf')
                best_rvec = None
                best_tvec = None
                
                for i in range(rvecs.shape[0]):
                    rvec = rvecs[i].reshape(3, 1)
                    tvec = tvecs[i].reshape(3, 1)
                    
                    yaw, pitch, roll = Rvec2Euler(rvec)
                    current_pose = np.array([
                        tvec[0][0], tvec[1][0], tvec[2][0],
                        yaw, pitch, roll
                    ])
                    
                    if np.any(self.last_pose != Constants.INVALID_VALUE):
                        error_array = current_pose - self.last_pose
                        error = np.var(error_array)
                        if error < min_error:
                            min_error = error
                            best_rvec = rvec
                            best_tvec = tvec
                            
                if best_rvec is None:
                    best_rvec = rvecs[0].reshape(3, 1)
                    best_tvec = tvecs[0].reshape(3, 1)
                    
                rvec, tvec = best_rvec, best_tvec
                
            elif num_points >= 4:
                flags = cv2.SOLVEPNP_ITERATIVE
                success, rvec, tvec = cv2.solvePnP(
                    world_pts_np, img_pts_np, camera_matrix,
                    self.dist_coeffs, flags=flags
                )
                
                if not success:
                    return None
                    
            else:
                return None
                
            reproj_pts, _ = cv2.projectPoints(world_pts_np, rvec, tvec,
                                              camera_matrix, self.dist_coeffs)
            error = cv2.norm(img_pts_np, reproj_pts, cv2.NORM_L2) / len(img_pts)
            
            return {
                'rvec': rvec,
                'tvec': tvec,
                'error': error
            }
            
        except Exception as e:
            logger.error(f"PNP solving error: {e}")
            return None

    def _process_detections(self, frame: np.ndarray) -> List[Tuple[int, int, int, float]]:
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.2,
            iou=0.2,
            verbose=False,
            imgsz=640
        )
        
        detected = []
        if results[0].keypoints is not None and len(results[0].keypoints) > 0:
            keypoints = results[0].keypoints.xy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy().tolist()
            
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.cpu().numpy().astype(int).tolist()
                
                for i in range(len(keypoints)):
                    if len(keypoints[i]) > Constants.BEACON_KEYPOINT_INDEX:
                        x, y = keypoints[i][Constants.BEACON_KEYPOINT_INDEX]
                        x_int, y_int = int(round(x)), int(round(y))
                        detected.append((track_ids[i], x_int, y_int, confidences[i]))
                        
        return detected

    def _process_authentication(self, detected: List[Tuple[int, int, int, float]]) -> None:
        new_track_ids = set()
        for (track_id, x, y, conf) in detected:
            new_track_ids.add(track_id)
            
            if track_id not in self.init_data:
                self.init_data[track_id] = {
                    'count': 0,
                    'total_conf': 0.0,
                    'coords': [],
                    'first_seen': self.frame_idx,
                    'last_seen': self.frame_idx
                }
                
            self.init_data[track_id]['count'] += 1
            self.init_data[track_id]['total_conf'] += conf
            self.init_data[track_id]['coords'].append((x, y))
            self.init_data[track_id]['last_seen'] = self.frame_idx
        
        current_frame = self.frame_idx
        self.init_data = {
            k: v for k, v in self.init_data.items()
            if current_frame - v['last_seen'] < 5
        }
        
        has_new_tracks = False
        for track_id in new_track_ids:
            if track_id not in self.authenticated_map and track_id in self.init_data:
                data = self.init_data[track_id]
                avg_conf = data['total_conf'] / data['count'] if data['count'] > 0 else 0.0
                if data['count'] >= self.min_init_count and avg_conf >= self.min_init_conf:
                    has_new_tracks = True
                    break
        
        required_frames = self.auth_required_frames
        trigger_conditions = [
            (not self.authentication_done and self.consecutive_valid_frames >= required_frames),
            (self.authentication_done and has_new_tracks),
            (self.authentication_done and self.authenticated_count < 6)
        ]
        
        if any(trigger_conditions):
            logger.info("Triggering beacon authentication...")
            incremental = self.authentication_done
            auth_count = self.authenticate_beacons(incremental=incremental)
            
            self.authenticated_map = self.compare_authentication_results(self.authenticated_map)
            self.authenticated_count = len(self.authenticated_map)
            
            if not self.authentication_done and self.authenticated_count >= self.min_auth_count:
                self.authentication_done = True
                logger.info(f"Authentication successful, authenticated {self.authenticated_count}/{self.min_auth_count} beacons")
            elif not self.authentication_done:
                logger.info(f"Authentication continuing, need at least {self.min_auth_count} valid beacons (current {self.authenticated_count})")
            else:
                if auth_count > 0:
                    logger.info(f"Incremental authentication successful, added {auth_count} new beacons")

    def _match_and_track_beacons(self, detected: List[Tuple[int, int, int, float]],
                                h: int, w: int) -> Tuple[List[Tuple[int, int, Tuple[int, int]]], Dict[int, Tuple[int, int]]]:
        light_num = len(detected)
        current_matched = []
        active_beacons = set()
        
        if light_num > 0:
            centroids = np.array([(x, y) for (_, x, y, _) in detected], dtype=np.float32)
            centroids_with_bg = np.zeros((light_num + 1, 2), dtype=np.float32)
            centroids_with_bg[1:] = centroids
            
            geo_map = map_detections_to_beacons(centroids_with_bg, light_num, (h, w))
            
            authenticated_track_ids = set(self.authenticated_map.keys())
            for (track_id, x, y, conf) in detected:
                if track_id in authenticated_track_ids:
                    beacon_id = self.authenticated_map[track_id]
                    current_matched.append((track_id, beacon_id, (x, y)))
                    active_beacons.add(beacon_id)
                    self.beacon_tracker.update(beacon_id, x, y, detection_score=conf)
                else:
                    if track_id in self.track_beacon_map:
                        beacon_id = self.track_beacon_map[track_id]
                        current_matched.append((track_id, beacon_id, (x, y)))
                        active_beacons.add(beacon_id)
                        self.beacon_tracker.update(beacon_id, x, y)
                    else:
                        min_dist = float('inf')
                        best_beacon = None
                        for beacon_id, (bx, by) in geo_map.items():
                            if beacon_id not in self.authenticated_map.values():
                                dist = abs(x - bx) + abs(y - by)
                                if dist < min_dist:
                                    min_dist = dist
                                    best_beacon = beacon_id
                        if best_beacon is not None:
                            self.track_beacon_map[track_id] = best_beacon
                            current_matched.append((track_id, best_beacon, (x, y)))
                            active_beacons.add(best_beacon)
                            self.beacon_tracker.update(best_beacon, x, y, detection_score=conf)
        
        for beacon_id in range(Constants.MAX_BEACON_ID + 1):
            if beacon_id not in active_beacons:
                self.beacon_tracker.mark_missing(beacon_id)
                
        predicted_beacons = {}
        for _, beacon_id in self.authenticated_map.items():
            if beacon_id not in active_beacons:
                pred_pos = self.beacon_tracker.predict(beacon_id)
                if pred_pos is not None:
                    predicted_beacons[beacon_id] = pred_pos
                    
        for beacon_id in range(Constants.MAX_BEACON_ID + 1):
            if beacon_id not in active_beacons and beacon_id not in predicted_beacons:
                pred_pos = self.beacon_tracker.predict(beacon_id)
                if pred_pos is not None:
                    predicted_beacons[beacon_id] = pred_pos
                        
        return current_matched, predicted_beacons

    def _draw_beacon_info(self, frame: np.ndarray, current_matched: List[Tuple[int, int, Tuple[int, int]]], 
                         predicted_beacons: Dict[int, Tuple[int, int]], outliers: List[int] = []) -> None:
        detected_beacon_ids = set()
        for (_, beacon_id, (x, y)) in current_matched:
            detected_beacon_ids.add(beacon_id)
            is_authenticated = any(bid == beacon_id for _, bid in self.authenticated_map.items())
            is_new = False
            if is_authenticated and beacon_id in self.new_authenticated:
                frames_since_added = self.frame_idx - self.new_authenticated[beacon_id]
                is_new = frames_since_added < Constants.NEW_AUTHENTICATED_DURATION
                
            is_outlier = beacon_id in outliers
            
            color = Constants.OUTLIER_COLOR if is_outlier else \
                    Constants.NEW_AUTHENTICATED_COLOR if is_new else \
                    Constants.AUTHENTICATED_COLOR if is_authenticated else \
                    Constants.UNAUTHENTICATED_COLOR
            
            cv2.circle(frame, (x, y), 8, color, -1)
            cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)
            
            velocity_text = ""
            if is_authenticated:
                vx, vy = self.beacon_tracker.get_velocity(beacon_id)
                speed = math.hypot(vx, vy)
                if speed > 1.0:
                    velocity_text = f" v:{speed:.1f}"
            
            text = f"B{beacon_id}: ({x}, {y}){velocity_text}{' [New]' if is_new else ''}{' [Outlier]' if is_outlier else ''}"
            cv2.putText(frame, text, (x + 15, y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        for beacon_id, (x, y) in predicted_beacons.items():
            if beacon_id not in detected_beacon_ids:
                overlay = frame.copy()
                cv2.circle(overlay, (x, y), 8, Constants.PREDICTED_COLOR, -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                cv2.circle(frame, (x, y), 10, (255, 255, 255), 1)
                text = f"B{beacon_id} (Predicted)"
                cv2.putText(frame, text, (x + 15, y - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, Constants.PREDICTED_COLOR, 1)

    def _draw_overlay(self, frame: np.ndarray, x: float, y: float, z: float,
                     yaw: float, pitch: float, roll: float,
                     avg_center_x: float = Constants.INVALID_VALUE, avg_center_y: float = Constants.INVALID_VALUE,
                     current_matched: List[Tuple[int, int, Tuple[int, int]]] = [],
                     predicted_beacons: Dict[int, Tuple[int, int]] = {},
                     outliers: List[int] = [],
                     frame_idx: int = 0) -> None:
        self._draw_beacon_info(frame, current_matched, predicted_beacons, outliers)
        
        cv2.putText(
            frame,
            f"FPS: {self.last_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        
        auth_status = f"Auth: {self.authenticated_count}/6" if self.authentication_done else "Authenticating..."
        cv2.putText(
            frame,
            auth_status,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if self.authenticated_count >= self.min_auth_count else (0, 0, 255),
            2
        )
        
        reauth_countdown = self.full_reauth_interval - (self.frame_idx - self.last_full_reauth)
        cv2.putText(
            frame,
            #f"Next Full Auth: {reauth_countdown} frames",
            f"",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 165, 0),
            2
        )
        
        elapsed = time.time() - self.last_valid_detection_time
        #timeout_status = f"Timeout: {elapsed:.1f}/{self.reset_timeout}s"
        timeout_status = f""
        cv2.putText(
            frame,
            timeout_status,
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255) if elapsed > self.reset_timeout * 0.7 else (0, 255, 0),
            2
        )
        
        if outliers:
            cv2.putText(
                frame,
                f"Outliers: {len(outliers)}",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        
        if self.udp_communicator.udp_socket:
            cv2.putText(
                frame,
                f"UDP: {self.config['udp_config']['ip']}:{self.config['udp_config']['port']}",
                (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )
        
        if avg_center_x != Constants.INVALID_VALUE and avg_center_y != Constants.INVALID_VALUE:
            pose_text = f"Avg Center: X:{avg_center_x:.1f} Y:{avg_center_y:.1f}"
        else:
            pose_text = f"X:{x:.1f} Y:{y:.1f} Z:{z:.1f}"
            
        cv2.putText(
            frame,
            pose_text,
            (10, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255) if avg_center_x == Constants.INVALID_VALUE else (0, 165, 255),
            2
        )
        
        if x != Constants.INVALID_VALUE and y != Constants.INVALID_VALUE and z != Constants.INVALID_VALUE:
            angle_text = f"Yaw:{yaw:.1f} Pitch:{pitch:.1f} Roll:{roll:.1f}"
            cv2.putText(
                frame,
                angle_text,
                (10, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            
        if avg_center_x != Constants.INVALID_VALUE and avg_center_y != Constants.INVALID_VALUE:
            cv2.circle(frame, (int(avg_center_x), int(avg_center_y)), 10, (0, 165, 255), -1)
            cv2.putText(
                frame,
                "Avg",
                (int(avg_center_x) + 15, int(avg_center_y) - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2
            )
    
    def periodic_revalidation(self, current_matched: List[Tuple[int, int, Tuple[int, int]]], 
                             predicted_beacons: Dict[int, Tuple[int, int]]) -> Tuple[bool, List[int]]:
        current_points = {}
        for (_, beacon_id, (x, y)) in current_matched:
            current_points[beacon_id] = (x, y)
        for beacon_id, (x, y) in predicted_beacons.items():
            if beacon_id not in current_points:
                current_points[beacon_id] = (x, y)
        
        outliers = self.detect_outliers(current_points)
        
        if len(current_points) < 3:
            return True, outliers
            
        if len(outliers) / len(current_points) > 1/3:
            logger.warning(f"Beacon matching consistency check failed! Triggering rematching. Outlier count: {len(outliers)}")
            
            reliable_matches = {}
            for track_id, bid in self.authenticated_map.items():
                if bid not in outliers:
                    reliable_matches[track_id] = bid
            
            self.authenticated_map = reliable_matches
            self.track_beacon_map = {k: v for k, v in self.track_beacon_map.items() if v not in outliers}
            self.authenticated_count = len(self.authenticated_map)
            
            self.authenticate_beacons(incremental=True)
            
            self.output_container = []
            self.temp_pose_buffer = []
            self.last_pose = np.array([Constants.INVALID_VALUE] * 6)
            
            return True, outliers
        return False, outliers

    def process_frame(self) -> bool:
        start_time = time.time()
        
        ret, frame = self.video_processor.read_frame()
        if not ret or frame is None:
            return False
            
        if self.frame_idx - self.last_full_reauth >= self.full_reauth_interval:
            self.reset_full_authentication()
            
        elapsed = time.time() - self.last_valid_detection_time
        if elapsed > self.reset_timeout and self.authentication_done:
            self.reset_authentication()
            
        h, w = frame.shape[:2]
        if self.config.get('downscale', True):
            scale = 0.75
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            h, w = frame.shape[:2]
            
        detected = self._process_detections(frame)
        
        self._process_authentication(detected)
        
        current_matched, predicted_beacons = self._match_and_track_beacons(detected, h, w)
        
        outliers = []
        
        if self.authentication_done and self.frame_idx - self.last_revalidation >= self.revalidation_interval:
            revalidation_triggered, outliers = self.periodic_revalidation(current_matched, predicted_beacons)
            self.last_revalidation = self.frame_idx
        
        all_image_pts = []
        all_world_pts = []
        
        for (_, beacon_id, (x, y)) in current_matched:
            if beacon_id not in outliers:
                all_image_pts.append((x, y))
                all_world_pts.append(self.beacon_world[beacon_id])
                
        for beacon_id, (x, y) in predicted_beacons.items():
            if beacon_id not in outliers:
                all_image_pts.append((x, y))
                all_world_pts.append(self.beacon_world[beacon_id])
        
        avg_center_x, avg_center_y = Constants.INVALID_VALUE, Constants.INVALID_VALUE
        if 0 < len(all_image_pts) < 3:
            xs = [pt[0] for pt in all_image_pts]
            ys = [pt[1] for pt in all_image_pts]
            avg_center_x = sum(xs) / len(xs)
            avg_center_y = sum(ys) / len(ys)
            
        pnp_result = None
        if len(all_image_pts) >= 3:
            filtered_img_pts, filtered_world_pts = self.ransac_filter(all_image_pts, all_world_pts)
            
            if len(filtered_img_pts) >= 3:
                pnp_result = self.solve_pnp_fast(filtered_img_pts, filtered_world_pts)
            else:
                logger.warning(f"Insufficient points after RANSAC filtering, original: {len(all_image_pts)}, filtered: {len(filtered_img_pts)}")
                pnp_result = self.solve_pnp_fast(all_image_pts, all_world_pts)
        
        frame_time = time.time() - start_time
        x, y, z = Constants.INVALID_VALUE, Constants.INVALID_VALUE, Constants.INVALID_VALUE
        yaw, pitch, roll = Constants.INVALID_VALUE, Constants.INVALID_VALUE, Constants.INVALID_VALUE
        reprojection_error = float('inf')
        
        if pnp_result is not None:
            tvec = pnp_result['tvec'].flatten()
            rvec = pnp_result['rvec'].flatten()
            reprojection_error = pnp_result['error']
            
            current_threshold = self.reprojection_threshold if self.authentication_done else self.initial_reprojection_threshold
            
            if reprojection_error <= current_threshold:
                yaw, pitch, roll = Rvec2Euler(rvec)
                
                x, y, z = tvec[0], tvec[1], tvec[2]
                
                self.last_pose = np.array([x, y, z, yaw, pitch, roll])
            else:
                logger.warning(f"Reprojection error too high: {reprojection_error:.2f} > {current_threshold}")
        elif 0 < len(all_image_pts) < 3:
            x, y = avg_center_x, avg_center_y
            z = Constants.INVALID_VALUE
            yaw, pitch, roll = Constants.INVALID_VALUE, Constants.INVALID_VALUE, Constants.INVALID_VALUE
        
        self.update_container(x, y, z, yaw, pitch, roll, reprojection_error)
        x, y, z, yaw, pitch, roll = self.extract_output()
        
        processed_yaw = (yaw - 180.0 if yaw != Constants.INVALID_VALUE and yaw >= 0 else 
                        yaw + 180.0 if yaw != Constants.INVALID_VALUE else Constants.INVALID_VALUE)
        
        self.output_file.write(
            f"{self.frame_idx},{x:.2f},{y:.2f},{z:.2f},"
            f"{processed_yaw:.2f},{pitch:.2f},{roll:.2f},"
            f"{reprojection_error:.2f},{frame_time:.4f},"
            f"{avg_center_x:.2f},{avg_center_y:.2f}\n"
        )
        
        if self.frame_idx % self.config['udp_config']['send_interval'] == 0:
            self.udp_communicator.send_data(x, y, z, processed_yaw, pitch, roll, avg_center_x, avg_center_y)
            
        self.total_time += frame_time
        self.total_frames += 1
        
        if self.total_frames % 10 == 0 and frame_time > 0:
            recent_time = 10 * frame_time
            self.last_fps = 10 / recent_time if recent_time > 0 else 0
            self.fps_list.append(self.last_fps)
            
        if not self.video_processor.process_and_display(
            frame, 
            self._draw_overlay,
            x, y, z, 
            processed_yaw, pitch, roll, 
            avg_center_x, avg_center_y,
            current_matched,
            predicted_beacons,
            outliers,
            frame_idx=self.frame_idx
        ):
            return False
                
        self.frame_idx += 1
        return True

    def get_supported_camera_resolutions(self) -> List[Tuple[int, int]]:
        if hasattr(self.video_processor, 'get_supported_resolutions'):
            return self.video_processor.get_supported_resolutions()
        return []
    
    def run(self, video_source: Union[str, int] = 0) -> None:
        try:
            if not self.video_processor.initialize(video_source):
                logger.error("Video initialization failed")
                return
                
            logger.info("Starting high FPS pose estimation...")
            logger.info(f"Dynamic beacon authentication: Automatic when sufficient beacons detected (requires at least {self.min_auth_count} beacons)")
            logger.info(f"Incremental authentication: New beacons verified and added immediately when detected")
            logger.info(f"Periodic full re-authentication: Complete reset and re-authentication every {self.full_reauth_interval} frames")
            logger.info(f"Periodic revalidation: Consistency check every {self.revalidation_interval} frames")
            logger.info(f"RANSAC configuration: Max iterations={self.ransac_max_iterations}, threshold={self.ransac_threshold} pixels")
            logger.info(f"Kalman filter configuration: Time interval={self.beacon_tracker.dt} seconds")
            logger.info("Press 'q' to exit")
            
            while True:
                if not self.process_frame():
                    break
                    
        except Exception as e:
            logger.error(f"Processing error: {str(e)}", exc_info=True)
            
        finally:
            self.udp_communicator.close()
            self.video_processor.release()
            
            if self.output_file:
                self.output_file.close()
                
            if self.total_frames > 0:
                avg_fps = self.total_frames / self.total_time
                logger.info("\n===== Performance Statistics =====")
                logger.info(f"Total frames: {self.total_frames}")
                logger.info(f"Total time: {self.total_time:.2f} seconds")
                logger.info(f"Average FPS: {avg_fps:.2f}")
                logger.info(f"Beacon authentication count: {self.authenticated_count}/6")
                logger.info(f"Pose results saved to: {self.pose_output_path}")
                logger.info(f"Processed video saved to: {self.video_processor.output_video_path}")


if __name__ == "__main__":
    config = {
        'model_path': "best.pt",
        'camera_matrix': [
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0]
        ],
        'dist_coeffs': [1, 1, 0, 0, 1],
        #your beacon world coordinates in mm
        'beacon_world_coords': [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([1.5, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            np.array([2.5, 0.0, 0.0])
        ],
        'display': False,
        'downscale': False,
        'output_video_path': 'processed_video_with_fps.avi',
        'pose_output_path': "pose_results.txt",
        'split_interval_minutes': 20,
        'video_output_dir': 'output_videos',
        'save_frames': False,
        'frames_save_path': 'processed_frames',
        'reprojection_threshold': 5.0,
        'initial_reprojection_threshold': 50.0,
        'tracker_max_missing': 5,
        'kalman_dt': 0.1,
        'max_cache_size': 10,
        'min_init_count': 3,
        'min_init_conf': 0.10,
        'symmetry_threshold': 30,
        'min_auth_count': 5,
        'auth_required_frames': 4,
        'full_reauth_interval': 150,
        'reset_timeout': 1,
        'ransac_max_iterations': 100,
        'ransac_threshold': 5.0,
        'ransac_min_inliers': 3,
        'revalidation_interval': 20,
        'geometric_tolerance': 15.0,
        'min_consistent_points': 3,
        'outlier_detection_threshold': 1.8,
        'outlier_consecutive_frames': 2,
        'udp_config': {
            'ip': '192.168.1.1',
            'port': 6678,
            'send_interval': 1
        }
    }
    
    estimator = HighFpsPoseEstimator(config)
    # estimator.run(video_source=r"your outline underwater video.mp4")
    estimator.run(video_source=0)