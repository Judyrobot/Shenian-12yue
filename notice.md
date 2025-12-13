Real-time Incremental Beacon Authentication System
Overview
This system implements a high-performance beacon tracking and pose estimation system designed for real-time applications. It uses advanced computer vision techniques including object detection, Kalman filtering, and geometric constraints to accurately track beacons and estimate their positions in 3D space.

Key Features
Real-time beacon detection and tracking using Kalman filtering
Dynamic beacon authentication supporting both initial and incremental authentication
Geometric constraint validation for new beacons based on previously authenticated ones
Continuous beacon set optimization to ensure geometric and temporal consistency
Beacon state visualization (unauthenticated/authenticated/newly authenticated/predicted)
Pose estimation and UDP data transmission
RANSAC algorithm for outlier filtering
Enhanced outlier detection and rejection mechanisms
Dynamic Q & R calculation based on ByteTrack paper (Zhang et al., 2022)
System Architecture
The system consists of several key components working together:

Beacon Detection: Uses YOLO model for object detection and keypoint extraction
Beacon Tracking: Implements Kalman filtering for smooth tracking
Beacon Authentication: Validates beacon identities using geometric constraints
Pose Estimation: Calculates 3D position and orientation using PnP algorithm
Data Communication: Transmits pose data via UDP
Visualization: Provides real-time visualization of tracking results
Core Components
Kalman Filter
The Kalman filter is used to predict and update beacon positions over time, providing smooth tracking even when beacons are temporarily occluded.

class KalmanFilter:
    def __init__(self, dt: float = 0.1, sigma_a: float = 0.05):
        # Initializes Kalman filter with time step and acceleration noise parameters
        # ...
Key features:

Dynamic process noise (Q) calculation based on time step and acceleration
Adaptive measurement noise (R) based on detection confidence scores
State prediction and update functions
Beacon Tracker
Manages multiple Kalman filters for tracking all beacons simultaneously.

class BeaconTracker:
    def __init__(self, max_missing: int = 3, dt: float = 0.1):
        # Initializes trackers for all beacons (0-5)
        # ...
Key features:

Individual tracking for each beacon ID (0-5)
Missing beacon handling with configurable timeout
Velocity estimation for each beacon
UDP Communicator
Handles data transmission of pose estimation results.

class UDPCommunicator:
    def __init__(self, config: Dict[str, Any]):
        # Initializes UDP communication with specified configuration
        # ...
Key features:

Configurable target IP and port
Structured data packing for efficient transmission
Error handling for network issues
Video Processor
Manages video input/output operations.

class VideoProcessor:
    def __init__(self, config: Dict[str, Any]):
        # Initializes video processing with display and recording options
        # ...
Key features:

Video source handling (camera or file)
Configurable resolution and frame rate
Video recording with automatic splitting
Frame saving capabilities
High FPS Pose Estimator
The main class that coordinates all components and implements the core algorithm.

class HighFpsPoseEstimator:
    def __init__(self, config: Dict[str, Any]):
        # Initializes the complete pose estimation system
        # ...
Key features:

Dynamic beacon authentication
Incremental beacon addition
Periodic revalidation of beacon matches
RANSAC-based outlier filtering
PnP-based pose estimation
Performance monitoring
Algorithms
Beacon Authentication
The system implements a sophisticated beacon authentication algorithm that:

Collects candidate beacons based on detection count and confidence
Validates geometric constraints between candidate beacons
Performs symmetry checks to verify beacon pairs
Supports both initial authentication (3-6 beacons) and incremental authentication
Geometric Mapping
Two mapping algorithms are available:

Traditional Mapping: Based on relative positions (leftmost, rightmost, top, bottom)
Geometric Mapping: Based on circular geometry and theoretical angles from world coordinates
Outlier Detection
Multiple mechanisms are used to detect and filter outliers:

Distance-based outlier detection: Identifies points with abnormal distances to other beacons
RANSAC algorithm: Iteratively estimates model parameters and identifies inliers
Reprojection error analysis: Filters points with high reprojection errors
Pose Estimation
The system uses the Perspective-n-Point (PnP) algorithm to estimate camera pose:

Selects appropriate algorithm based on number of points (P3P for 3 points, iterative for 4+ points)
Uses camera intrinsic parameters for accurate projection
Applies RANSAC for robust estimation
Calculates Euler angles from rotation vector
Configuration
The system is highly configurable through a dictionary-based configuration:

config = {
    'model_path': "best.pt",
    'camera_matrix': [...],
    'dist_coeffs': [...],
    'beacon_world_coords': [...],
    # Additional configuration parameters
}
Key configuration categories:

Camera parameters: Intrinsic matrix and distortion coefficients
Beacon configuration: World coordinates of beacons
Tracking parameters: Kalman filter settings
Authentication parameters: Thresholds and validation criteria
Visualization options: Display and recording settings
UDP communication: Target IP, port, and transmission interval
Performance Considerations
The system is optimized for high-performance operation:

Frame downscaling: Optional frame resizing for faster processing
Selective processing: Different algorithms based on number of points
Caching: Maintains results buffer for consistent output
Adaptive thresholds: Uses different criteria before/after authentication
Usage
# Initialize the estimator
estimator = HighFpsPoseEstimator(config)

# Run with video file
estimator.run(video_source="path/to/video.mp4")

# Or run with camera
estimator.run(video_source=0)
Output
The system provides multiple output formats:

UDP packets: Real-time pose data transmission
Text file: Detailed log of pose estimates and processing metrics
Video recording: Visualization with overlay information
Optional frame saving: Individual processed frames
Troubleshooting
Common issues and solutions:

Poor beacon detection: Adjust detection confidence threshold or improve lighting conditions
Authentication failures: Ensure sufficient beacons are visible and well-distributed
High reprojection errors: Check camera calibration and beacon world coordinates
Performance issues: Enable frame downscaling or reduce processing frequency
References
Zhang, Y., Sun, P., Jiang, Y., Yu, D., Yuan, Z., Luo, Z., ... & Wang, X. (2022). ByteTrack: Multi-object tracking by associating every detection box. In Proceedings of the European Conference on Computer Vision (ECCV).