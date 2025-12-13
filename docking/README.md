# Beacon Tracking System Documentation

This directory contains documentation and optimized code for the Real-time Incremental Beacon Authentication System.

## Files

1. **technical_documentation.md** - Comprehensive technical documentation in English explaining:
   - System architecture and components
   - Core algorithms and methodologies
   - Configuration options
   - Performance considerations
   - Usage instructions

2. **optimized_beacon_tracker.py** - Cleaned and optimized implementation of the beacon tracking system with:
   - Removed redundant comments
   - Improved code readability
   - Maintained full functionality
   - English documentation strings

## System Overview

The Real-time Incremental Beacon Authentication System is a high-performance computer vision application designed for:

- Real-time detection and tracking of beacons using Kalman filtering
- Dynamic authentication of beacon identities
- 3D pose estimation using camera calibration data
- Robust outlier detection and filtering
- UDP data transmission for integration with other systems

## Key Features

- **Dynamic Authentication**: Automatically authenticates beacons when sufficient candidates are detected
- **Incremental Authentication**: Continuously adds new beacons that meet geometric constraints
- **Robust Tracking**: Uses Kalman filtering to maintain tracking even with temporary occlusions
- **Outlier Detection**: Implements RANSAC and distance-based methods to filter outlier points
- **Performance Optimized**: Configurable downscaling and selective processing for high frame rates
- **Visualization**: Real-time display of beacon states, tracking information, and pose data

## Getting Started

### Prerequisites

- Python 3.6+
- OpenCV (cv2)
- NumPy
- Ultralytics YOLO
- Socket (standard library)
- Math (standard library)
- Logging (standard library)

### Configuration

The system is configured through a dictionary-based configuration. Key parameters include:

- Camera intrinsic matrix and distortion coefficients
- Beacon world coordinates
- Kalman filter parameters
- Authentication thresholds
- UDP communication settings

### Running the System

```python
from optimized_beacon_tracker import HighFpsPoseEstimator

# Create configuration dictionary
config = {
    'model_path': "best.pt",
    'camera_matrix': [...],
    'dist_coeffs': [...],
    'beacon_world_coords': [...],
    # Additional configuration...
}

# Initialize the estimator
estimator = HighFpsPoseEstimator(config)

# Run with video file
estimator.run(video_source="path/to/video.mp4")

# Or run with camera
estimator.run(video_source=0)
```

## Performance Considerations

- Enable frame downscaling for better performance on lower-end hardware
- Adjust the RANSAC parameters based on your specific environment
- Configure appropriate authentication thresholds for your use case
- Monitor the FPS output to ensure real-time performance

## Troubleshooting

Common issues and solutions:

- **Poor detection**: Adjust YOLO confidence threshold or improve lighting conditions
- **Authentication failures**: Ensure sufficient beacons are visible and well-distributed
- **High reprojection errors**: Verify camera calibration and beacon world coordinates
- **Performance issues**: Enable downscaling or reduce processing frequency

