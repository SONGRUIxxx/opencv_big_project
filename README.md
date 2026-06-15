# Fisheye Motion Tracking

A simple OpenCV/C++ project for fisheye motion extraction, tracking, and evaluation.

## Project structure

- `homework2/` - dataset folders including RGB images, previous frames, calibration data, and annotations.
- `homework2_cv/` - C++ implementation and build system.
  - `CMakeLists.txt` - project configuration.
  - `src/` - source code for fisheye undistortion, motion extraction, tracking, evaluation, and visualization.
  - `build/` - recommended build directory.
  - `output/` - default output directory for results and charts.

## Build

1. Open a terminal in `homework2_cv/`.
2. Create a build directory and run CMake:

```powershell
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

> If using Conda with OpenCV installed, the project can auto-detect `OpenCV_DIR` from `CONDA_PREFIX`.

## Run

From `homework2_cv/build` or the project folder:

```powershell
.\fisheye_motion_tracking.exe --data-root "..\homework2" --output-root ".\output" --max-samples 50
```

### Common options

- `--data-root PATH` : dataset root (default `../homework2`)
- `--output-root PATH` : output directory (default `./output`)
- `--max-samples N` : limit processing to the first `N` samples
- `--no-route-a` : disable fisheye-domain processing
- `--no-route-b` : disable undistorted-domain processing
- `--no-frame-diff` : disable frame differencing
- `--no-farneback` : disable Farneback optical flow
- `--no-lk` : disable Lucas-Kanade sparse flow

## Notes

- Route A processes motion directly in the fisheye image domain.
- Route B undistorts the fisheye images before processing.
- The program generates evaluation results, visualizations, and CSV/chart outputs in the `output` folder.

## Requirements

- CMake 3.20+
- C++17 compiler
- OpenCV with modules: `core`, `imgproc`, `video`, `calib3d`, `highgui`, `imgcodecs`
