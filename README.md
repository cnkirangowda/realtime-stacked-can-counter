 Real-Time Stacked Can Counter (Computer Vision)

A real-time computer vision system that counts stacked cans from a single overhead camera using pinhole geometry, edge-band rim detection, and temporal state tracking.

Built for industrial inventory monitoring and automated quality control on packaging and production lines.

---

 Key Features

- Real-Time Performance: Runs at 25–30 FPS on standard camera feeds.
- Multi-Height Stack Detection: Automatically detects stack heights (1, 2, 3+ cans high) using depth-to-radius projection.
- Interactive 3-Point Calibration: Simple 3-point rim click to establish camera geometry constant (`K`).
- Robust Rim Fitting:** 72-ray edge search with iterative least-squares and outlier rejection (RMS error < 2.5px).
- Material Validation:** HSV-based silver surface check to eliminate false positives.
- Temporal Tracking & Smoothing:** Majority-vote state filtering over rolling frames to eliminate flicker.
- Per-Layer & Missing Cans Reporting:** Real-time count of total cans, layer breakdown, and missing units.

---

 How It Works

The system operates on the pinhole camera model:

1. Calibration: Clicking 3 points around the rim of top-level cans calculates the camera constant:
   K = cam_dist_mm * r_top
