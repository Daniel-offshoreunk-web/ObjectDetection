# FTC Object Detection — 2026-27 Season

A lightweight, high-speed computer vision pipeline built with **OpenCV** and an optimized geometric evaluation layer to track multiple game elements simultaneously.

---

## Key Features
* **Multi-Ball Centroid Tracking:** Tracks multiple targets in real-time using custom distance vectors.
* **Hysteresis Memory Buffer:** Mitigates tracking stutter and frame dropouts to ensure smooth swerve drive steering.
* **HSV Resiliency:** Utilizes the HSV color space to minimize vulnerability to changing field lighting conditions.

---

## Performance & Limitations

| Constraint | Details / Impact |
| :--- | :--- |
| **Range Threshold** | Effective tracking occurs up to approximately **1 meter** away due to aggressive background noise filtering. |
| **High Exposure** | Blindingly bright overhead lighting or direct reflections can wash out the HSV signature. |
| **Target Size** | Automatically rejects shapes larger than typical game elements to filter out colored field walls. |

---

## Setup & Installation

If you already have `numpy` and `opencv-python` installed on your system, you can bypass this step.

```bash
pip install -r requirements.txt