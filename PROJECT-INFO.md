# Project Technical & Domain Overview

This document explains how the Triple Jump Analysis System works: the domain concepts, algorithms, code architecture, and how outputs are produced (annotated video, charts, and a performance report).

---

## Domain background: Triple Jump phases
- **Approach**: run-up phase before the first takeoff (not scored as a phase but shown on the timeline).
- **Hop**: takeoff from one foot, landing on the same foot.
- **Step**: takeoff from the same foot, landing on the other foot.
- **Jump**: final takeoff from the other foot, landing in the sand pit.

Key biomechanical signals across phases include lower-limb joint angles (hip, knee, ankle), airborne vs. ground-contact states, and smooth phase transitions (hop → step → jump).

---

## High-level pipeline
1. **Pose estimation**: Extract 2D body keypoints per frame using MediaPipe Pose.
2. **Trajectory building**: Track key joints over time (hips, knees, ankles).
3. **Angle computation**: Compute joint angles frame-by-frame.
4. **Event detection**: Infer takeoff/landing from vertical motion patterns.
5. **Phase classification**: Map event sequences to phases: hop, step, jump.
6. **Visualization**: Produce a static chart and an annotated video overlay.
7. **Model training (optional)**: Aggregate average angles per phase across benchmark videos and store in a pickle file.
8. **Performance analysis (optional)**: Compare a new video’s phase averages to the trained model; score and generate recommendations.

---

## Core components in `app.py`

### `TripleJumpDetector`
- Initializes MediaPipe Pose once for efficiency.
- `extract_keypoints(video_path)`: reads frames via OpenCV, runs pose detection, stores pixel-space coordinates for:
  - `left_ankle (27)`, `right_ankle (28)`, `left_knee (25)`, `right_knee (26)`, `left_hip (23)`, `right_hip (24)`.
  - Missing detections are stored as `None`, preserving alignment.
- `calculate_joint_angles(trajectories)`: computes angles for knees and hips using the vectors defined by three points and arccos of the normalized dot product.
- `detect_events(foot_left_traj, foot_right_traj)`:
  - Cleans trajectories to indices where both feet are detected.
  - Uses the pixel y-coordinate (image origin at top-left) to derive a ground-contact signal as `max(left_y, right_y)`.
  - Computes vertical velocity via `np.gradient`, smooths with `scipy.ndimage.gaussian_filter1d`.
  - Detects transitions:
    - Takeoff when velocity crosses from positive to sufficiently negative.
    - Landing when velocity crosses from negative to sufficiently positive.
  - Debounces events with `min_distance` frames and alternates to enforce logical sequences.
- `classify_phases(events)`: groups pairs of `takeoff→landing` into `['hop','step','jump']`. Incomplete sequences are handled robustly.
- `visualize_results(...)`: creates a 3-panel figure saved as `analysis_results_<video>.png`:
  - Foot trajectories (y vs. frame)
  - Knee angle curves
  - Phase timeline (categorical values mapped to integers)
- `create_annotated_video(...)`: overlays on each frame:
  - MediaPipe pose skeleton
  - Short trailing trajectories for ankles and knees
  - Current phase label with color coding
  - Event markers for detected takeoffs/landings
  - Instantaneous knee angles when available

Implementation details worth noting:
- Phases and events are mapped to frame indices with simple uniform spacing (`_get_event_frames`) to provide readable markers even if the exact event frame is ambiguous.
- Phase timeline is discretized by dividing video into equal segments per detected phase (`_create_phase_timeline`).

### `TripleJumpModelTrainer`
- Processes multiple videos from `dataset/` using the detector to derive angles and phases.
- `_extract_phase_angles(...)` slices the time series into per-phase windows, aggregates all frames’ angles per phase.
- `create_model(...)` composes a `TripleJumpModel` (dataclass) with average angles per phase and saves a pickle to `ai_model/triple_jump_model.pkl`.

### `TripleJumpAnalyzer`
- Loads the trained `TripleJumpModel`.
- `analyze_performance(video_path, output_dir)`: runs the detector on the input video, computes its per-phase average angles, compares to the model’s references, and compiles `PerformanceMetrics`:
  - Angle differences per phase
  - A simple score (0–100) where lower average absolute deviation → higher score
  - Human-readable recommendations when deviations exceed 10° thresholds
- `_generate_report(...)` writes `performance_report_<video>.txt` containing:
  - Overall score
  - Phase-wise scores and angle comparisons
  - Specific recommendations list

---

## Algorithms and design choices

- **Pose estimation (MediaPipe Pose)**: Robust, real-time-capable 2D keypoint extractor. The code uses `model_complexity=2` and confidence thresholds of 0.5 for detection and tracking.
- **Event detection**: Rather than explicit foot-contact segmentation, this implementation infers contact events from vertical y-position dynamics of the lower foot, using sign changes in smoothed velocity. This approach is simple, interpretable, and works without force plates.
- **Phase classification**: Enforces alternation of takeoff/landing and truncates inconsistent sequences to maintain hop→step→jump ordering.
- **Averaging by phase**: For clarity and robustness, the timeline is partitioned equally across detected phases. This sacrifices some temporal precision but simplifies aggregation and reporting.
- **Scoring**: The performance score maps mean absolute angle deviation to a 0–100 scale with 20° mapped to 0; tunable to domain needs.

Trade-offs:
- Uniform phase partitioning and approximate event frames are pragmatic and stable for varied footage; for research-grade precision you can replace with foot-contact detection using velocity minima, zero-crossings with hysteresis, or learned contact classifiers.

---

## Outputs explained (`output/`)
- `analysis_results_<video>.png`: 3 charts summarizing trajectories, knee angles, and phase timeline.
- `<video>_output.mp4`: original video with overlays: skeleton, trails, phase labels, event indicators, angles, and legend.
- `performance_report_<video>.txt`: comprehensive textual summary. Example metrics seen in the repo include overall and per-phase scores and specific coaching feedback (e.g., "left knee is over-flexed").

---

## Extending or customizing
- Tune thresholds in `detect_events` (e.g., `min_height_threshold`, `min_distance`).
- Include more joints (e.g., ankles’ angles) and add new recommendation rules.
- Replace equal-length phase partitioning with event-timestamped windows.
- Persist raw trajectories and angles to CSV/Parquet for further analysis.
- Swap MediaPipe with alternative pose estimators if needed (e.g., OpenPose, MoveNet).

---

## Limitations and considerations
- 2D keypoints are sensitive to camera angle and occlusion; ensure clear side view.
- Pixel coordinates mean angle consistency depends on stable cropping and scale; camera motion may degrade results.
- The trained model is only as good as the benchmark videos; include diverse, high-quality exemplars.

---

## References
- MediaPipe Pose: high-fidelity, cross-platform 2D pose tracking.
- OpenCV: frame I/O and visualization utilities.
- NumPy/SciPy: numerical operations, smoothing, and gradients.
- Matplotlib: static charts summarizing the analysis.
