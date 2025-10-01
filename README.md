# Triple Jump Analysis System

AI based custom project to detect triple jump phases (hop, step, jump) using MediaPipe Pose and computer vision, generate annotated output videos, charts, and a performance report using a trained reference model.

---

## Quick Start

### 1) Prerequisites
- Python 3.9+ recommended
- OS: Linux/macOS/Windows
- FFmpeg (optional but recommended for best MP4 compatibility)

### 2) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Project structure
```
Triple-Jump/
  app.py
  ai_model/
    triple_jump_model.pkl        # created after training (or already provided)
  dataset/
    Benchmark_1.mp4
    Benchmark_2.mp4
    Benchmark_3.mp4
    Benchmark_4.mp4
  input/
    input_video_1.mp4
    input_video_2.mp4
    input_video_3.mp4
    input_video_4.mp4            # default input used by app.py
  output/                        # results are written here
  requirements.txt
```

---

## How it works (at a glance)
- Extracts pose keypoints per frame via MediaPipe.
- Computes joint angles and detects takeoff/landing events.
- Classifies phases: hop → step → jump.
- Optionally trains a reference model (average angles per phase) from `dataset/` videos.
- Analyzes an `input/` video against the reference model and generates:
  - Annotated MP4 with overlays
  - Analysis PNG chart
  - Text report with phase-wise scores and recommendations

---

## Running the program

The entrypoint is `app.py`. By default it targets `input/input_video_4.mp4` and writes outputs to `output/`.

```bash
python app.py
```

At runtime, you will be prompted:
- If a model exists at `ai_model/triple_jump_model.pkl`, choose:
  1. Use existing model for analysis
  2. Retrain model
- If a model does not exist, choose:
  1. Train new model (mapped to retrain)
  2. Basic analysis only

Outputs appear in `output/`:
- `<video_name>_output.mp4` (annotated video)
- `analysis_results_<video_name>.png` (charts)
- `performance_report_<video_name>.txt` (detailed report)

---

## Typical workflows

### A) Use existing model for analysis
1. Ensure `ai_model/triple_jump_model.pkl` exists.
2. Place your test video in `input/` and set `INPUT_VIDEO_FILE_PATH` in `app.py` if needed.
3. Run `python app.py` and choose option `1`.

### B) Train a new model, then analyze
1. Put 2+ benchmark videos into `dataset/` (already included as examples).
2. Run `python app.py` and choose option `2`.
3. The script will train a model from `dataset/` and then analyze the target input video.

### C) Basic analysis only (no model comparison)
- If you choose basic analysis, the system still extracts phases/angles and produces the annotated video and charts. The performance report requires a trained model.

---

## Configuration
- Edit these constants in `app.py` as needed:
  - `AI_MODEL_FILE_PATH`: model pickle path (default: `ai_model/triple_jump_model.pkl`)
  - `DATASET_FILE_PATH_LIST`: training video list (defaults to items in `dataset/`)
  - `INPUT_VIDEO_FILE_PATH`: the input video to analyze (default: `input/input_video_4.mp4`)
  - `OUTPUT_DIR`: output folder (default: `output/`)

---

## Troubleshooting
- OpenCV cannot open video:
  - Verify the path and codec. Try converting with FFmpeg: `ffmpeg -i input.mp4 -vcodec libx264 -acodec aac output_fixed.mp4`.
- MediaPipe runtime errors on CPU-only systems:
  - Upgrade `mediapipe` and `opencv-python` to latest; ensure Python version compatibility.
- Empty or noisy detections:
  - Ensure the athlete is clearly visible and well-lit; adjust thresholds in `detect_events` if necessary.
- No model found / report missing:
  - Train first (option 2) to generate `ai_model/triple_jump_model.pkl`.

---

## License
This project is provided as-is for research and educational use.
