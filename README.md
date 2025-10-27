# AI Coaching App — Triple Jump (Flask Dashboard)

AI-powered Flask dashboard to analyze triple jump technique. Upload your video(
s), detect phases (hop, step, jump) using MediaPipe Pose, and get an annotated
output video, analysis chart, and a detailed performance report. You can also
train a reference model from multiple videos and evaluate new runs against it.

---

## Quick Start

### 1) Prerequisites

- Python 3.9+ (Linux/macOS/Windows)
- FFmpeg (optional; useful for video codec conversions)
- System packages required by OpenCV/MediaPipe as per your OS

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

### 4) Run the Flask server

```bash
python main.py
```

Then open your browser at `http://127.0.0.1:5000/` (or
`http://localhost:5000/`).

---

## Project Structure

```
AI-Coaching-App/
  main.py                          # Flask entrypoint
  requirements.txt
  logs/
    base.log                      # application logs
  base/
    __init__.py                   # creates Flask app (templates/static registered)
    controllers/
      athlete_controller.py       # routes: upload page and analysis handler
    services/
      athlete_service.py          # core CV/ML logic (MediaPipe, angles, events, video annotate)
    templates/
      upload.html                 # dashboard UI (mode select + upload)
      results.html                # results UI (video, metrics, summary)
    static/
      uploads/                    # uploaded videos and generated outputs
        Benchmark_1.mp4
        Benchmark_2.mp4
        Benchmark_3.mp4
        Benchmark_4.mp4
        Benchmark_1_annotated.webm
        analysis_results.png
        performance_report_Benchmark_1.txt
        triple_jump_model.pkl     # created after training
    utils/
      logger.py                   # writes to logs/base.log
      exception.py
  test/
    ai_test_app.py
```

---

## Using the Dashboard

1) Open the dashboard at `/` and choose an Analysis Mode:

- Training Mode (`mode=train`)
    - Upload 2 or more videos; the app will compute phase-wise average angles
      and create a model at `base/static/uploads/triple_jump_model.pkl`.
- Analysis Mode (`mode=inference`)
    - Upload a video to analyze. If a trained model exists, the app computes
      performance metrics and recommendations compared to the model.

2) Upload your video files (MP4/AVI/MOV)

- Files are saved to `base/static/uploads/`.

3) Submit to start processing

- You’ll be redirected to the results page when done.

---

## Outputs

All generated files are written to `base/static/uploads/`:

- Annotated video: `<video_name>_annotated.webm`
- Analysis chart: `analysis_results.png`
- Performance report: `performance_report_<video_name>.txt`

Example artifacts already in this repository:

- Annotated
  sample: [Benchmark_1_annotated.webm](base/static/uploads/Benchmark_1_annotated.webm)
- Analysis
  chart: [analysis_results.png](base/static/uploads/analysis_results.png)
-
Report: [performance_report_Benchmark_1.txt](base/static/uploads/performance_report_Benchmark_1.txt)

Note: The annotated video is generated with a WebM (VP8) codec for broad
browser compatibility.

---

## Routes and Flow

- `GET /` → renders `upload.html` (mode selection + file upload UI)
- `POST /athlete` → handles Training or Analysis
    - Saves uploads to `base/static/uploads/`
    - Training: builds model and shows training summary on `results.html`
    - Analysis: runs detection, compares with model (if present), and shows
      annotated video, metrics, and summary on `results.html`

---

## Configuration and Paths

- Uploads directory: `base/services/athlete_service.py` →
  `UPLOAD_FOLDER = base/static/uploads`
- Allowed formats: `mp4`, `avi`, `mov`
- Trained model: `base/static/uploads/triple_jump_model.pkl`
- Logs: `logs/base.log` (configured in `base/utils/logger.py`)

---

## Troubleshooting

- OpenCV cannot open video
    - Verify path/codec; try converting with FFmpeg, e.g.:
  ```bash
  ffmpeg -i input.mp4 -c:v libvpx -b:v 2M -c:a libvorbis output.webm
  ```
- MediaPipe runtime errors
    - Upgrade `mediapipe` and `opencv-python` to match your Python/OS; ensure
      you have system dependencies installed.
- “No trained model found” in Analysis Mode
    - First run Training Mode with 2+ videos to create `triple_jump_model.pkl`.
- Browser cannot play video
    - Use a modern browser (WebM support). If needed, transcode to another
      format with FFmpeg.

---

## License

This project is provided as-is for research and educational use.
