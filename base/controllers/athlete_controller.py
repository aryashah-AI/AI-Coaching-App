import os

from flask import render_template, request

from base import app
from base.services.athlete_service import AthleteService, TripleJumpAnalyzer
from base.utils.exception import AppServices
from base.utils.logger import logger


def generate_performance_summary(performance_metrics, overall_score):
    """Generate a text summary of the performance analysis."""
    if not performance_metrics:
        return "No performance metrics available."

    summary_parts = []
    summary_parts.append(f"Overall Performance Score: {overall_score:.1f}/100")

    # Phase scores
    phase_scores = []
    for metric in performance_metrics:
        phase_scores.append(
            f"{metric.phase.title()}: {metric.performance_score:.1f}/100")
    summary_parts.append("Phase Scores: " + ", ".join(phase_scores))

    # Key recommendations
    all_recommendations = []
    for metric in performance_metrics:
        all_recommendations.extend(metric.recommendations)

    if all_recommendations:
        unique_recommendations = list(set(all_recommendations))[
                                 :3]  # Top 3 recommendations
        summary_parts.append("Key Recommendations:")
        for i, rec in enumerate(unique_recommendations, 1):
            summary_parts.append(f"{i}. {rec}")
    else:
        summary_parts.append("Excellent technique across all phases!")

    return "\n".join(summary_parts)


@app.route("/", methods=["GET"])
def upload():
    return render_template("upload.html")


@app.route('/athlete', methods=['GET', 'POST'])
def athlete():
    athlete_service = AthleteService()

    if request.method == 'GET':
        return render_template("upload.html")

    # Handle POST request
    mode = request.form.get("mode", "inference")  # Default to inference
    files = request.files.getlist("files")

    # Validate that files were uploaded
    if not files or all(f.filename == '' for f in files):
        logger.error("No files uploaded")
        return render_template("results.html",
                               error=True,
                               message="Please select at least one video file to upload.")

    try:
        # Save uploaded files
        filepaths = []
        for f in files:
            if f and f.filename:
                filepath = athlete_service.save_file(f)
                if filepath:
                    filepaths.append(filepath)

        if not filepaths:
            logger.error("No valid files were saved")
            return render_template("results.html",
                                   error=True,
                                   message="No valid video files were uploaded. Please check file format and try again.")

        # Handle training mode
        if mode == "train":
            logger.info(f"Starting training with {len(filepaths)} videos")
            model_path = athlete_service.train_model_from_videos(filepaths)

            if model_path:
                logger.info(
                    f"Training completed successfully. Model saved to: {model_path}")
                return render_template("results.html",
                                       mode="training",
                                       training_success=True,
                                       model_path=model_path,
                                       num_videos=len(filepaths),
                                       message="Model training completed successfully!")
            else:
                logger.error("Training failed")
                return render_template("results.html",
                                       mode="training",
                                       training_success=False,
                                       error=True,
                                       message="Training failed. Please check logs for details.")

        # Handle inference mode  
        elif mode == "inference":
            logger.info(
                f"Starting performance analysis on video: {filepaths[0]}")

            # Check if trained model exists
            model_path = "base/static/uploads/triple_jump_model.pkl"

            if os.path.exists(model_path):
                logger.info(
                    f"Using trained model for performance analysis: {model_path}")
                try:
                    # Use performance analyzer with trained model
                    analyzer = TripleJumpAnalyzer(model_path)
                    results, performance_metrics, output_video_path = analyzer.analyze_performance(
                        filepaths[0], "base/static/uploads")

                    # Calculate overall performance score
                    overall_score = sum(m.performance_score for m in
                                        performance_metrics) / len(
                        performance_metrics) if performance_metrics else 0

                    # Generate summary text
                    summary = generate_performance_summary(performance_metrics,
                                                           overall_score)

                    # Create correct path for Flask static file serving
                    logger.info(f"Raw output video path: {output_video_path}")
                    
                    if output_video_path and os.path.exists(output_video_path):
                        # Convert base/static/uploads/video.mp4 to uploads/video.mp4
                        video_filename = output_video_path.replace("base/static/", "")
                        logger.info(f"Converted video path: {video_filename}")
                        logger.info(f"Video file exists: {os.path.exists(output_video_path)}")
                    else:
                        video_filename = None
                        logger.warning(f"Video output not available or doesn't exist: {output_video_path}")

                    return render_template("results.html",
                                           mode="inference",
                                           results=results,
                                           performance_metrics=performance_metrics,
                                           overall_score=overall_score,
                                           summary=summary,
                                           video_out=video_filename,
                                           message="Performance analysis completed successfully!")

                except Exception as e:
                    logger.error(f"Error in performance analysis: {e}")
            else:
                logger.error(
                    "No trained model found. Performing basic analysis only.")
                return render_template("results.html",
                                       error=True,
                                       message="No trained model found. Performing basic analysis only.")

        else:
            logger.error(f"Invalid mode: {mode}")
            return render_template("results.html",
                                   error=True,
                                   message="Invalid analysis mode selected.")

    except Exception as exception:
        logger.error(f"Error in athlete controller: {str(exception)}")
        AppServices.handle_exception(exception)
        return render_template("results.html",
                               error=True,
                               message=f"An error occurred during processing: {str(exception)}")
