#!/usr/bin/env python3
"""
Complete Triple Jump Phase Detection System
Detects hop, step, and jump phases using pose estimation and biomechanical analysis.
"""

import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from scipy.ndimage import gaussian_filter1d

AI_MODEL_FILE_PATH = "ai_model/triple_jump_model.pkl"

DATASET_FILE_PATH_LIST = ["dataset/Benchmark_1.mp4", "dataset/Benchmark_2.mp4",
                          "dataset/Benchmark_3.mp4", "dataset/Benchmark_4.mp4"]

INPUT_VIDEO_FILE_PATH = "input/input_video_4.mp4"

OUTPUT_DIR = "output/"


class TripleJumpDetector:
    def __init__(self):
        """Initialize MediaPipe pose detection."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def extract_keypoints(self, video_path):
        """
        Extract pose keypoints from video.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with keypoint trajectories
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        trajectories = {
            'left_ankle': [],
            'right_ankle': [],
            'left_knee': [],
            'right_knee': [],
            'left_hip': [],
            'right_hip': [],
            'frame_count': 0
        }

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Extract relevant keypoints (normalized coordinates)
                h, w = frame.shape[:2]

                # Left ankle (landmark 27)
                left_ankle = landmarks[27]
                trajectories['left_ankle'].append(
                    (left_ankle.x * w, left_ankle.y * h))

                # Right ankle (landmark 28)
                right_ankle = landmarks[28]
                trajectories['right_ankle'].append(
                    (right_ankle.x * w, right_ankle.y * h))

                # Left knee (landmark 25)
                left_knee = landmarks[25]
                trajectories['left_knee'].append(
                    (left_knee.x * w, left_knee.y * h))

                # Right knee (landmark 26)
                right_knee = landmarks[26]
                trajectories['right_knee'].append(
                    (right_knee.x * w, right_knee.y * h))

                # Left hip (landmark 23)
                left_hip = landmarks[23]
                trajectories['left_hip'].append(
                    (left_hip.x * w, left_hip.y * h))

                # Right hip (landmark 24)
                right_hip = landmarks[24]
                trajectories['right_hip'].append(
                    (right_hip.x * w, right_hip.y * h))
            else:
                # Fill with None if no pose detected
                for key in trajectories:
                    if key != 'frame_count':
                        trajectories[key].append(None)

            frame_count += 1

        cap.release()
        trajectories['frame_count'] = frame_count

        return trajectories

    def calculate_joint_angles(self, trajectories):
        """
        Calculate hip, knee, and ankle angles from keypoints.

        Args:
            trajectories: Dictionary with keypoint trajectories

        Returns:
            Dictionary with joint angles over time
        """
        angles = {
            'left_hip_angle': [],
            'right_hip_angle': [],
            'left_knee_angle': [],
            'right_knee_angle': []
        }

        for i in range(len(trajectories['left_ankle'])):
            if (trajectories['left_ankle'][i] is not None and
                    trajectories['left_knee'][i] is not None and
                    trajectories['left_hip'][i] is not None):

                # Left leg angles
                left_hip_angle = self._calculate_angle(
                    trajectories['left_hip'][i],
                    trajectories['left_knee'][i],
                    trajectories['left_ankle'][i]
                )
                angles['left_hip_angle'].append(left_hip_angle)

                left_knee_angle = self._calculate_angle(
                    trajectories['left_hip'][i],
                    trajectories['left_knee'][i],
                    trajectories['left_ankle'][i]
                )
                angles['left_knee_angle'].append(left_knee_angle)
            else:
                angles['left_hip_angle'].append(None)
                angles['left_knee_angle'].append(None)

            if (trajectories['right_ankle'][i] is not None and
                    trajectories['right_knee'][i] is not None and
                    trajectories['right_hip'][i] is not None):

                # Right leg angles
                right_hip_angle = self._calculate_angle(
                    trajectories['right_hip'][i],
                    trajectories['right_knee'][i],
                    trajectories['right_ankle'][i]
                )
                angles['right_hip_angle'].append(right_hip_angle)

                right_knee_angle = self._calculate_angle(
                    trajectories['right_hip'][i],
                    trajectories['right_knee'][i],
                    trajectories['right_ankle'][i]
                )
                angles['right_knee_angle'].append(right_knee_angle)
            else:
                angles['right_hip_angle'].append(None)
                angles['right_knee_angle'].append(None)

        return angles

    def _calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points."""
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (
                np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    def detect_events(self, foot_left_traj, foot_right_traj,
                      min_height_threshold=0.1, min_distance=10):
        """
        Detect triple jump events (takeoffs and landings) from foot trajectories.

        Args:
            foot_left_traj: List of (x, y) coordinates for left foot
            foot_right_traj: List of (x, y) coordinates for right foot
            min_height_threshold: Minimum height change to consider as takeoff/landing
            min_distance: Minimum frames between consecutive events

        Returns:
            List of event types: ['takeoff', 'landing', 'takeoff', 'landing', ...]
        """
        # Filter out None values
        valid_indices = []
        left_y_clean = []
        right_y_clean = []

        for i, (left, right) in enumerate(
                zip(foot_left_traj, foot_right_traj)):
            if left is not None and right is not None:
                valid_indices.append(i)
                left_y_clean.append(left[1])
                right_y_clean.append(right[1])

        if len(left_y_clean) < 3:
            return []

        left_y = np.array(left_y_clean)
        right_y = np.array(right_y_clean)

        # Use the lower foot (higher y-value in image coordinates) as ground contact reference
        ground_y = np.maximum(left_y, right_y)

        events = []
        event_frames = []

        # Calculate vertical velocity (derivative)
        velocity = np.gradient(ground_y)

        # Smooth the velocity to reduce noise
        try:
            velocity_smooth = gaussian_filter1d(velocity, sigma=2.0)
        except ImportError:
            # Fallback: simple moving average if scipy not available
            velocity_smooth = np.convolve(velocity, np.ones(5) / 5,
                                          mode='same')

        # Detect takeoffs and landings
        for i in range(1, len(velocity_smooth) - 1):
            # Skip if too close to previous event
            if event_frames and i - event_frames[-1] < min_distance:
                continue

            # Takeoff: transition from positive to negative velocity (going up)
            if (velocity_smooth[i - 1] > 0 and velocity_smooth[
                i] < -min_height_threshold):
                events.append("takeoff")
                event_frames.append(
                    valid_indices[i] if i < len(valid_indices) else i)

            # Landing: transition from negative to positive velocity (going down)
            elif (velocity_smooth[i - 1] < 0 and velocity_smooth[
                i] > min_height_threshold):
                events.append("landing")
                event_frames.append(
                    valid_indices[i] if i < len(valid_indices) else i)

        # Post-process events to ensure logical sequence
        filtered_events = []
        for i, event in enumerate(events):
            if i == 0 or event != events[i - 1]:
                filtered_events.append(event)

        # Ensure we start with a takeoff
        if filtered_events and filtered_events[0] == "landing":
            filtered_events = filtered_events[1:]

        # Ensure alternating pattern
        final_events = []
        expected_next = "takeoff"

        for event in filtered_events:
            if event == expected_next:
                final_events.append(event)
                expected_next = "landing" if event == "takeoff" else "takeoff"

        # Handle the case where the last event should be a landing
        if final_events and len(final_events) % 2 == 1:
            if len(final_events) >= 3:
                final_events.append("landing")

        return final_events

    def classify_phases(self, events):
        """
        Classify triple jump phases based on detected events.

        Args:
            events: List of event types from detect_events()

        Returns:
            List of phase labels corresponding to segments between events
        """
        if len(events) < 2:
            return ["approach"] if len(events) == 0 else ["hop"]

        phases = []
        phase_names = ["hop", "step", "jump"]
        phase_index = 0

        # Each pair of takeoff->landing represents one phase
        for i in range(0, len(events) - 1, 2):
            if i + 1 < len(events) and events[i] == "takeoff" and events[
                i + 1] == "landing":
                if phase_index < len(phase_names):
                    phases.append(phase_names[phase_index])
                    phase_index += 1
                else:
                    phases.append("landing")

        # Handle incomplete sequences
        if len(events) % 2 == 1 and events[-1] == "takeoff":
            if phase_index < len(phase_names):
                phases.append(phase_names[phase_index])

        return phases

    def safe_detect_events(self, foot_left_traj, foot_right_traj):
        """Safe wrapper for event detection with error handling."""
        try:
            if not foot_left_traj or not foot_right_traj:
                print("Warning: Empty trajectory data")
                return []

            if len(foot_left_traj) != len(foot_right_traj):
                print("Warning: Trajectory lengths don't match")
                min_len = min(len(foot_left_traj), len(foot_right_traj))
                foot_left_traj = foot_left_traj[:min_len]
                foot_right_traj = foot_right_traj[:min_len]

            events = self.detect_events(foot_left_traj, foot_right_traj)

            if not events:
                print(
                    "Warning: No events detected. Check trajectory data and thresholds.")
                return []

            return events

        except Exception as e:
            print(f"Error in event detection: {e}")
            return []

    def visualize_results(self, trajectories, events, phases,
                          output_path=None):
        """
        Visualize the analysis results.

        Args:
            trajectories: Keypoint trajectories
            events: Detected events
            phases: Classified phases
            output_path: Optional path to save visualization
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        # Plot 1: Foot trajectories
        left_y = [pos[1] if pos else None for pos in
                  trajectories['left_ankle']]
        right_y = [pos[1] if pos else None for pos in
                   trajectories['right_ankle']]

        frames = range(len(left_y))
        axes[0].plot(frames, left_y, 'b-', label='Left Ankle', alpha=0.7)
        axes[0].plot(frames, right_y, 'r-', label='Right Ankle', alpha=0.7)
        axes[0].set_ylabel('Y Position (pixels)')
        axes[0].set_title('Foot Trajectories')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Joint angles
        angles = self.calculate_joint_angles(trajectories)
        left_knee = [angle if angle else None for angle in
                     angles['left_knee_angle']]
        right_knee = [angle if angle else None for angle in
                      angles['right_knee_angle']]

        axes[1].plot(frames, left_knee, 'g-', label='Left Knee Angle',
                     alpha=0.7)
        axes[1].plot(frames, right_knee, 'm-', label='Right Knee Angle',
                     alpha=0.7)
        axes[1].set_ylabel('Angle (degrees)')
        axes[1].set_title('Joint Angles')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Phase detection results
        phase_timeline = np.zeros(len(frames))
        phase_colors = {'hop': 1, 'step': 2, 'jump': 3, 'approach': 0,
                        'landing': 4}

        # Color code the timeline based on phases
        if phases:
            phase_duration = len(frames) // max(len(phases), 1)
            for i, phase in enumerate(phases):
                start_idx = i * phase_duration
                end_idx = min((i + 1) * phase_duration, len(frames))
                phase_timeline[start_idx:end_idx] = phase_colors.get(phase, 0)

        axes[2].plot(frames, phase_timeline, 'k-', linewidth=3)
        axes[2].set_ylabel('Phase')
        axes[2].set_xlabel('Frame')
        axes[2].set_title('Detected Phases')
        axes[2].set_yticks(list(phase_colors.values()))
        axes[2].set_yticklabels(list(phase_colors.keys()))
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {output_path}")
        else:
            plt.show()

    def create_annotated_video(self, video_path, results, output_video_path):
        """
        Create an annotated video with overlaid analysis results.

        Args:
            video_path: Path to original video
            results: Analysis results dictionary
            output_video_path: Path to save annotated video
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        trajectories = results['trajectories']
        events = results['events']
        phases = results['phases']
        angles = results['angles']

        # Determine current phase for each frame
        phase_timeline = self._create_phase_timeline(
            len(trajectories['left_ankle']), phases)

        # Determine event frames
        event_frames = self._get_event_frames(trajectories, events)

        frame_idx = 0
        trajectory_history = {
            'left_ankle': [],
            'right_ankle': [],
            'left_knee': [],
            'right_knee': []
        }

        print(f"Creating annotated video: {output_video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process pose for current frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_pose = self.pose.process(frame_rgb)

            # Draw pose landmarks if detected
            if results_pose.pose_landmarks:
                # Draw pose connections
                self.mp_drawing.draw_landmarks(
                    frame, results_pose.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2,
                                                circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

            # Add trajectory points to history
            if frame_idx < len(trajectories['left_ankle']):
                for joint in ['left_ankle', 'right_ankle', 'left_knee',
                              'right_knee']:
                    if trajectories[joint][frame_idx] is not None:
                        trajectory_history[joint].append(
                            trajectories[joint][frame_idx])
                        # Keep only last 30 points for trajectory trail
                        if len(trajectory_history[joint]) > 30:
                            trajectory_history[joint].pop(0)

            # Draw trajectory trails
            colors = {
                'left_ankle': (255, 0, 0),  # Blue
                'right_ankle': (0, 0, 255),  # Red
                'left_knee': (0, 255, 0),  # Green
                'right_knee': (255, 0, 255)  # Magenta
            }

            for joint, color in colors.items():
                if len(trajectory_history[joint]) > 1:
                    for i in range(1, len(trajectory_history[joint])):
                        pt1 = tuple(map(int, trajectory_history[joint][i - 1]))
                        pt2 = tuple(map(int, trajectory_history[joint][i]))
                        alpha = i / len(trajectory_history[joint])
                        thickness = max(1, int(3 * alpha))
                        cv2.line(frame, pt1, pt2, color, thickness)

            # Add text overlays
            y_offset = 30
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2

            # Frame count
            cv2.putText(frame,
                        f"Frame: {frame_idx + 1}/{results['frame_count']}",
                        (10, y_offset), font, font_scale, (255, 255, 255),
                        thickness)
            y_offset += 30

            # Current phase
            if frame_idx < len(phase_timeline):
                current_phase = phase_timeline[frame_idx]
                phase_color = self._get_phase_color(current_phase)
                cv2.putText(frame, f"Phase: {current_phase.upper()}",
                            (10, y_offset), font, font_scale, phase_color,
                            thickness)
                y_offset += 30

            # Event indicator
            if frame_idx in event_frames:
                event_type = event_frames[frame_idx]
                event_color = (0, 255, 255) if event_type == 'takeoff' else (
                    255, 255, 0)
                cv2.putText(frame, f"EVENT: {event_type.upper()}",
                            (10, y_offset), font, font_scale, event_color,
                            thickness)
                # Draw event marker circle
                cv2.circle(frame, (width - 50, 50), 20, event_color, -1)
                y_offset += 30

            # Joint angles
            if frame_idx < len(angles['left_knee_angle']):
                if angles['left_knee_angle'][frame_idx] is not None:
                    left_knee_angle = angles['left_knee_angle'][frame_idx]
                    cv2.putText(frame, f"L Knee: {left_knee_angle:.1f}°",
                                (10, y_offset), font, 0.6, (0, 255, 0), 2)
                    y_offset += 25

                if angles['right_knee_angle'][frame_idx] is not None:
                    right_knee_angle = angles['right_knee_angle'][frame_idx]
                    cv2.putText(frame, f"R Knee: {right_knee_angle:.1f}°",
                                (10, y_offset), font, 0.6, (255, 0, 255), 2)
                    y_offset += 25

            # Add legend
            self._draw_legend(frame, width, height)

            # Write frame to output video
            out.write(frame)
            frame_idx += 1

            # Progress indicator
            if frame_idx % 30 == 0:
                progress = (frame_idx / results['frame_count']) * 100
                print(f"Progress: {progress:.1f}%")

        # Release everything
        cap.release()
        out.release()
        print(f"Annotated video saved to: {output_video_path}")

    def _create_phase_timeline(self, num_frames, phases):
        """Create timeline showing which phase each frame belongs to."""
        timeline = ['approach'] * num_frames

        if phases and len(phases) > 0:
            phase_duration = num_frames // max(len(phases), 1)
            for i, phase in enumerate(phases):
                start_idx = i * phase_duration
                end_idx = min((i + 1) * phase_duration, num_frames)
                for j in range(start_idx, end_idx):
                    timeline[j] = phase

        return timeline

    def _get_event_frames(self, trajectories, events):
        """Map events to approximate frame numbers."""
        event_frames = {}
        if not events:
            return event_frames

        # Simple approximation: distribute events evenly across video
        num_frames = len(trajectories['left_ankle'])
        event_spacing = num_frames // max(len(events), 1)

        for i, event in enumerate(events):
            frame_num = min(i * event_spacing, num_frames - 1)
            event_frames[frame_num] = event

        return event_frames

    def _get_phase_color(self, phase):
        """Get color for phase text."""
        phase_colors = {
            'approach': (128, 128, 128),  # Gray
            'hop': (0, 255, 0),  # Green
            'step': (255, 165, 0),  # Orange
            'jump': (255, 0, 0),  # Red
            'landing': (0, 0, 255)  # Blue
        }
        return phase_colors.get(phase, (255, 255, 255))

    def _draw_legend(self, frame, width, height):
        """Draw legend on the frame."""
        legend_x = width - 200
        legend_y = height - 150

        # Background for legend
        cv2.rectangle(frame, (legend_x - 10, legend_y - 10),
                      (width - 10, height - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (legend_x - 10, legend_y - 10),
                      (width - 10, height - 10), (255, 255, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5

        # Legend items
        legend_items = [
            ("Left Ankle", (255, 0, 0)),
            ("Right Ankle", (0, 0, 255)),
            ("Left Knee", (0, 255, 0)),
            ("Right Knee", (255, 0, 255))
        ]

        for i, (label, color) in enumerate(legend_items):
            y_pos = legend_y + i * 20
            cv2.circle(frame, (legend_x, y_pos), 5, color, -1)
            cv2.putText(frame, label, (legend_x + 15, y_pos + 5),
                        font, font_scale, (255, 255, 255), 1)

    def process_video(self, video_path, output_dir=None,
                      create_video_output=True):
        """
        Complete processing pipeline for triple jump video analysis.

        Args:
            video_path: Path to input video
            output_dir: Optional directory to save results
            create_video_output: Whether to create annotated video output

        Returns:
            Dictionary with analysis results
        """
        print(f"Processing video: {video_path}")
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Extract keypoints
        print("Extracting pose keypoints...")
        trajectories = self.extract_keypoints(video_path)

        # Detect events
        print("Detecting takeoff and landing events...")
        events = self.safe_detect_events(
            trajectories['left_ankle'],
            trajectories['right_ankle']
        )

        # Classify phases
        print("Classifying triple jump phases...")
        phases = self.classify_phases(events)

        # Calculate joint angles
        print("Calculating joint angles...")
        angles = self.calculate_joint_angles(trajectories)

        results = {
            'trajectories': trajectories,
            'events': events,
            'phases': phases,
            'angles': angles,
            'frame_count': trajectories['frame_count']
        }

        # Print results
        print(f"\nAnalysis Results:")
        print(f"Total frames: {results['frame_count']}")
        print(f"Detected events: {events}")
        print(f"Identified phases: {phases}")

        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # Visualize results (static plots)
            viz_path = os.path.join(output_dir,
                                    'analysis_results_{}.png'.format(
                                        video_name))
            self.visualize_results(trajectories, events, phases, viz_path)

            # Create annotated video output
            if create_video_output:
                print("\nCreating annotated video output...")
                output_video_path = os.path.join(output_dir,
                                                 f'{video_name}_output.mp4')
                try:
                    self.create_annotated_video(video_path, results,
                                                output_video_path)
                except Exception as e:
                    print(f"Warning: Could not create annotated video: {e}")
        else:
            self.visualize_results(trajectories, events, phases)

        return results


@dataclass
class PhaseAngles:
    """Data class for storing phase-specific angle data."""
    left_knee_angles: List[float]
    right_knee_angles: List[float]
    left_hip_angles: List[float]
    right_hip_angles: List[float]

    def get_average_angles(self) -> Dict[str, float]:
        """Calculate average angles for this phase."""
        return {
            'left_knee_avg': np.mean(
                self.left_knee_angles) if self.left_knee_angles else 0.0,
            'right_knee_avg': np.mean(
                self.right_knee_angles) if self.right_knee_angles else 0.0,
            'left_hip_avg': np.mean(
                self.left_hip_angles) if self.left_hip_angles else 0.0,
            'right_hip_avg': np.mean(
                self.right_hip_angles) if self.right_hip_angles else 0.0,
        }


@dataclass
class TripleJumpModel:
    """Data class for storing the trained triple jump model."""
    hop_angles: Dict[str, float]
    step_angles: Dict[str, float]
    jump_angles: Dict[str, float]
    training_videos: List[str]
    creation_date: str
    version: str = "1.0"


@dataclass
class PerformanceMetrics:
    """Data class for storing performance analysis results."""
    phase: str
    reference_angles: Dict[str, float]
    actual_angles: Dict[str, float]
    differences: Dict[str, float]
    recommendations: List[str]
    performance_score: float


class TripleJumpModelTrainer:
    """
    Class for training triple jump models from multiple video files.
    Creates pickle-based models with average angles per phase.
    """

    def __init__(self):
        """Initialize the model trainer."""
        self.detector = TripleJumpDetector()

    def process_training_videos(self, video_paths: List[str]) -> Dict[
        str, PhaseAngles]:
        """
        Process multiple training videos to extract phase-wise angle data.

        Args:
            video_paths: List of paths to training videos

        Returns:
            Dictionary containing phase-wise angle data
        """
        phase_data = {
            'hop': PhaseAngles([], [], [], []),
            'step': PhaseAngles([], [], [], []),
            'jump': PhaseAngles([], [], [], [])
        }

        successful_videos = 0

        for video_path in video_paths:
            try:
                if not os.path.exists(video_path):
                    continue

                # Process video using existing detector
                results = self.detector.process_video(video_path,
                                                      create_video_output=False)

                # Extract phase-wise angles
                self._extract_phase_angles(results, phase_data)
                successful_videos += 1

            except Exception as e:
                continue

        return phase_data

    def _extract_phase_angles(self, results: Dict,
                              phase_data: Dict[str, PhaseAngles]):
        """Extract angles for each detected phase."""
        phases = results.get('phases', [])
        angles = results.get('angles', {})
        frame_count = results.get('frame_count', 0)

        if not phases or not angles:
            return

        # Calculate frames per phase
        frames_per_phase = frame_count // max(len(phases), 1)

        for phase_idx, phase in enumerate(phases):
            if phase not in phase_data:
                continue

            start_frame = phase_idx * frames_per_phase
            end_frame = min((phase_idx + 1) * frames_per_phase, frame_count)

            # Extract angles for this phase
            for frame_idx in range(start_frame, end_frame):
                if frame_idx < len(angles.get('left_knee_angle', [])):
                    left_knee = angles['left_knee_angle'][frame_idx]
                    right_knee = angles['right_knee_angle'][frame_idx]
                    left_hip = angles.get('left_hip_angle', [None] * len(
                        angles['left_knee_angle']))[frame_idx]
                    right_hip = angles.get('right_hip_angle', [None] * len(
                        angles['left_knee_angle']))[frame_idx]

                    if left_knee is not None:
                        phase_data[phase].left_knee_angles.append(left_knee)
                    if right_knee is not None:
                        phase_data[phase].right_knee_angles.append(right_knee)
                    if left_hip is not None:
                        phase_data[phase].left_hip_angles.append(left_hip)
                    if right_hip is not None:
                        phase_data[phase].right_hip_angles.append(right_hip)

    def create_model(self, video_paths: List[str],
                     ai_model_file_path: str) -> TripleJumpModel:
        """
        Create and save a triple jump model from training videos.

        Args:
            video_paths: List of training video paths
            ai_model_file_path: Path to save the model pickle file

        Returns:
            Trained TripleJumpModel
        """

        # Process all training videos
        phase_data = self.process_training_videos(video_paths)

        # Calculate average angles per phase
        model = TripleJumpModel(
            hop_angles=phase_data['hop'].get_average_angles(),
            step_angles=phase_data['step'].get_average_angles(),
            jump_angles=phase_data['jump'].get_average_angles(),
            training_videos=video_paths,
            creation_date=datetime.now().isoformat()
        )

        # Save model to pickle file
        try:
            with open(ai_model_file_path, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            raise

        return model


class TripleJumpAnalyzer:
    """
    Class for analyzing athlete performance against trained models.
    Provides detailed comparison and recommendations.
    """

    def __init__(self, model_path: str):
        """
        Initialize the analyzer with a trained model.

        Args:
            model_path: Path to the pickle model file
        """
        self.detector = TripleJumpDetector()
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> TripleJumpModel:
        """Load the trained model from pickle file."""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            raise

    def analyze_performance(self, video_path: str,
                            output_dir: Optional[str] = None) -> List[
        PerformanceMetrics]:
        """
        Analyze athlete performance against the trained model.

        Args:
            video_path: Path to athlete's video
            output_dir: Optional directory to save analysis results

        Returns:
            List of PerformanceMetrics for each phase
        """

        # Process the input video
        results = self.detector.process_video(video_path, output_dir)

        # Extract phase-wise angles from the input video
        phase_angles = self._extract_input_phase_angles(results)

        # Compare with model and generate metrics
        performance_metrics = []

        for phase in ['hop', 'step', 'jump']:
            if phase in phase_angles:
                metrics = self._compare_phase_performance(
                    phase,
                    phase_angles[phase],
                    getattr(self.model, f"{phase}_angles")
                )
                performance_metrics.append(metrics)

        # Generate comprehensive report
        if output_dir:
            self._generate_report(performance_metrics, video_path, output_dir)

        return performance_metrics

    def _extract_input_phase_angles(self, results: Dict) -> Dict[
        str, Dict[str, float]]:
        """Extract average angles per phase from input video."""
        phases = results.get('phases', [])
        angles = results.get('angles', {})
        frame_count = results.get('frame_count', 0)

        phase_angles = {}

        if not phases or not angles:
            return phase_angles

        frames_per_phase = frame_count // max(len(phases), 1)

        for phase_idx, phase in enumerate(phases):
            start_frame = phase_idx * frames_per_phase
            end_frame = min((phase_idx + 1) * frames_per_phase, frame_count)

            phase_data = {
                'left_knee': [],
                'right_knee': [],
                'left_hip': [],
                'right_hip': []
            }

            for frame_idx in range(start_frame, end_frame):
                if frame_idx < len(angles.get('left_knee_angle', [])):
                    left_knee = angles['left_knee_angle'][frame_idx]
                    right_knee = angles['right_knee_angle'][frame_idx]
                    left_hip = angles.get('left_hip_angle', [None] * len(
                        angles['left_knee_angle']))[frame_idx]
                    right_hip = angles.get('right_hip_angle', [None] * len(
                        angles['left_knee_angle']))[frame_idx]

                    if left_knee is not None:
                        phase_data['left_knee'].append(left_knee)
                    if right_knee is not None:
                        phase_data['right_knee'].append(right_knee)
                    if left_hip is not None:
                        phase_data['left_hip'].append(left_hip)
                    if right_hip is not None:
                        phase_data['right_hip'].append(right_hip)

            # Calculate averages
            phase_angles[phase] = {
                'left_knee_avg': np.mean(phase_data['left_knee']) if
                phase_data['left_knee'] else 0.0,
                'right_knee_avg': np.mean(phase_data['right_knee']) if
                phase_data['right_knee'] else 0.0,
                'left_hip_avg': np.mean(phase_data['left_hip']) if phase_data[
                    'left_hip'] else 0.0,
                'right_hip_avg': np.mean(phase_data['right_hip']) if
                phase_data['right_hip'] else 0.0,
            }

        return phase_angles

    def _compare_phase_performance(self, phase: str,
                                   actual_angles: Dict[str, float],
                                   reference_angles: Dict[
                                       str, float]) -> PerformanceMetrics:
        """Compare actual performance with reference model for a specific phase."""
        differences = {}
        recommendations = []

        # Calculate differences
        for angle_type in ['left_knee_avg', 'right_knee_avg', 'left_hip_avg',
                           'right_hip_avg']:
            if angle_type in actual_angles and angle_type in reference_angles:
                diff = actual_angles[angle_type] - reference_angles[angle_type]
                differences[angle_type] = diff

                # Generate recommendations based on differences
                recommendations.extend(
                    self._generate_angle_recommendations(angle_type, diff,
                                                         phase))

        # Calculate performance score (0-100)
        performance_score = self._calculate_performance_score(differences)

        return PerformanceMetrics(
            phase=phase,
            reference_angles=reference_angles,
            actual_angles=actual_angles,
            differences=differences,
            recommendations=recommendations,
            performance_score=performance_score
        )

    def _generate_angle_recommendations(self, angle_type: str,
                                        difference: float, phase: str) -> List[
        str]:
        """Generate specific recommendations based on angle differences."""
        recommendations = []
        threshold = 10.0  # degrees

        if abs(difference) < threshold:
            return recommendations

        joint = "knee" if "knee" in angle_type else "hip"
        side = "left" if "left" in angle_type else "right"

        if joint == "knee":
            if difference > threshold:
                recommendations.append(
                    f"In {phase} phase: {side} knee is too extended (+{difference:.1f}°). "
                    f"Focus on maintaining better knee flexion for optimal power transfer."
                )
            elif difference < -threshold:
                recommendations.append(
                    f"In {phase} phase: {side} knee is over-flexed ({difference:.1f}°). "
                    f"Work on improving leg extension for better distance."
                )

        elif joint == "hip":
            if difference > threshold:
                recommendations.append(
                    f"In {phase} phase: {side} hip shows excessive extension (+{difference:.1f}°). "
                    f"Focus on hip flexor strength and mobility."
                )
            elif difference < -threshold:
                recommendations.append(
                    f"In {phase} phase: {side} hip is under-extended ({difference:.1f}°). "
                    f"Work on hip extension power for better takeoff."
                )

        return recommendations

    def _calculate_performance_score(self,
                                     differences: Dict[str, float]) -> float:
        """Calculate overall performance score (0-100) based on angle differences."""
        if not differences:
            return 50.0

        # Calculate average absolute difference
        avg_diff = np.mean([abs(diff) for diff in differences.values()])

        # Convert to score (lower difference = higher score)
        # Assume 20° difference gives 0 score, 0° difference gives 100 score
        max_diff = 20.0
        score = max(0, 100 - (avg_diff / max_diff) * 100)

        return round(score, 1)

    def _generate_report(self, metrics: List[PerformanceMetrics],
                         video_path: str, output_dir: str):
        """Generate comprehensive performance report."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        report_path = os.path.join(output_dir,
                                   f"performance_report_{video_name}.txt")

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TRIPLE JUMP PERFORMANCE ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Video Analyzed: {os.path.basename(video_path)}\n")
            f.write(
                f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Version: {self.model.version}\n")
            f.write(f"Model Created: {self.model.creation_date}\n\n")

            # Overall Performance Summary
            overall_score = np.mean([m.performance_score for m in metrics])
            f.write(f"OVERALL PERFORMANCE SCORE: {overall_score:.1f}/100\n\n")

            # Phase-by-phase analysis
            for metric in metrics:
                f.write(f"{metric.phase.upper()} PHASE ANALYSIS:\n")
                f.write("-" * 40 + "\n")
                f.write(
                    f"Performance Score: {metric.performance_score:.1f}/100\n\n")

                f.write("Angle Comparison:\n")
                for angle_type in ['left_knee_avg', 'right_knee_avg',
                                   'left_hip_avg', 'right_hip_avg']:
                    if angle_type in metric.actual_angles:
                        actual = metric.actual_angles[angle_type]
                        reference = metric.reference_angles.get(angle_type, 0)
                        diff = metric.differences.get(angle_type, 0)

                        f.write(f"  {angle_type.replace('_', ' ').title()}: "
                                f"{actual:.1f}° (Reference: {reference:.1f}°, "
                                f"Difference: {diff:+.1f}°)\n")

                f.write("\nRecommendations:\n")
                if metric.recommendations:
                    for i, rec in enumerate(metric.recommendations, 1):
                        f.write(f"  {i}. {rec}\n")
                else:
                    f.write(
                        "  • Excellent technique! Keep up the good work.\n")

                f.write("\n")

            # Training Recommendations
            f.write("SPECIFIC TRAINING RECOMMENDATIONS:\n")
            f.write("-" * 40 + "\n")

            all_recommendations = []
            for metric in metrics:
                all_recommendations.extend(metric.recommendations)

            if all_recommendations:
                unique_recommendations = list(set(all_recommendations))
                for i, rec in enumerate(unique_recommendations, 1):
                    f.write(f"{i}. {rec}\n")
            else:
                f.write("Your technique is excellent across all phases! "
                        "Focus on maintaining consistency and building power.\n")


def train_model_from_videos():
    """Train a new model from multiple training videos."""
    print("=== TRIPLE JUMP MODEL TRAINING ===")

    # Training video paths (add your training videos here)
    # Filter existing videos
    existing_videos = [v for v in DATASET_FILE_PATH_LIST if os.path.exists(v)]

    if len(existing_videos) < 2:
        print(
            f"Error: Need at least 2 training videos. Found: {len(existing_videos)}")
        print("Available videos:", existing_videos)
        return None

    print(f"Training with {len(existing_videos)} videos: {existing_videos}")

    # Initialize trainer and create model
    trainer = TripleJumpModelTrainer()

    try:
        model = trainer.create_model(existing_videos, AI_MODEL_FILE_PATH)
        print(f"\n✓ Model training completed successfully!")
        print(f"✓ Model saved to: {AI_MODEL_FILE_PATH}")
        print(f"✓ Training data: {len(existing_videos)} videos")
        return AI_MODEL_FILE_PATH

    except Exception as e:
        print(f"✗ Error during model training: {e}")
        return None


def analyze_performance_with_model(video_path: str, ai_model_file_path: str):
    """Analyze athlete performance using trained model."""
    print(f"\n=== PERFORMANCE ANALYSIS ===")
    print(f"Analyzing video: {video_path}")
    print(f"Using model: {ai_model_file_path}")

    try:
        # Initialize analyzer
        analyzer = TripleJumpAnalyzer(ai_model_file_path)

        # Analyze performance
        metrics = analyzer.analyze_performance(video_path, OUTPUT_DIR)

        # Print summary
        if metrics:
            overall_score = np.mean([m.performance_score for m in metrics])
            print(f"\n✓ Analysis completed!")
            print(f"✓ Overall Performance Score: {overall_score:.1f}/100")

            for metric in metrics:
                print(
                    f"✓ {metric.phase.title()} Phase Score: {metric.performance_score:.1f}/100")

            print(f"✓ Detailed report saved to output directory")
        else:
            print("✗ No performance metrics generated")

    except Exception as e:
        print(f"✗ Error during performance analysis: {e}")


def main():
    """Main function with enhanced functionality."""
    print("=" * 60)
    print("TRIPLE JUMP ANALYSIS SYSTEM - PRODUCTION VERSION")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check if target video exists
    if not os.path.exists(INPUT_VIDEO_FILE_PATH):
        print(f"Error: Target video not found: {INPUT_VIDEO_FILE_PATH}")
        return

    # Mode selection based on model availability
    if os.path.exists(AI_MODEL_FILE_PATH):
        print(f"✓ Found existing model: {AI_MODEL_FILE_PATH}")
        choice = input(
            "Choose mode:\n1. Use existing model for analysis\n2. "
            "Retrain model\nEnter choice (1 OR 2): ").strip()
    else:
        print("✗ No trained model found")
        choice = input(
            "Choose mode:\n1. Train new model\n2. Basic analysis only\nEnter choice (1-2): ").strip()
        if choice == "1":
            choice = "2"  # Map to retrain

    if choice == "1":
        # Use existing model for analysis
        analyze_performance_with_model(INPUT_VIDEO_FILE_PATH,
                                       AI_MODEL_FILE_PATH)

    elif choice == "2":
        # Train new model first, then analyze
        print("\nStep 1: Training new model...")
        trained_model_path = train_model_from_videos()

        if trained_model_path:
            print("\nStep 2: Analyzing performance with new model...")
            analyze_performance_with_model(INPUT_VIDEO_FILE_PATH,
                                           trained_model_path)
        else:
            print("✗ Model training failed. Falling back to basic analysis.")

    print(f"\n{'=' * 60}")
    print("Analysis completed! Check the output directory for results.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
