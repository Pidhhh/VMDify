"""
VMDify AI Pipeline - Core processing pipeline for video to VMD conversion.

This module will contain the main AI pipeline:
1. Video Decoding
2. 2D Pose Detection
3. 3D Pose Lifting  
4. Motion Refinement
5. VMD Export

Currently: Basic structure and placeholder implementations.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable, Awaitable
import logging
import os
import json
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VMDifyPipeline:
    """Main pipeline for converting video to VMD motion files."""
    
    def __init__(self):
        self.is_initialized = False
        logger.info("VMDify pipeline initialized")
    
    async def process_video(self, video_path: str, job_id: str, on_progress: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None) -> Dict[str, Any]:
        """
        Process a video file through the complete AI pipeline.
        
        Args:
            video_path: Path to input video file
            job_id: Unique identifier for this job
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Starting video processing for job {job_id}")
        
        try:
            # Step 1: Decode video and extract frames
            frames, fps = await self._decode_video(video_path)
            logger.info(f"Decoded {len(frames)} frames from video @ {fps:.2f} fps")

            # Process per frame to enable live preview streaming
            refined_motion: List[Dict[str, Any]] = []
            alpha = 0.2  # EMA smoothing for root trajectory
            prev_rt = None
            # Track cumulative displacement from 2D hip-center to drive preview motion
            prev_cx: Optional[float] = None
            prev_cy: Optional[float] = None
            cum_x = 0.0
            cum_z = 0.0
            kx = 4.0  # horizontal scale (larger for visible movement)
            kz = 3.0  # vertical-to-forward scale

            # Build iterator for 2D pose (mediapipe with fallback)
            for pose in self._iter_2d_pose(frames):
                idx = pose['frame']
                # 3D lift (for future joints); not used for root in preview
                lifted = await self._lift_to_3d([pose])

                # Estimate simple body/head orientation from 2D keypoints
                body_yaw, body_pitch, head_yaw, head_pitch = self._estimate_orient(pose.get('keypoints', []))

                # Compute hip-center in normalized coords for displacement-based root motion
                kps = pose.get('keypoints', [])
                try:
                    lhip = kps[23]
                    rhip = kps[24]
                    cx = float((lhip[0] + rhip[0]) * 0.5)
                    cy = float((lhip[1] + rhip[1]) * 0.5)
                except Exception:
                    cx, cy = 0.5, 0.6

                if prev_cx is None:
                    prev_cx, prev_cy = cx, cy

                dx = cx - (prev_cx if prev_cx is not None else cx)
                dy = cy - (prev_cy if prev_cy is not None else cy)
                prev_cx, prev_cy = cx, cy

                # Integrate displacement; invert dy so moving down (toward feet) becomes forward +Z
                cum_x += dx * kx
                cum_z += (-dy) * kz

                rt_raw = [cum_x, 0.0, cum_z]
                if prev_rt is None:
                    prev_rt = rt_raw
                else:
                    prev_rt = [prev_rt[j] * (1 - alpha) + rt_raw[j] * alpha for j in range(3)]

                refined_motion.append({
                    'frame': idx,
                    'joints_3d': lifted[0].get('joints_3d', []),
                    'root_translation': prev_rt,
                    'orient': {
                        'body': {'yaw': body_yaw, 'pitch': body_pitch},
                        'head': {'yaw': head_yaw, 'pitch': head_pitch},
                    }
                })

                # Live preview callback with overlay image
                if on_progress:
                    try:
                        img_b64 = self._render_overlay(frames[idx], pose.get('keypoints', []))
                    except Exception as e:
                        img_b64 = None
                        logger.debug(f"overlay render failed: {e}")
                    # Optionally compact keypoints
                    kps = pose.get('keypoints', [])
                    payload = {
                        'type': 'frame',
                        'frame': idx,
                        'root_translation': prev_rt,
                        'image': img_b64,
                        'format': 'jpeg',
                        'pose2d': kps,
                        'orient': {
                            'body': {'yaw': body_yaw, 'pitch': body_pitch},
                            'head': {'yaw': head_yaw, 'pitch': head_pitch},
                        }
                    }
                    try:
                        await on_progress(payload)
                    except Exception as e:
                        logger.debug(f"on_progress error ignored: {e}")

            # Step 5: VMD Export (placeholder)
            vmd_path, debug_json_path, preview_traj = await self._export_vmd(refined_motion, job_id)
            logger.info(f"VMD file exported to: {vmd_path}")
            
            return {
                "status": "completed",
                "job_id": job_id,
                "total_frames": len(frames),
                "fps": fps,
                "output_path": vmd_path,
                "debug_json_path": debug_json_path,
                "preview_trajectory": preview_traj,
                "processing_steps": {
                    "decode": "✓",
                    "pose_2d": "✓", 
                    "pose_3d": "✓",
                    "refine": "✓",
                    "export": "✓"
                }
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed for job {job_id}: {str(e)}")
            return {
                "status": "failed",
                "job_id": job_id,
                "error": str(e)
            }
    
    async def _decode_video(self, video_path: str) -> Tuple[List[np.ndarray], float]:
        """Decode video into individual frames using OpenCV. Returns (frames, fps)."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        frames: List[np.ndarray] = []
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1e-3:
            fps = 30.0  # fallback
        max_frames = 600  # cap to avoid huge memory
        success, frame = cap.read()
        while success and len(frames) < max_frames:
            # Convert BGR to RGB for ML
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            success, frame = cap.read()
        cap.release()
        if not frames:
            raise RuntimeError("No frames decoded")
        return frames, fps
    
    async def _detect_2d_poses(self, frames: List[np.ndarray]) -> List[Dict]:
        """Detect 2D human poses using MediaPipe Pose if available; fallback to simple center track."""
        poses: List[Dict] = []
        try:
            from mediapipe.python.solutions import pose as mp_pose
            with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                for i, frame in enumerate(frames):
                    results = pose.process(frame)  # type: ignore[attr-defined]
                    keypoints = []
                    if getattr(results, 'pose_landmarks', None):  # type: ignore[truthy-function]
                        for lm in results.pose_landmarks.landmark:  # type: ignore[attr-defined]
                            keypoints.append([lm.x, lm.y, lm.visibility])
                    else:
                        h, w, _ = frame.shape
                        keypoints = [[0.5, 0.5, 0.0]] * 33
                    poses.append({
                        'frame': i,
                        'keypoints': keypoints
                    })
        except Exception as e:
            logger.warning(f"Mediapipe not available or failed ({e}); using center fallback")
            for i, frame in enumerate(frames):
                poses.append({
                    'frame': i,
                    'keypoints': [[0.5, 0.5, 0.0]] * 33
                })
        return poses
    
    async def _lift_to_3d(self, poses_2d: List[Dict]) -> List[Dict]:
        """Convert 2D mediapipe landmarks to a simple 3D representation and root translation."""
        poses_3d: List[Dict] = []
        for pose in poses_2d:
            kps = pose['keypoints']
            # Use left/right hip indices (23, 24) for root approx (mediapipe indices)
            try:
                lhip = kps[23]
                rhip = kps[24]
                cx = (lhip[0] + rhip[0]) / 2.0
                cy = (lhip[1] + rhip[1]) / 2.0
            except Exception:
                cx, cy = 0.5, 0.6
            # Map to a simple ground plane: x centered around 0, z forward
            root_x = (cx - 0.5) * 2.0  # [-1,1]
            root_z = (0.7 - cy) * 3.0  # forward based on vertical position
            root_y = 0.0
            # Fake joints_3d as zeros to keep payload compact
            joints_3d = [[0.0, 0.0, 0.0] for _ in range(24)]
            poses_3d.append({
                'frame': pose['frame'],
                'joints_3d': joints_3d,
                'root_translation': [root_x, root_y, root_z]
            })
        return poses_3d
    
    async def _refine_motion(self, poses_3d: List[Dict]) -> List[Dict]:
        """Apply basic EMA smoothing to root translation."""
        logger.info("Applying motion refinement...")
        alpha = 0.2
        smoothed = []
        prev = None
        for item in poses_3d:
            rt = item.get('root_translation', [0.0, 0.0, 0.0])
            if prev is None:
                prev = rt
            else:
                prev = [prev[j] * (1 - alpha) + rt[j] * alpha for j in range(3)]
            new_item = dict(item)
            new_item['root_translation'] = prev
            smoothed.append(new_item)
        return smoothed
    
    async def _export_vmd(self, motion_data: List[Dict], job_id: str) -> Tuple[str, str, list]:
        """
        Export motion data to VMD format and write debug JSON.
        Returns tuple of (vmd_path, debug_json_path, preview_trajectory)
        """
        # Resolve Outputs directory at project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
        outputs_dir = os.path.join(project_root, 'Outputs', job_id)
        os.makedirs(outputs_dir, exist_ok=True)

        # Placeholder VMD binary (not a valid VMD yet, but creates a file)
        vmd_path = os.path.join(outputs_dir, f"output_{job_id}.vmd")
        try:
            with open(vmd_path, 'wb') as f:
                # Write a small header to indicate placeholder
                f.write(b'VMDIFY_PLACEHOLDER')
        except Exception as e:
            logger.error(f"Failed writing VMD placeholder: {e}")

        # Write debug JSON with motion data
        debug_json_path = os.path.join(outputs_dir, f"debug_{job_id}.json")
        try:
            with open(debug_json_path, 'w', encoding='utf-8') as jf:
                json.dump({
                    'job_id': job_id,
                    'total_frames': len(motion_data),
                    'motion_data': motion_data[:200]  # cap for size
                }, jf, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed writing debug JSON: {e}")

        # Build a lightweight preview trajectory from root translations (or synthesize)
        preview_traj = []
        for i, frm in enumerate(motion_data):
            rt = frm.get('root_translation', [0.0, 0.0, 0.0])
            # Use refined root translation directly; no artificial drift/noise
            px = float(rt[0])
            py = float(rt[1])
            pz = float(rt[2])
            preview_traj.append({'frame': i, 'pos': [px, py, pz]})

        return vmd_path, debug_json_path, preview_traj

    def _render_overlay(self, frame_rgb: np.ndarray, keypoints: List[List[float]]) -> str:
        """Draw keypoints and connections on the RGB frame, downscale, and return base64 JPEG string."""
        # Convert to BGR for OpenCV
        img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        h, w, _ = img.shape
        # Downscale target width
        target_w = 480
        scale = target_w / float(w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
        # Mediapipe has 33 landmarks; define some common connections
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
        ]
        color = (245, 180, 80)
        for (a, b) in connections:
            if a < len(keypoints) and b < len(keypoints):
                ax, ay, _ = keypoints[a]
                bx, by, _ = keypoints[b]
                ax_i = int(ax * w * scale)
                ay_i = int(ay * h * scale)
                bx_i = int(bx * w * scale)
                by_i = int(by * h * scale)
                cv2.line(img, (ax_i, ay_i), (bx_i, by_i), color, 2, cv2.LINE_AA)
        # Draw small circles
        for i, kp in enumerate(keypoints):
            x, y, v = kp
            x_i = int(x * w * scale)
            y_i = int(y * h * scale)
            cv2.circle(img, (x_i, y_i), 2, (80, 200, 255), -1, cv2.LINE_AA)

        # Encode to JPEG and base64
        ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            return ''
        return base64.b64encode(buf.tobytes()).decode('ascii')

    def _estimate_orient(self, keypoints: List[List[float]]) -> Tuple[float, float, float, float]:
        """Compute rough body and head yaw/pitch from 2D keypoints in normalized coords.
        Returns (body_yaw, body_pitch, head_yaw, head_pitch) in radians.
        """
        try:
            import math
            # Mediapipe indices: 11(L shoulder), 12(R shoulder), 23(L hip), 24(R hip), 0(nose)
            if len(keypoints) < 25:
                return 0.0, 0.0, 0.0, 0.0
            ls, rs = keypoints[11], keypoints[12]
            lh, rh = keypoints[23], keypoints[24]
            nose = keypoints[0]
            # Body yaw from shoulder line horizontal direction
            dx = (rs[0] - ls[0])
            body_yaw = math.atan2(dx, 0.35)
            # Body pitch from shoulder vs hip vertical separation
            s_y = (ls[1] + rs[1]) * 0.5
            h_y = (lh[1] + rh[1]) * 0.5
            dy = (h_y - s_y)
            body_pitch = math.atan2(dy, 0.5)
            # Head yaw/pitch from nose relative to shoulder center
            cx = (ls[0] + rs[0]) * 0.5
            cy = (ls[1] + rs[1]) * 0.5
            head_dx = (nose[0] - cx)
            head_dy = (nose[1] - cy)
            head_yaw = math.atan2(head_dx, 0.2)
            head_pitch = -math.atan2(head_dy, 0.25)
            # Clamp small angles to avoid noise
            def clamp(a: float, lim: float) -> float:
                if a > lim: return lim
                if a < -lim: return -lim
                return a
            body_yaw = clamp(body_yaw, 0.6)
            body_pitch = clamp(body_pitch, 0.5)
            head_yaw = clamp(head_yaw, 0.8)
            head_pitch = clamp(head_pitch, 0.6)
            return body_yaw, body_pitch, head_yaw, head_pitch
        except Exception:
            return 0.0, 0.0, 0.0, 0.0

    def _iter_2d_pose(self, frames: List[np.ndarray]):
        """Yield per-frame 2D keypoints using MediaPipe when available, else center fallback."""
        try:
            from mediapipe.python.solutions import pose as mp_pose
            with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                for i, frame in enumerate(frames):
                    results = pose.process(frame)  # type: ignore[attr-defined]
                    keypoints = []
                    if getattr(results, 'pose_landmarks', None):  # type: ignore[truthy-function]
                        for lm in results.pose_landmarks.landmark:  # type: ignore[attr-defined]
                            keypoints.append([lm.x, lm.y, lm.visibility])
                    else:
                        keypoints = [[0.5, 0.6, 0.0]] * 33
                    yield {'frame': i, 'keypoints': keypoints}
        except Exception as e:
            logger.warning(f"Mediapipe not available or failed ({e}); using center fallback")
            for i, _ in enumerate(frames):
                yield {'frame': i, 'keypoints': [[0.5, 0.6, 0.0]] * 33}

# Global pipeline instance
pipeline = VMDifyPipeline()
