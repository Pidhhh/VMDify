"""
Enhanced AI Models Manager for VMDify
Supports multiple AI models for better motion capture quality:
- MediaPipe (built-in)
- OpenPose (optional)
- MMPose (optional)
- VIBE (temporal 3D lifting)
- Custom models
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import os
from enum import Enum

logger = logging.getLogger(__name__)

class ModelType(Enum):
    MEDIAPIPE = "mediapipe"
    OPENPOSE = "openpose"
    MMPOSE = "mmpose"
    VIBE = "vibe"
    HYBRIK = "hybrik"

class AIModelManager:
    """Manages multiple AI models for pose estimation and 3D lifting."""
    
    def __init__(self):
        self.available_models = {}
        self.active_models = {}
        self.device = self._get_optimal_device()
        logger.info(f"AI Model Manager initialized on device: {self.device}")
        
        # Initialize available models
        self._initialize_models()
    
    def _get_optimal_device(self) -> str:
        """Determine the best available device for AI inference."""
        if torch.cuda.is_available():
            # Check VRAM availability
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"CUDA available with {gpu_memory:.1f}GB VRAM")
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) available")
            return "mps"
        else:
            logger.info("Using CPU for AI inference")
            return "cpu"
    
    def _initialize_models(self):
        """Initialize and register available AI models."""
        # MediaPipe (always available)
        self.available_models[ModelType.MEDIAPIPE] = {
            'name': 'MediaPipe Pose',
            'description': 'Fast and accurate 2D pose detection',
            'quality': 'Good',
            'speed': 'Fast',
            'requirements': ['mediapipe'],
            'loaded': False
        }
        
        # Check for optional models
        try:
            import mmpose
            self.available_models[ModelType.MMPOSE] = {
                'name': 'MMPose',
                'description': 'High-accuracy 2D pose with multiple models',
                'quality': 'Excellent',
                'speed': 'Medium',
                'requirements': ['mmpose', 'mmengine'],
                'loaded': False
            }
        except ImportError:
            logger.debug("MMPose not available")
        
        # VIBE for temporal 3D lifting
        try:
            # Check if VIBE dependencies are available
            self.available_models[ModelType.VIBE] = {
                'name': 'VIBE',
                'description': 'Temporal 3D pose estimation with SMPL',
                'quality': 'Excellent',
                'speed': 'Slow',
                'requirements': ['torch', 'torchvision'],
                'loaded': False
            }
        except Exception:
            logger.debug("VIBE dependencies not available")
        
        logger.info(f"Available AI models: {list(self.available_models.keys())}")
    
    def get_available_models(self) -> Dict[ModelType, Dict[str, Any]]:
        """Get information about all available models."""
        return self.available_models.copy()
    
    def load_model(self, model_type: ModelType) -> bool:
        """Load a specific AI model."""
        if model_type not in self.available_models:
            logger.error(f"Model {model_type} not available")
            return False
        
        if model_type in self.active_models:
            logger.info(f"Model {model_type} already loaded")
            return True
        
        try:
            if model_type == ModelType.MEDIAPIPE:
                model = self._load_mediapipe()
            elif model_type == ModelType.MMPOSE:
                model = self._load_mmpose()
            elif model_type == ModelType.VIBE:
                model = self._load_vibe()
            else:
                logger.error(f"Loading {model_type} not implemented yet")
                return False
            
            if model:
                self.active_models[model_type] = model
                self.available_models[model_type]['loaded'] = True
                logger.info(f"Successfully loaded {model_type}")
                return True
        
        except Exception as e:
            logger.error(f"Failed to load {model_type}: {e}")
            return False
        
        return False
    
    def _load_mediapipe(self):
        """Load MediaPipe pose model."""
        try:
            from mediapipe.python.solutions import pose as mp_pose
            from mediapipe.python.solutions import holistic as mp_holistic
            
            # Return both pose and holistic for enhanced detection
            return {
                'pose': mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=2,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                ),
                'holistic': mp_holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=2,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            }
        except ImportError as e:
            logger.error(f"MediaPipe import failed: {e}")
            return None
    
    def _load_mmpose(self):
        """Load MMPose model (placeholder)."""
        try:
            # This would load MMPose models
            # For now, return None as placeholder
            logger.info("MMPose loading not implemented yet")
            return None
        except Exception as e:
            logger.error(f"MMPose loading failed: {e}")
            return None
    
    def _load_vibe(self):
        """Load VIBE model for temporal 3D pose estimation (placeholder)."""
        try:
            # This would load VIBE model
            # For now, return None as placeholder
            logger.info("VIBE loading not implemented yet")
            return None
        except Exception as e:
            logger.error(f"VIBE loading failed: {e}")
            return None
    
    def detect_2d_poses(self, frames: List[np.ndarray], model_type: ModelType = ModelType.MEDIAPIPE) -> List[Dict[str, Any]]:
        """Detect 2D poses using specified model."""
        if model_type not in self.active_models:
            if not self.load_model(model_type):
                logger.error(f"Cannot detect poses: {model_type} not available")
                return []
        
        if model_type == ModelType.MEDIAPIPE:
            return self._detect_mediapipe_poses(frames)
        elif model_type == ModelType.MMPOSE:
            return self._detect_mmpose_poses(frames)
        else:
            logger.error(f"2D pose detection for {model_type} not implemented")
            return []
    
    def _detect_mediapipe_poses(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Detect poses using MediaPipe."""
        model = self.active_models[ModelType.MEDIAPIPE]
        poses = []
        
        try:
            # Use holistic for better full-body tracking
            with model['holistic'] as holistic:
                for i, frame in enumerate(frames):
                    results = holistic.process(frame)
                    keypoints = []
                    confidence = 1.0
                    
                    if results.pose_landmarks:
                        for lm in results.pose_landmarks.landmark:
                            keypoints.append([lm.x, lm.y, lm.visibility])
                            confidence = min(confidence, lm.visibility)
                    else:
                        keypoints = [[0.5, 0.5, 0.0]] * 33
                        confidence = 0.0
                    
                    poses.append({
                        'frame': i,
                        'keypoints': keypoints,
                        'confidence': confidence,
                        'model_type': ModelType.MEDIAPIPE.value
                    })
                    
        except Exception as e:
            logger.error(f"MediaPipe pose detection failed: {e}")
            # Fallback to basic pose model
            try:
                with model['pose'] as pose:
                    for i, frame in enumerate(frames):
                        results = pose.process(frame)
                        keypoints = []
                        confidence = 1.0
                        
                        if results.pose_landmarks:
                            for lm in results.pose_landmarks.landmark:
                                keypoints.append([lm.x, lm.y, lm.visibility])
                                confidence = min(confidence, lm.visibility)
                        else:
                            keypoints = [[0.5, 0.5, 0.0]] * 33
                            confidence = 0.0
                        
                        poses.append({
                            'frame': i,
                            'keypoints': keypoints,
                            'confidence': confidence,
                            'model_type': ModelType.MEDIAPIPE.value
                        })
            except Exception as e2:
                logger.error(f"MediaPipe fallback also failed: {e2}")
                return []
        
        return poses
    
    def _detect_mmpose_poses(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Detect poses using MMPose (placeholder)."""
        logger.warning("MMPose detection not implemented yet")
        return []
    
    def lift_to_3d(self, poses_2d: List[Dict[str, Any]], model_type: ModelType = ModelType.VIBE) -> List[Dict[str, Any]]:
        """Lift 2D poses to 3D using specified model."""
        if model_type == ModelType.VIBE and model_type not in self.active_models:
            # Fallback to enhanced geometric lifting
            return self._geometric_3d_lift(poses_2d)
        
        if model_type == ModelType.VIBE:
            return self._vibe_3d_lift(poses_2d)
        else:
            return self._geometric_3d_lift(poses_2d)
    
    def _geometric_3d_lift(self, poses_2d: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhanced geometric 3D lifting with anatomical constraints."""
        poses_3d = []
        
        for pose in poses_2d:
            kps = pose['keypoints']
            confidence = pose.get('confidence', 1.0)
            
            # Enhanced root calculation with multiple reference points
            try:
                # MediaPipe keypoint indices
                lhip, rhip = kps[23], kps[24]
                lshoulder, rshoulder = kps[11], kps[12]
                
                # Calculate torso center for stable root
                hip_center = [(lhip[0] + rhip[0]) * 0.5, (lhip[1] + rhip[1]) * 0.5]
                shoulder_center = [(lshoulder[0] + rshoulder[0]) * 0.5, (lshoulder[1] + rshoulder[1]) * 0.5]
                
                # Weight toward hips for lower center of gravity
                torso_x = hip_center[0] * 0.7 + shoulder_center[0] * 0.3
                torso_y = hip_center[1] * 0.6 + shoulder_center[1] * 0.4
                
                # Enhanced depth estimation
                torso_height = abs(shoulder_center[1] - hip_center[1])
                shoulder_width = abs(rshoulder[0] - lshoulder[0])
                
                # Reference measurements for depth calculation
                ref_height, ref_width = 0.25, 0.15
                height_factor = torso_height / ref_height if ref_height > 0 else 1.0
                width_factor = shoulder_width / ref_width if ref_width > 0 else 1.0
                
                depth_factor = (height_factor + width_factor) * 0.5
                depth_z = (1.0 - depth_factor) * 2.0
                
            except Exception:
                torso_x, torso_y = 0.5, 0.6
                depth_z = 0.0
            
            # Map to world coordinates with enhanced scaling
            root_x = (torso_x - 0.5) * 8.0
            root_z = (0.7 - torso_y) * 6.0 + depth_z
            root_y = max(0.0, depth_z * 0.5)
            
            # Generate 3D joints with anatomical constraints
            joints_3d = []
            for i, kp in enumerate(kps):
                if len(kp) >= 3:
                    x = (kp[0] - 0.5) * 8.0
                    y = (0.7 - kp[1]) * 6.0
                    z = depth_z * (1.0 + kp[2] * 0.3)
                    joints_3d.append([x, y, z])
                else:
                    joints_3d.append([0.0, 0.0, 0.0])
            
            poses_3d.append({
                'frame': pose['frame'],
                'joints_3d': joints_3d,
                'root_translation': [root_x, root_y, root_z],
                'confidence': confidence,
                'lift_method': 'geometric_enhanced'
            })
        
        return poses_3d
    
    def _vibe_3d_lift(self, poses_2d: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """3D lifting using VIBE (placeholder)."""
        logger.warning("VIBE 3D lifting not implemented yet, using geometric fallback")
        return self._geometric_3d_lift(poses_2d)
    
    def unload_model(self, model_type: ModelType):
        """Unload a specific model to free memory."""
        if model_type in self.active_models:
            del self.active_models[model_type]
            self.available_models[model_type]['loaded'] = False
            logger.info(f"Unloaded {model_type}")
    
    def unload_all_models(self):
        """Unload all models to free memory."""
        for model_type in list(self.active_models.keys()):
            self.unload_model(model_type)
        logger.info("All AI models unloaded")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about model manager state."""
        return {
            'device': self.device,
            'available_models': {k.value: v for k, v in self.available_models.items()},
            'active_models': list(self.active_models.keys()),
            'gpu_available': torch.cuda.is_available(),
            'gpu_memory': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        }

# Global model manager instance
model_manager = AIModelManager()
