# VMDify - AI-Powered Motion Capture for MMD

> ⚠️ **UNDER ACTIVE DEVELOPMENT** ⚠️  
> This project is currently in development and may contain bugs or incomplete features. Contributions and testing are welcome!

**VMDify** is a cross-platform desktop application that converts **video or webcam input** into smooth **MikuMikuDance (.vmd)** motion files using modern AI pose estimation and motion refinement techniques.

## 🎯 What VMDify Does

**VMDify** transforms ordinary videos into professional-quality MMD animations:

- **📹 Input**: Video files or live webcam feed
- **🤖 AI Processing**: Advanced pose estimation with multiple AI models (MediaPipe, VIBE, HybrIK)
- **✨ Motion Refinement**: Reduces jitter, fixes foot-sliding, and produces smooth animations
- **🎭 Output**: High-quality `.vmd` motion files ready for MMD or Blender
- **🔄 Real-time Preview**: Live 3D preview of generated motion on PMX models

## 🚀 Key Features

- **Multi-AI Backend**: Supports MediaPipe, VIBE, and other state-of-the-art pose estimation models
- **GPU Acceleration**: CUDA/MPS support with CPU fallback
- **Advanced Refinement**: Outlier suppression, temporal smoothing, foot contact detection
- **Real-time 3D Preview**: See your motion applied to MMD models instantly
- **Cross-platform**: Windows, macOS, and Linux support
- **Professional Quality**: Motion refinement pipeline for smooth, natural animations

## 🛠️ Technology Stack

- **Frontend**: Electron + React + Three.js + MMDLoader
- **Backend**: Python + FastAPI + PyTorch
- **AI Models**: MediaPipe, VIBE, HybrIK-X for pose estimation
- **3D Graphics**: Three.js with MMD model support
- **Motion Processing**: Advanced filtering and IK-based refinement

## 🎬 Pipeline Overview

1. **Decode** → Extract frames from video/webcam
2. **2D Pose** → Detect human pose keypoints
3. **3D Lift** → Convert 2D poses to 3D motion
4. **Refine** → Apply smoothing, contact detection, and motion polish
5. **Retarget** → Map to MMD bone structure
6. **Export** → Generate `.vmd` files for MMD/Blender

## 🎯 Target Users

- **MMD Creators** who want motion capture from ordinary videos without expensive mocap suits
- **Blender Artists** who need quick `.vmd` animations from reference footage
- **Content Creators** looking to animate characters with real human motion

## 📋 Current Status

This project is **actively under development**. Current focus areas:

- ✅ Basic pose detection and 3D preview working
- 🚧 Improving animation smoothness and responsiveness
- 🚧 Enhanced PMX model loading and bone mapping
- 🚧 Multi-model AI integration and optimization
- 🔄 Real-time parameter tuning and quality controls

## 🤝 Contributing

This is my first time creating an .exe application, so I'm learning as I go! The project is publicly available on GitHub, so contributions, bug reports, and suggestions are very welcome.

**Note**: Please ensure environment variables and secrets are not exposed in contributions.

## 📚 References

- [OpenMMD](https://github.com/peterljq/OpenMMD) - Main reference for MMD integration
- [web-mmd](https://github.com/culdo/web-mmd) - 3D preview implementation reference

## 📄 License

MIT License - See LICENSE file for details

---
