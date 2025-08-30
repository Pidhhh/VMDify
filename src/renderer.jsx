import './index.css';
import React, { useState, useCallback } from 'react';
import ReactDOM from 'react-dom/client';
import MMDPreview from './components/MMDPreview';
import BackendConnection from './components/BackendConnection';
import VideoInput from './components/VideoInput';
import VideoPreview from './components/VideoPreview';
import ModelPicker from './components/ModelPicker';

const App = () => {
    const [selectedVideo, setSelectedVideo] = useState(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [progressData, setProgressData] = useState(null);
    const [trajectory, setTrajectory] = useState(null);
    const [orient, setOrient] = useState(null);
    const [pose2d, setPose2d] = useState(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [fps, setFps] = useState(30);
    const [latestFrameB64, setLatestFrameB64] = useState(null);
    const [customModel, setCustomModel] = useState(null); // { modelUrl, urlMap, displayName }
    
    // PMX model path (default or user-chosen)
    const modelPath = (customModel && customModel.modelUrl) ? customModel.modelUrl : '/models/Second%20Test/PMX/MMD_STD.pmx';

    const handleVideoSelected = (file) => {
        console.log('Video selected:', file.name);
        setSelectedVideo(file);
    };

    const handleProcessingStart = () => {
        setIsProcessing(true);
        // Reset previous state for a clean new run
        setTrajectory([]);
        setOrient(null);
        setLatestFrameB64(null);
        setIsPlaying(false);
    };

    const handleProcessingComplete = useCallback((result) => {
        setIsProcessing(false);
        if (result && result.pipeline_result) {
            const traj = result.pipeline_result.preview_trajectory;
            setTrajectory(traj || null);
            if (typeof result.pipeline_result.fps === 'number') setFps(result.pipeline_result.fps);
            setProgressData(result.pipeline_result);
        }
    }, []);

    const handleJobComplete = useCallback((payload) => {
        // From websocket: job_complete
        if (!payload) return;
    setTrajectory(payload.preview_trajectory || null);
    if (typeof payload.fps === 'number') setFps(payload.fps);
    setIsPlaying(true);
    }, []);

    const handleFrame = useCallback((frameMsg) => {
        if (!frameMsg || typeof frameMsg.frame !== 'number') return;
        const rt = frameMsg.root_translation || [0,0,0];
        if (frameMsg.image) {
            setLatestFrameB64(frameMsg.image);
            if (!isPlaying) setIsPlaying(true);
        }
        setTrajectory(prev => {
            const next = Array.isArray(prev) ? [...prev] : [];
            const f = frameMsg.frame;
            // Append or fill sequentially; ignore wildly out-of-order frames
            if (f <= next.length) {
                next[f] = { frame: f, pos: [Number(rt[0])||0, Number(rt[1])||0, Number(rt[2])||0] };
            } else if (f === next.length) {
                next.push({ frame: f, pos: [Number(rt[0])||0, Number(rt[1])||0, Number(rt[2])||0] });
            }
            return next;
        });
        if (frameMsg.orient) {
            setOrient(frameMsg.orient);
        }
        if (frameMsg.pose2d) {
            setPose2d(frameMsg.pose2d);
        }
    }, [isPlaying]);

    return (
        <div className="min-h-screen bg-dark text-white flex flex-col">
            {/* Header */}
            <header className="bg-dark-lighter border-b border-gray-700 px-6 py-4 fade-in">
                <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 rounded-lg flex items-center justify-center" 
                             style={{background: 'linear-gradient(to bottom right, var(--primary), var(--secondary))'}}>
                            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24" style={{color: 'var(--dark)'}}>
                                <path d="M12 2L2 7v10c0 5.55 3.84 9.74 9 11 5.16-1.26 9-5.45 9-11V7l-10-5z"/>
                            </svg>
                        </div>
                        <div>
                            <h1 className="text-2xl font-bold bg-gradient-to-r">
                                VMDify
                            </h1>
                            <p className="text-gray-400 text-sm">AI-powered Motion Capture</p>
                        </div>
                    </div>
                    <div className="flex items-center space-x-2">
                        <div className={`w-2 h-2 rounded-full ${isProcessing ? 'bg-yellow-500 pulse' : 'bg-green-500'}`}></div>
                        <span className="text-sm text-gray-400">
                            {isProcessing ? 'Processing' : 'Ready'}
                        </span>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <div className="flex-1 flex">
                {/* Left Panel - 3D Preview */}
                <div className="flex-1 bg-dark p-6 flex flex-col">
                    <div className="card h-full flex flex-col fade-in">
                                                <div className="p-4 border-b border-gray-700 flex items-center justify-between">
                            <h2 className="text-lg font-semibold text-primary flex items-center space-x-2">
                                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                                </svg>
                                <span>3D Model Preview</span>
                            </h2>
                                        <div className="flex items-center space-x-2">
                                                                <button
                                                                    onClick={() => setIsPlaying(!isPlaying)}
                                                                    disabled={!Array.isArray(trajectory) || trajectory.length === 0}
                                                                    className={`btn py-2 px-3 text-sm flex items-center space-x-1 ${Array.isArray(trajectory) && trajectory.length > 0 ? 'btn-primary' : 'btn-disabled'}`}
                                                                >
                                                                    {isPlaying ? (
                                                                        <>
                                                                            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                                                                                <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/>
                                                                            </svg>
                                                                            <span>Pause</span>
                                                                        </>
                                                                    ) : (
                                                                        <>
                                                                            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                                                                                <path d="M8 5v14l11-7z"/>
                                                                            </svg>
                                                                            <span>Play</span>
                                                                        </>
                                                                    )}
                                                                </button>
                                                                <button
                                                                    onClick={() => {setIsPlaying(false); /* Reset to start */}}
                                                                    disabled={!Array.isArray(trajectory) || trajectory.length === 0}
                                                                    className={`btn py-2 px-3 text-sm ${Array.isArray(trajectory) && trajectory.length > 0 ? 'btn-secondary' : 'btn-disabled'}`}
                                                                >
                                                                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                                                                        <path d="M6 6h2v12H6zm3.5 6l8.5 6V6z"/>
                                                                    </svg>
                                                                </button>
                                                        </div>
                        </div>
                        <div className="flex-1">
                            <MMDPreview modelPath={modelPath} motionData={progressData} trajectory={trajectory} isPlaying={isPlaying} orient={orient} fps={fps} pose2d={pose2d} urlMap={customModel?.urlMap} />
                        </div>
                    </div>
                </div>

                {/* Right Panel - Controls */}
                <div className="w-96 bg-dark-lighter border-l border-gray-700 flex flex-col">
                    {/* Video Input Section */}
                    <div className="p-6 border-b border-gray-700 fade-in">
                        <VideoInput 
                            onVideoSelected={handleVideoSelected}
                            onProcessingStart={handleProcessingStart}
                            onProcessingComplete={handleProcessingComplete}
                            selectedVideo={selectedVideo}
                            isProcessing={isProcessing}
                        />
                    </div>

                                        {/* Enhanced Model Picker */}
                                        <div className="p-6 border-b border-gray-700 fade-in">
                                                <h3 className="text-primary font-semibold mb-3 flex items-center space-x-2">
                                                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                                                        <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                                                    </svg>
                                                    <span>3D Model</span>
                                                </h3>
                                                <ModelPicker onModelLoaded={setCustomModel} />
                                                {customModel && (
                                                    <div className="mt-3 p-3 bg-dark rounded-lg space-y-2">
                                                        <div className="text-sm font-medium text-green-400 flex items-center space-x-1">
                                                            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                                                                <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                                                            </svg>
                                                            <span>Model Loaded</span>
                                                        </div>
                                                        <div className="text-xs text-gray-300 truncate">
                                                            <strong>File:</strong> {customModel.displayName}
                                                        </div>
                                                        {customModel.fileCount && (
                                                            <div className="text-xs text-gray-400">
                                                                <strong>Files:</strong> {customModel.fileCount} total
                                                                {customModel.textureCount > 0 && ` (${customModel.textureCount} textures)`}
                                                            </div>
                                                        )}
                                                        <button 
                                                            onClick={() => setCustomModel(null)}
                                                            className="text-xs text-gray-500 hover:text-red-400 transition-colors"
                                                        >
                                                            Clear Model
                                                        </button>
                                                    </div>
                                                )}
                                        </div>

                    {/* Live Video/Dets Preview */}
                    <div className="p-6 border-b border-gray-700 fade-in">
                        <h3 className="text-primary font-semibold mb-3">Preview: Video + Detections</h3>
                        <VideoPreview frameB64={latestFrameB64} />
                    </div>

                    {/* Backend Connection */}
                    <div className="p-6 flex-1 overflow-y-auto fade-in">
                        <BackendConnection onJobComplete={handleJobComplete} onFrame={handleFrame} />
                    </div>

                    {/* Bottom Actions */}
                    <div className="p-6 border-t border-gray-700 space-y-3 fade-in">
                        <button className="w-full btn btn-primary py-3 px-4 space-x-2" disabled={!trajectory}>
                            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                            </svg>
                            <span>Export VMD</span>
                        </button>
                        <div className="flex space-x-2">
                            <button className="flex-1 btn btn-secondary py-2 px-3 text-sm">
                                Settings
                            </button>
                            <button className="flex-1 btn btn-secondary py-2 px-3 text-sm">
                                Help
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
