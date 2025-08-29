import './index.css';
import React, { useState, useCallback } from 'react';
import ReactDOM from 'react-dom/client';
import MMDPreview from './components/MMDPreview';
import BackendConnection from './components/BackendConnection';
import VideoInput from './components/VideoInput';

const App = () => {
    const [selectedVideo, setSelectedVideo] = useState(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [progressData, setProgressData] = useState(null);
    const [trajectory, setTrajectory] = useState(null);
    const [isPlaying, setIsPlaying] = useState(false);
    
    // Using your specific PMX model
    const modelPath = '/models/SakamataAlter.pmx';

    const handleVideoSelected = (file) => {
        console.log('Video selected:', file.name);
        setSelectedVideo(file);
    };

    const handleProcessingStart = () => {
        setIsProcessing(true);
    };

    const handleProcessingComplete = useCallback((result) => {
        setIsProcessing(false);
        if (result && result.pipeline_result) {
            const traj = result.pipeline_result.preview_trajectory;
            setTrajectory(traj || null);
            setProgressData(result.pipeline_result);
        }
    }, []);

    const handleJobComplete = useCallback((payload) => {
        // From websocket: job_complete
        if (!payload) return;
        setTrajectory(payload.preview_trajectory || null);
        setIsPlaying(true);
    }, []);

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
                                                                    onClick={() => setIsPlaying(true)}
                                                                    disabled={!trajectory}
                                                                    className={`btn py-2 px-3 text-sm ${trajectory ? 'btn-primary' : 'btn-disabled'}`}
                                                                >
                                                                    Start
                                                                </button>
                                                                <button
                                                                    onClick={() => setIsPlaying(false)}
                                                                    className="btn btn-secondary py-2 px-3 text-sm"
                                                                >
                                                                    Stop
                                                                </button>
                                                        </div>
                        </div>
                        <div className="flex-1">
                                                        <MMDPreview modelPath={modelPath} motionData={progressData} trajectory={trajectory} isPlaying={isPlaying} />
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

                    {/* Progress Section */}
                    {isProcessing && (
                        <div className="p-6 border-b border-gray-700 fade-in">
                            <div className="space-y-4">
                                <div className="flex items-center justify-between">
                                    <h3 className="text-primary font-semibold">Processing Pipeline</h3>
                                    <div className="text-xs text-gray-400">Stage 2/6</div>
                                </div>
                                <div className="space-y-2">
                                    <div className="flex" style={{justifyContent: 'space-between', fontSize: '0.875rem'}}>
                                        <span>2D Pose Detection</span>
                                        <span className="text-primary">45%</span>
                                    </div>
                                    <div className="progress-bar">
                                        <div className="progress-fill" style={{width: '45%'}}></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Backend Connection */}
                    <div className="p-6 flex-1 overflow-y-auto fade-in">
                        <BackendConnection onJobComplete={handleJobComplete} />
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
