import React, { useState, useRef } from 'react';

const VideoInput = ({ onVideoSelected, onProcessingStart, onProcessingComplete, selectedVideo, isProcessing }) => {
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('video/')) {
      onVideoSelected(file);
    } else {
      alert('Please select a valid video file');
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (event) => {
    event.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragOver(false);
    const files = event.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (file.type.startsWith('video/')) {
        onVideoSelected(file);
      } else {
        alert('Please drop a valid video file');
      }
    }
  };

  const processVideo = async () => {
    if (!selectedVideo) return;

    onProcessingStart();
    
    try {
      // Send video to Python backend for processing
      const formData = new FormData();
      formData.append('video', selectedVideo);
      
      const response = await fetch('http://127.0.0.1:8000/process_video', {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('Video processing result:', result);
        onProcessingComplete(result);
      } else {
        console.error('Failed to process video');
      }
    } catch (error) {
      console.error('Error processing video:', error);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const getDropZoneClass = () => {
    if (dragOver) return 'drop-zone drag-over';
    if (selectedVideo) return 'drop-zone has-file';
    return 'drop-zone';
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center space-x-2" style={{marginBottom: '1rem'}}>
        <svg className="w-5 h-5 text-primary" fill="currentColor" viewBox="0 0 24 24">
          <path d="M8 5v14l11-7z"/>
        </svg>
        <h3 className="text-primary font-semibold">Video Input</h3>
      </div>
      
      {/* File Drop Zone */}
      <div
        className={getDropZoneClass()}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={handleFileSelect}
          className="hidden"
        />
        
        <div className="space-y-3">
          {selectedVideo ? (
            <>
              <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center" style={{margin: '0 auto'}}>
                <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>
              </div>
              <div>
                <p className="text-green-400 font-medium">
                  {selectedVideo.name}
                </p>
                <p className="text-gray-400 text-sm">
                  {formatFileSize(selectedVideo.size)}
                </p>
              </div>
            </>
          ) : (
            <>
              <div className="w-12 h-12 rounded-full flex items-center justify-center transition-colors" 
                   style={{
                     margin: '0 auto',
                     backgroundColor: dragOver ? 'var(--primary)' : 'var(--gray-600)',
                     color: dragOver ? 'var(--dark)' : 'var(--gray-300)'
                   }}>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <div>
                <p className="text-gray-300 font-medium">
                  {dragOver ? 'Drop your video here' : 'Drop video file or click to browse'}
                </p>
                <p className="text-gray-500 text-sm">
                  Supported: MP4, AVI, MOV, WMV (max 500MB)
                </p>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Video Info Panel */}
      {selectedVideo && (
        <div className="card-dark p-4 space-y-3">
          <h4 className="text-sm font-medium text-gray-300">Video Details</h4>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <span className="text-gray-400">File:</span>
              <span className="truncate" style={{marginLeft: '0.5rem', color: 'white'}}>{selectedVideo.name}</span>
            </div>
            <div>
              <span className="text-gray-400">Size:</span>
              <span style={{marginLeft: '0.5rem', color: 'white'}}>{formatFileSize(selectedVideo.size)}</span>
            </div>
            <div>
              <span className="text-gray-400">Type:</span>
              <span style={{marginLeft: '0.5rem', color: 'white'}}>{selectedVideo.type}</span>
            </div>
            <div>
              <span className="text-gray-400">Modified:</span>
              <span style={{marginLeft: '0.5rem', color: 'white'}}>{new Date(selectedVideo.lastModified).toLocaleDateString()}</span>
            </div>
          </div>
        </div>
      )}

      {/* Process Button */}
      <button
        onClick={processVideo}
        disabled={!selectedVideo || isProcessing}
        className={`w-full py-3 px-4 rounded-lg font-semibold transition-all duration-200 flex items-center justify-center space-x-2 ${
          !selectedVideo || isProcessing ? 'btn-disabled' : 'btn btn-primary hover:scale-105 hover:shadow-lg'
        }`}
      >
        {isProcessing ? (
          <>
            <div className="w-4 h-4 animate-spin" style={{
              border: '2px solid var(--dark)',
              borderTop: '2px solid transparent',
              borderRadius: '50%'
            }}></div>
            <span>Processing...</span>
          </>
        ) : (
          <>
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
              <path d="M13 10V3L4 14h7v7l9-11h-7z"/>
            </svg>
            <span>Start Motion Capture</span>
          </>
        )}
      </button>

      {/* Quick Actions */}
      {selectedVideo && !isProcessing && (
        <div className="flex space-x-2">
          <button 
            onClick={() => onVideoSelected(null)}
            className="flex-1 btn btn-secondary py-2 px-3 text-sm"
          >
            Clear
          </button>
        </div>
      )}
    </div>
  );
};

export default VideoInput;
