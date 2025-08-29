/**
 * This file will automatically be loaded by vite and run in the "renderer" context.
 * To learn more about the differences between the "main" and the "renderer" context in
 * Electron, visit:
 *
 * https://electronjs.org/docs/tutorial/process-model
 *
 * By default, Node.js integration in this file is disabled. When enabling Node.js integration
 * in a renderer process, please be aware of potential security implications. You can read
 * more about security risks here:
 *
 * https://electronjs.org/docs/tutorial/security
 *
 * To enable Node.js integration in this file, open up `main.js` and enable the `nodeIntegration`
 * flag:
 *
 * ```
 *  // Create the browser window.
 *  mainWindow = new BrowserWindow({
 *    width: 800,
 *    height: 600,
 *    webPreferences: {
 *      nodeIntegration: true
 *    }
 *  });
 * ```
 */

import './index.css';
import React from 'react';
import ReactDOM from 'react-dom/client';
import MMDPreview from './components/MMDPreview';
import BackendConnection from './components/BackendConnection';
import VideoInput from './components/VideoInput';

const App = () => {
    // Using your specific PMX model
    const modelPath = './models/SakamataAlter.pmx';

    const handleVideoSelected = (file) => {
        console.log('Video selected:', file.name);
        // TODO: This will trigger the AI pipeline
    };

    return (
        <div className="app-container">
            <h1>VMDify - AI Motion Capture</h1>
            
            <div className="main-content">
                <div className="preview-section">
                    <h2>3D Model Preview</h2>
                    <MMDPreview modelPath={modelPath} />
                </div>
                
                <div className="control-section">
                    <VideoInput onVideoSelected={handleVideoSelected} />
                    <BackendConnection />
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
