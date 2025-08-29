import React, { useRef, useState } from 'react';

// Enhanced PMX model picker with better file handling and preview
const ModelPicker = ({ onModelLoaded }) => {
  const inputRef = useRef(null);
  const folderInputRef = useRef(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState('');

  const handleFileChange = async (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    
    setIsLoading(true);
    setLoadingStatus('Processing files...');
    
    try {
      await processFiles(files);
    } catch (error) {
      console.error('Error processing files:', error);
      alert('Error loading model files. Please check the files and try again.');
    } finally {
      setIsLoading(false);
      setLoadingStatus('');
    }
  };

  const handleFolderChange = async (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    
    setIsLoading(true);
    setLoadingStatus('Processing folder...');
    
    try {
      await processFiles(files);
    } catch (error) {
      console.error('Error processing folder:', error);
      alert('Error loading model folder. Please check the folder contents and try again.');
    } finally {
      setIsLoading(false);
      setLoadingStatus('');
    }
  };

  const processFiles = async (files) => {
    // Create maps for different file types
    const map = new Map();
    const textureFiles = new Map();
    let pmxFile = null;
    
    setLoadingStatus('Analyzing files...');
    
    // Process all files
    for (const f of files) {
      const nameLower = f.name.toLowerCase();
      const url = URL.createObjectURL(f);
      
      // Store all files in the main map
      map.set(nameLower, url);
      map.set(f.name, url); // Also store with original case
      
      // Identify file types
      if (nameLower.endsWith('.pmx') && !pmxFile) {
        pmxFile = f;
      } else if (nameLower.match(/\.(png|jpg|jpeg|bmp|tga|dds|sph|spa)$/)) {
        textureFiles.set(nameLower, url);
        textureFiles.set(f.name, url);
      }
    }
    
    if (!pmxFile) {
      alert('No PMX file found in the selected files.');
      return;
    }

    setLoadingStatus('Loading model...');
    
    // Enhanced URL map for better texture resolution
    const enhancedMap = new Map();
    
    // Add all files to enhanced map
    for (const [name, url] of map.entries()) {
      enhancedMap.set(name, url);
      
      // Add common texture name variations
      const baseName = name.replace(/\.[^/.]+$/, ''); // Remove extension
      enhancedMap.set(baseName, url);
      enhancedMap.set(baseName.toLowerCase(), url);
    }
    
    // Common texture suffixes that might be referenced in PMX
    const textureSuffixes = ['', '_d', '_n', '_s', '_diffuse', '_normal', '_specular'];
    for (const [name, url] of textureFiles.entries()) {
      const baseName = name.replace(/\.[^/.]+$/, '');
      for (const suffix of textureSuffixes) {
        enhancedMap.set(baseName + suffix, url);
        enhancedMap.set((baseName + suffix).toLowerCase(), url);
      }
    }
    
    const modelUrl = map.get(pmxFile.name.toLowerCase());
    
    if (onModelLoaded) {
      onModelLoaded({
        modelUrl,
        urlMap: enhancedMap,
        displayName: pmxFile.name,
        fileCount: files.length,
        textureCount: textureFiles.size
      });
    }
    
    setLoadingStatus('Model loaded successfully!');
    setTimeout(() => setLoadingStatus(''), 2000);
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-sm text-gray-300">Custom PMX Model</div>
        {isLoading && (
          <div className="flex items-center space-x-2 text-primary text-xs">
            <div className="w-3 h-3 animate-spin" style={{
              border: '2px solid var(--primary)',
              borderTop: '2px solid transparent',
              borderRadius: '50%'
            }}></div>
            <span>{loadingStatus}</span>
          </div>
        )}
      </div>
      
      {/* Hidden file inputs */}
      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".pmx,.png,.jpg,.jpeg,.bmp,.tga,.dds,.sph,.spa"
        className="hidden"
        onChange={handleFileChange}
      />
      
      <input
        ref={folderInputRef}
        type="file"
        multiple
        webkitdirectory=""
        directory=""
        className="hidden"
        onChange={handleFolderChange}
      />
      
      {/* Action buttons */}
      <div className="space-y-2">
        <button
          className="btn btn-primary w-full py-2 px-3 text-sm flex items-center justify-center space-x-2"
          onClick={() => folderInputRef.current && folderInputRef.current.click()}
          disabled={isLoading}
        >
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
            <path d="M10 4H4c-1.11 0-2 .89-2 2v12c0 1.11.89 2 2 2h16c1.11 0 2-.89 2-2V8c0-1.11-.89-2-2-2h-8l-2-2z"/>
          </svg>
          <span>Choose Model Folder</span>
        </button>
        
        <button
          className="btn btn-secondary w-full py-2 px-3 text-sm flex items-center justify-center space-x-2"
          onClick={() => inputRef.current && inputRef.current.click()}
          disabled={isLoading}
        >
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
            <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z"/>
          </svg>
          <span>Choose Files</span>
        </button>
      </div>
      
      <div className="text-xs text-gray-500 space-y-1">
        <div>💡 <strong>Recommended:</strong> Choose the entire model folder for best results</div>
        <div>📁 Include: .pmx file + textures (.png, .jpg, .bmp, etc.)</div>
        <div>🎯 Supports: MMD models (.pmx) with Japanese/English bone names</div>
      </div>
    </div>
  );
};

export default ModelPicker;
