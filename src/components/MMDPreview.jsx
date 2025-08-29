import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import { MMDLoader } from 'three-stdlib';
import { OrbitControls } from 'three-stdlib';

const MMDPreview = ({ modelPath, motionData, trajectory, isPlaying }) => {
  const [loadingStatus, setLoadingStatus] = useState('Initializing 3D scene...');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const mountRef = useRef(null);

  useEffect(() => {
    if (!mountRef.current) return;

    const currentMount = mountRef.current;

    // === Basic Three.js Scene Setup ===
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);

    const camera = new THREE.PerspectiveCamera(
      45,
      currentMount.clientWidth / currentMount.clientHeight,
      0.1,
      1000
    );
    camera.position.set(0, 15, 40);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(currentMount.clientWidth, currentMount.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    currentMount.appendChild(renderer.domElement);
    
    // OrbitControls for camera manipulation
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 10, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 100;
    controls.minDistance = 5;
    controls.update();

    // Add a modern grid helper
    const gridHelper = new THREE.GridHelper(30, 30, 0x61dafb, 0x333333);
    gridHelper.material.opacity = 0.3;
    gridHelper.material.transparent = true;
    scene.add(gridHelper);
    
    // Enhanced lighting setup
    const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 10, 5);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.width = 2048;
    directionalLight.shadow.mapSize.height = 2048;
    directionalLight.shadow.camera.near = 0.5;
    directionalLight.shadow.camera.far = 50;
    scene.add(directionalLight);

    // Add rim lighting
    const rimLight = new THREE.DirectionalLight(0x61dafb, 0.5);
    rimLight.position.set(-5, 10, -5);
    scene.add(rimLight);

  // === Model or Placeholder ===
  // Container that will always stay in the scene and be moved by trajectory
  const group = new THREE.Group();
    
  // Placeholder model assembled in its own group so we can swap it out cleanly
  const placeholderGroup = new THREE.Group();

  // Main body
    const bodyGeometry = new THREE.CapsuleGeometry(1, 3, 4, 8);
    const bodyMaterial = new THREE.MeshPhongMaterial({ 
      color: 0x61dafb,
      shininess: 100,
      transparent: true,
      opacity: 0.8
    });
    const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
    body.position.y = 3;
    body.castShadow = true;
    body.receiveShadow = true;
  placeholderGroup.add(body);
    
    // Head
    const headGeometry = new THREE.SphereGeometry(0.8, 32, 32);
    const headMaterial = new THREE.MeshPhongMaterial({ 
      color: 0x4fa8c5,
      shininess: 100
    });
    const head = new THREE.Mesh(headGeometry, headMaterial);
    head.position.y = 5.5;
    head.castShadow = true;
    head.receiveShadow = true;
  placeholderGroup.add(head);

    // Arms
    const armGeometry = new THREE.CylinderGeometry(0.3, 0.3, 2, 8);
    const armMaterial = new THREE.MeshPhongMaterial({ 
      color: 0x61dafb,
      transparent: true,
      opacity: 0.7
    });
    
    const leftArm = new THREE.Mesh(armGeometry, armMaterial);
    leftArm.position.set(-1.5, 3.5, 0);
    leftArm.rotation.z = Math.PI / 6;
    leftArm.castShadow = true;
  placeholderGroup.add(leftArm);
    
    const rightArm = new THREE.Mesh(armGeometry, armMaterial);
    rightArm.position.set(1.5, 3.5, 0);
    rightArm.rotation.z = -Math.PI / 6;
    rightArm.castShadow = true;
  placeholderGroup.add(rightArm);

    // Legs
    const legGeometry = new THREE.CylinderGeometry(0.4, 0.4, 2.5, 8);
    const legMaterial = new THREE.MeshPhongMaterial({ 
      color: 0x4fa8c5,
      transparent: true,
      opacity: 0.8
    });
    
    const leftLeg = new THREE.Mesh(legGeometry, legMaterial);
    leftLeg.position.set(-0.5, 0.5, 0);
    leftLeg.castShadow = true;
  placeholderGroup.add(leftLeg);
    
    const rightLeg = new THREE.Mesh(legGeometry, legMaterial);
    rightLeg.position.set(0.5, 0.5, 0);
    rightLeg.castShadow = true;
  placeholderGroup.add(rightLeg);
    
  group.add(placeholderGroup);
  scene.add(group);
    
    // Add a modern ground plane with gradient
    const planeGeometry = new THREE.PlaneGeometry(50, 50);
    const planeMaterial = new THREE.MeshLambertMaterial({ 
      color: 0x2a2a2a,
      transparent: true,
      opacity: 0.8
    });
    const plane = new THREE.Mesh(planeGeometry, planeMaterial);
    plane.rotation.x = -Math.PI / 2;
    plane.receiveShadow = true;
    scene.add(plane);
    
    // Try loading PMX model; if it fails, keep placeholder
    const loader = new MMDLoader();
    // Ensure resources resolve from public/models and its textures
    try {
      loader.setPath('/models/');
      loader.setResourcePath('/models/');
    } catch (_) {}
    setLoadingStatus('Loading 3D model...');
    try {
      loader.load(
        // If setPath above works, pass just the filename
        modelPath.startsWith('/models/') ? modelPath.replace('/models/', '') : modelPath,
        (mmd) => {
          mmd.scale.setScalar(1);
          mmd.position.set(0, 0, 0);
          mmd.castShadow = true;
          mmd.receiveShadow = true;
          // Replace placeholder with model, but keep container in scene
          if (placeholderGroup.parent) {
            group.remove(placeholderGroup);
          }
          group.add(mmd);
          setLoadingStatus('Model loaded successfully!');
          setIsLoading(false);
          console.log('MMD model loaded:', mmd);
        },
        (xhr) => {
          if (xhr.lengthComputable) {
            const percentComplete = (xhr.loaded / xhr.total * 100);
            setLoadingStatus(`Loading: ${Math.round(percentComplete)}%`);
          }
        },
        (error) => {
          console.warn('PMX load failed, showing placeholder:', error);
          setLoadingStatus('Using placeholder (PMX load failed)');
          setIsLoading(false);
        }
      );
    } catch (e) {
      console.warn('PMX loader error, using placeholder:', e);
      setLoadingStatus('Using placeholder (PMX load failed)');
      setIsLoading(false);
    }

    // Simple animation for the placeholder
    let animationTime = 0;
    let playIndex = 0;
    const animate = () => {
      requestAnimationFrame(animate);
      
      animationTime += 0.01;

      // Drive from trajectory if available and isPlaying
      if (Array.isArray(trajectory) && trajectory.length > 0 && isPlaying) {
        // simple frame stepping
        const step = trajectory[Math.min(playIndex, trajectory.length - 1)];
        if (step && Array.isArray(step.pos)) {
          const [x, y, z] = step.pos;
          group.position.set(x, Math.max(0, y) + 0.0, z);
        }
        playIndex = (playIndex + 1) % trajectory.length;
      } else {
        // Gentle idle motion when not playing
        group.position.y = Math.sin(animationTime) * 0.2;
        group.rotation.y = Math.sin(animationTime * 0.5) * 0.1;
      }
      
      controls.update();
      renderer.render(scene, camera);
    };

    animate();

    // Handle resize
    const handleResize = () => {
        if (currentMount) {
            camera.aspect = currentMount.clientWidth / currentMount.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(currentMount.clientWidth, currentMount.clientHeight);
        }
    };
    window.addEventListener('resize', handleResize);

    // Handle component unmount
    return () => {
        window.removeEventListener('resize', handleResize);
        if (currentMount && renderer.domElement.parentNode === currentMount) {
            currentMount.removeChild(renderer.domElement);
        }
        renderer.dispose();
    };
  }, [modelPath, trajectory, isPlaying]);

  return (
    <div className="relative w-full h-full min-h-500 bg-dark rounded-lg overflow-hidden">
      <div ref={mountRef} className="w-full h-full" />
      
      {/* Status Overlay */}
      <div className="absolute top-4 left-4 space-y-2">
        <div className="bg-black/70 backdrop-blur-sm text-white px-3 py-2 rounded-lg text-sm flex items-center space-x-2">
          {isLoading ? (
            <div className="w-3 h-3 animate-spin" style={{
              border: '2px solid var(--primary)',
              borderTop: '2px solid transparent',
              borderRadius: '50%'
            }}></div>
          ) : error ? (
            <svg className="w-4 h-4 text-red-400" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2C6.47 2 2 6.47 2 12s4.47 10 10 10 10-4.47 10-10S17.53 2 12 2zm5 13.59L15.59 17 12 13.41 8.41 17 7 15.59 10.59 12 7 8.41 8.41 7 12 10.59 15.59 7 17 8.41 13.41 12 17 15.59z"/>
            </svg>
          ) : (
            <div className="w-2 h-2 bg-green-400 rounded-full"></div>
          )}
          <span>{loadingStatus}</span>
        </div>
        
        {/* Motion Data Indicator */}
        {motionData && (
          <div className="backdrop-blur-sm text-primary px-3 py-2 rounded-lg text-sm flex items-center space-x-2" style={{
            backgroundColor: 'rgba(97, 218, 251, 0.2)'
          }}>
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
              <path d="M13 10V3L4 14h7v7l9-11h-7z"/>
            </svg>
            <span>Motion Data Active</span>
          </div>
        )}
      </div>

      {/* Controls Help */}
      <div className="absolute bottom-4 right-4 bg-black/70 backdrop-blur-sm text-white px-3 py-2 rounded-lg text-xs">
        <div className="space-y-1">
          <div>Left Click + Drag: Rotate</div>
          <div>Right Click + Drag: Pan</div>
          <div>Scroll: Zoom</div>
        </div>
      </div>

      {/* Performance Info */}
      <div className="absolute top-4 right-4 bg-black/70 backdrop-blur-sm text-white px-3 py-2 rounded-lg text-xs">
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-green-400 rounded-full"></div>
          <span>60 FPS</span>
        </div>
      </div>
    </div>
  );
};

export default MMDPreview;
