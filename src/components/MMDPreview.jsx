import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import { MMDLoader } from 'three-stdlib';
import { OrbitControls } from 'three-stdlib';

const MMDPreview = ({ modelPath, motionData, trajectory, isPlaying, orient, fps=30, pose2d }) => {
  const [loadingStatus, setLoadingStatus] = useState('Initializing 3D scene...');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const mountRef = useRef(null);
  const trajRef = useRef(null);
  const playRef = useRef(false);
  const fpsRef = useRef(30);
  const orientRef = useRef(null);
  const poseRef = useRef(null);
  const boneMapRef = useRef({});

  // Keep latest values in refs so animate loop reads them without re-creating scene
  useEffect(() => { trajRef.current = trajectory; }, [trajectory]);
  useEffect(() => { playRef.current = isPlaying; }, [isPlaying]);
  useEffect(() => { orientRef.current = orient; }, [orient]);
  useEffect(() => { if (fps && Number.isFinite(fps)) fpsRef.current = fps; }, [fps]);
  useEffect(() => { poseRef.current = pose2d; }, [pose2d]);

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
          // Build a simple bone map (supports JP + EN variants)
          const nameVariants = {
            leftUpperArm: ['左腕', 'UpperArm_L', 'Arm_L', 'Left arm', 'Left Arm'],
            leftLowerArm: ['左ひじ', '左肘', 'LowerArm_L', 'Elbow_L', 'Left elbow', 'Left Elbow'],
            rightUpperArm: ['右腕', 'UpperArm_R', 'Arm_R', 'Right arm', 'Right Arm'],
            rightLowerArm: ['右ひじ', '右肘', 'LowerArm_R', 'Elbow_R', 'Right elbow', 'Right Elbow'],
            leftUpperLeg: ['左足', '左太もも', 'UpperLeg_L', 'Leg_L', 'Left leg', 'Left Leg'],
            leftLowerLeg: ['左ひざ', '左膝', 'LowerLeg_L', 'Knee_L', 'Left knee', 'Left Knee'],
            rightUpperLeg: ['右足', '右太もも', 'UpperLeg_R', 'Leg_R', 'Right leg', 'Right Leg'],
            rightLowerLeg: ['右ひざ', '右膝', 'LowerLeg_R', 'Knee_R', 'Right knee', 'Right Knee'],
            neck: ['首', 'Neck'],
            head: ['頭', 'Head']
          };
          const findBone = (root, candidates) => {
            for (const nm of candidates) {
              const obj = root.getObjectByName(nm);
              if (obj) return obj;
            }
            return null;
          };
          const bones = {};
          bones.leftUpperArm = findBone(mmd, nameVariants.leftUpperArm);
          bones.leftLowerArm = findBone(mmd, nameVariants.leftLowerArm);
          bones.rightUpperArm = findBone(mmd, nameVariants.rightUpperArm);
          bones.rightLowerArm = findBone(mmd, nameVariants.rightLowerArm);
          bones.leftUpperLeg = findBone(mmd, nameVariants.leftUpperLeg);
          bones.leftLowerLeg = findBone(mmd, nameVariants.leftLowerLeg);
          bones.rightUpperLeg = findBone(mmd, nameVariants.rightUpperLeg);
          bones.rightLowerLeg = findBone(mmd, nameVariants.rightLowerLeg);
          bones.neck = findBone(mmd, nameVariants.neck);
          bones.head = findBone(mmd, nameVariants.head);
          boneMapRef.current = bones;
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
    let lastTimeMs = performance.now();
    let accum = 0; // ms
    // Retarget helpers
    const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
    const toQuat = (rx, ry, rz) => new THREE.Quaternion().setFromEuler(new THREE.Euler(rx, ry, rz, 'XYZ'));
    const slerpTo = (bone, qTarget, alpha=0.25) => {
      if (!bone) return;
      bone.quaternion.slerp(qTarget, alpha);
      bone.updateMatrixWorld();
    };
    const retargetFromPose2D = (kps) => {
      if (!kps) return;
      // Indices per MediaPipe
      const idx = { LS:11, RS:12, LE:13, RE:14, LW:15, RW:16, LH:23, RH:24, LK:25, RK:26, LA:27, RA:28 };
      const get = (i) => (kps[i] || [0.5,0.5,0.0]);
      const v2 = (a, b) => ({ x: (b[0]-a[0]), y: (a[1]-b[1]) }); // screen up-positive
      const ang = (v, sx=0.35, sy=0.35) => ({ yaw: Math.atan2(v.x, sx), pitch: Math.atan2(v.y, sy) });
      const L = boneMapRef.current;
      // Arms
      const ls = get(idx.LS), le = get(idx.LE), lw = get(idx.LW);
      const rs = get(idx.RS), re = get(idx.RE), rw = get(idx.RW);
      const vLUp = v2(ls, le), aLUp = ang(vLUp);
      const vLLow = v2(le, lw), aLLow = ang(vLLow);
      const vRUp = v2(rs, re), aRUp = ang(vRUp);
      const vRLow = v2(re, rw), aRLow = ang(vRLow);
      const upScale = 0.9, lowScale = 1.1;
      const qLUp = toQuat(clamp(aLUp.pitch*upScale, -1.2, 1.2), clamp(aLUp.yaw*upScale, -1.2, 1.2), 0);
      const qLLow = toQuat(clamp(aLLow.pitch*lowScale, -1.5, 1.5), clamp(aLLow.yaw*0.6, -0.8, 0.8), 0);
      const qRUp = toQuat(clamp(aRUp.pitch*upScale, -1.2, 1.2), clamp(aRUp.yaw*upScale, -1.2, 1.2), 0);
      const qRLow = toQuat(clamp(aRLow.pitch*lowScale, -1.5, 1.5), clamp(aRLow.yaw*0.6, -0.8, 0.8), 0);
      slerpTo(L.leftUpperArm, qLUp, 0.25);
      slerpTo(L.leftLowerArm, qLLow, 0.25);
      slerpTo(L.rightUpperArm, qRUp, 0.25);
      slerpTo(L.rightLowerArm, qRLow, 0.25);
      // Legs (pitch dominant)
      const lh = get(idx.LH), lk = get(idx.LK), la = get(idx.LA);
      const rh = get(idx.RH), rk = get(idx.RK), ra = get(idx.RA);
      const vLThigh = v2(lh, lk), aLThigh = ang(vLThigh, 0.4, 0.4);
      const vLShin = v2(lk, la), aLShin = ang(vLShin, 0.35, 0.35);
      const vRThigh = v2(rh, rk), aRThigh = ang(vRThigh, 0.4, 0.4);
      const vRShin = v2(rk, ra), aRShin = ang(vRShin, 0.35, 0.35);
      const qLThigh = toQuat(clamp(aLThigh.pitch, -1.0, 1.0), clamp(aLThigh.yaw*0.4, -0.6, 0.6), 0);
      const qLShin = toQuat(clamp(aLShin.pitch, -1.2, 1.2), clamp(aLShin.yaw*0.3, -0.5, 0.5), 0);
      const qRThigh = toQuat(clamp(aRThigh.pitch, -1.0, 1.0), clamp(aRThigh.yaw*0.4, -0.6, 0.6), 0);
      const qRShin = toQuat(clamp(aRShin.pitch, -1.2, 1.2), clamp(aRShin.yaw*0.3, -0.5, 0.5), 0);
      slerpTo(L.leftUpperLeg, qLThigh, 0.25);
      slerpTo(L.leftLowerLeg, qLShin, 0.25);
      slerpTo(L.rightUpperLeg, qRThigh, 0.25);
      slerpTo(L.rightLowerLeg, qRShin, 0.25);
      // Head/neck from orient if available
      const curOrient = orientRef.current;
      if (curOrient && (L.neck || L.head)) {
        const yaw = curOrient.head?.yaw || 0;
        const pitch = curOrient.head?.pitch || 0;
        const qHead = toQuat(clamp(pitch, -0.6, 0.6), clamp(yaw, -0.8, 0.8), 0);
        slerpTo(L.neck, qHead, 0.2);
        slerpTo(L.head, qHead, 0.2);
      }
    };

    const animate = () => {
      requestAnimationFrame(animate);

      const now = performance.now();
      const dt = now - lastTimeMs;
      lastTimeMs = now;
      accum += dt;

      animationTime += dt * 0.001;

      // Drive from trajectory if available and isPlaying, at the video FPS
      const curTraj = trajRef.current;
      const curPlay = playRef.current;
      const targetFrameMs = 1000 / Math.max(1, fpsRef.current);
      if (Array.isArray(curTraj) && curTraj.length > 0 && curPlay) {
        while (accum >= targetFrameMs) {
          const step = curTraj[Math.min(playIndex, curTraj.length - 1)];
          if (step && Array.isArray(step.pos)) {
            // Map detection coords (x right, y up, z forward) to Three's (x right, y up, z forward)
            // If your pipeline z forward matches camera-in, but Three's camera looks -z, keep the sign consistent with your scene
            const [x, y, z] = step.pos;
            // Flip Z if camera faces -Z so forward movement is visible
            group.position.set(x, Math.max(0, y), -z);
          }
          const curOrient = orientRef.current;
          if (curOrient && curOrient.body) {
            const yaw = curOrient.body.yaw || 0;
            const pitch = curOrient.body.pitch || 0;
            // Apply body yaw/pitch on the container; heads will be bone-level later
            group.rotation.set(pitch, yaw, 0);
          }
          // Retarget limbs from latest 2D pose
          if (poseRef.current) {
            retargetFromPose2D(poseRef.current);
          }
          playIndex += 1;
          accum -= targetFrameMs;
          if (playIndex >= curTraj.length) {
            // Hold on last available frame; continue when new frames arrive
            playIndex = Math.max(0, curTraj.length - 1);
            accum = 0;
            break;
          }
        }
      } else {
        // Hold last pose when not playing to avoid random motion
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
        try {
          scene.traverse((obj) => {
            if (obj.isMesh) {
              if (obj.geometry) obj.geometry.dispose();
              if (obj.material) {
                const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
                mats.forEach((m) => {
                  if (m.map) m.map.dispose();
                  if (m.dispose) m.dispose();
                });
              }
            }
          });
          renderer.dispose();
        } catch (_) {}
    };
  }, [modelPath]);

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
