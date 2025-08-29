import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import { MMDLoader } from 'three-stdlib';
import { OrbitControls } from 'three-stdlib';

const MMDPreview = ({ modelPath, motionData, trajectory, isPlaying, orient, fps=30, pose2d, urlMap }) => {
  const [loadingStatus, setLoadingStatus] = useState('Initializing 3D scene...');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [boneInfo, setBoneInfo] = useState(null);
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

    // Enhanced grid helper with axes
    const gridHelper = new THREE.GridHelper(40, 40, 0x61dafb, 0x333333);
    gridHelper.material.opacity = 0.4;
    gridHelper.material.transparent = true;
    scene.add(gridHelper);
    
    // Add coordinate axes for better orientation
    const axesHelper = new THREE.AxesHelper(10);
    axesHelper.material.linewidth = 3;
    scene.add(axesHelper);
    
    // Add more detailed grid lines
    const centerLineX = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-20, 0, 0),
      new THREE.Vector3(20, 0, 0)
    ]);
    const centerLineZ = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, -20),
      new THREE.Vector3(0, 0, 20)
    ]);
    
    const centerLineMaterial = new THREE.LineBasicMaterial({ 
      color: 0x61dafb, 
      opacity: 0.8, 
      transparent: true 
    });
    
    scene.add(new THREE.Line(centerLineX, centerLineMaterial));
    scene.add(new THREE.Line(centerLineZ, centerLineMaterial));
    
    // Add height reference markers
    for (let h = 5; h <= 20; h += 5) {
      const heightMarker = new THREE.RingGeometry(0.5, 0.7, 8);
      const heightMaterial = new THREE.MeshBasicMaterial({ 
        color: 0x4fa8c5, 
        transparent: true, 
        opacity: 0.3 
      });
      const marker = new THREE.Mesh(heightMarker, heightMaterial);
      marker.rotation.x = -Math.PI / 2;
      marker.position.y = h;
      scene.add(marker);
    }
    
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
    // For default assets, resolve from public/models.
    // For custom uploads (blob:), use a URL modifier to remap texture paths to provided blob URLs.
    try {
      loader.setPath('/models/');
      loader.setResourcePath('/models/');
      if (urlMap && urlMap.size) {
        const mgr = loader.manager;
        const original = mgr.setURLModifier ? null : null; // placeholder; we only set if available
        mgr.setURLModifier((url) => {
          try {
            const lower = (url || '').toLowerCase();
            const base = lower.split(/[/\\]/).pop();
            if (base && urlMap.has(base)) {
              return urlMap.get(base);
            }
            return url;
          } catch (_) {
            return url;
          }
        });
      }
    } catch (_) {}
    setLoadingStatus('Loading 3D model...');
    try {
      loader.load(
        modelPath.startsWith('/models/') ? modelPath.replace('/models/', '') : modelPath,
        (mmd) => {
          mmd.scale.setScalar(1);
          mmd.position.set(0, 0, 0);
          mmd.castShadow = true;
          mmd.receiveShadow = true;
          if (placeholderGroup.parent) {
            group.remove(placeholderGroup);
          }
          group.add(mmd);
          // Robust bone discovery
          const collectBones = (rootObj) => {
            const bones = [];
            rootObj.traverse(obj => {
              if (obj.isSkinnedMesh && obj.skeleton) {
                for (const b of obj.skeleton.bones) bones.push(b);
              }
              if (obj.isBone) bones.push(obj);
            });
            return bones;
          };
          const allBones = collectBones(mmd);
          const findByPatterns = (patterns) => {
            const pat = patterns.map(p => p.toLowerCase());
            for (const b of allBones) {
              const n = (b.name||'').toLowerCase();
              if (pat.includes(n)) return b;
            }
            for (const b of allBones) {
              const n = (b.name||'').toLowerCase();
              if (pat.some(p => n.includes(p))) return b;
            }
            return null;
          };
          const nameVariants = {
            leftUpperArm: ['左腕','upperarm_l','arm_l','left arm','left_upper_arm','左腕ＩＫ','upper_arm_l','arm.l','shoulder.l'],
            leftLowerArm: ['左ひじ','左肘','lowerarm_l','elbow_l','left elbow','left_lower_arm','forearm_l','lower_arm_l','arm.l.001'],
            leftWrist: ['左手首','wrist_l','hand_l','left wrist','left hand','hand.l'],
            rightUpperArm: ['右腕','upperarm_r','arm_r','right arm','right_upper_arm','右腕ＩＫ','upper_arm_r','arm.r','shoulder.r'],
            rightLowerArm: ['右ひじ','右肘','lowerarm_r','elbow_r','right elbow','right_lower_arm','forearm_r','lower_arm_r','arm.r.001'],
            rightWrist: ['右手首','wrist_r','hand_r','right wrist','right hand','hand.r'],
            // Enhanced leg bone variants - common MMD and Blender names
            leftUpperLeg: ['左足','左太もも','左もも','upperleg_l','leg_l','left leg','thigh_l','leftleg','left_thigh','legupper_l','l thigh','thigh.l','bip l thigh','bip01 l thigh','bip l leg','upper_leg_l','leg_upper_l','左ひざ上','左足上','left_upper_leg','l_thigh'],
            leftLowerLeg: ['左ひざ','左膝','lowerleg_l','knee_l','left knee','shin_l','leftknee','left_shin','leglower_l','l calf','calf.l','bip l calf','bip01 l calf','l shin','lower_leg_l','leg_lower_l','左すね','左ひざ下','left_lower_leg','l_shin','l_calf'],
            rightUpperLeg: ['右足','右太もも','右もも','upperleg_r','leg_r','right leg','thigh_r','rightleg','right_thigh','legupper_r','r thigh','thigh.r','bip r thigh','bip01 r thigh','bip r leg','upper_leg_r','leg_upper_r','右ひざ上','右足上','right_upper_leg','r_thigh'],
            rightLowerLeg: ['右ひざ','右膝','lowerleg_r','knee_r','right knee','shin_r','rightknee','right_shin','leglower_r','r calf','calf.r','bip r calf','bip01 r calf','r shin','lower_leg_r','leg_lower_r','右すね','右ひざ下','right_lower_leg','r_shin','r_calf'],
            leftAnkle: ['左足首','ankle_l','left ankle','foot_l','left_foot','foot.l','l_foot','左足先'],
            rightAnkle: ['右足首','ankle_r','right ankle','foot_r','right_foot','foot.r','r_foot','右足先'],
            leftToe: ['左つま先','toe_l','left_toe','toe.l','l_toe'],
            rightToe: ['右つま先','toe_r','right_toe','toe.r','r_toe'],
            hip: ['下半身','hips','pelvis','hip','root','waist','センター'],
            spine: ['上半身','spine','chest','torso','body'],
            neck: ['首','neck','head_base'],
            head: ['頭','head']
          };
          const bones = {};
          bones.leftUpperArm = findByPatterns(nameVariants.leftUpperArm);
          bones.leftLowerArm = findByPatterns(nameVariants.leftLowerArm);
          bones.leftWrist = findByPatterns(nameVariants.leftWrist);
          bones.rightUpperArm = findByPatterns(nameVariants.rightUpperArm);
          bones.rightLowerArm = findByPatterns(nameVariants.rightLowerArm);
          bones.rightWrist = findByPatterns(nameVariants.rightWrist);
          bones.leftUpperLeg = findByPatterns(nameVariants.leftUpperLeg);
          bones.leftLowerLeg = findByPatterns(nameVariants.leftLowerLeg);
          bones.rightUpperLeg = findByPatterns(nameVariants.rightUpperLeg);
          bones.rightLowerLeg = findByPatterns(nameVariants.rightLowerLeg);
          bones.neck = findByPatterns(nameVariants.neck);
          bones.head = findByPatterns(nameVariants.head);
          bones.leftAnkle = findByPatterns(nameVariants.leftAnkle);
          bones.rightAnkle = findByPatterns(nameVariants.rightAnkle);
          bones.leftToe = findByPatterns(nameVariants.leftToe);
          bones.rightToe = findByPatterns(nameVariants.rightToe);
          bones.hip = findByPatterns(nameVariants.hip);
          bones.spine = findByPatterns(nameVariants.spine);
          
          boneMapRef.current = bones;
          // Fallback: if upper/lower leg missing, try IK bones (will still rotate something)
          if (!bones.leftUpperLeg) bones.leftUpperLeg = findByPatterns(['左足ＩＫ','leg_ik_l','foot_ik_l']);
          if (!bones.rightUpperLeg) bones.rightUpperLeg = findByPatterns(['右足ＩＫ','leg_ik_r','foot_ik_r']);
          // Log found bones for debugging
          const found = Object.fromEntries(Object.entries(bones).map(([k,v])=>[k, v? v.name : null]));
          console.log('[MMDPreview] Bone map:', found);
          
          // Debug: Log ALL bone names for reference
          console.log('[MMDPreview] All available bones:', allBones.map(b => b.name).sort());
          
          // Count successful bone mappings
          const successfulMappings = Object.values(bones).filter(b => b !== null).length;
          const totalMappings = Object.keys(bones).length;
          console.log(`[MMDPreview] Bone mapping success: ${successfulMappings}/${totalMappings}`);
          
          setBoneInfo(found);
          // Reset skeleton to bind pose for consistent rotations
          mmd.traverse(obj => { if (obj.isSkinnedMesh && obj.skeleton) obj.skeleton.pose(); });
          // Face camera baseline with X=0
          const FACE_CAMERA_BASE_Y = Math.PI;
          group.rotation.set(0, FACE_CAMERA_BASE_Y, 0);
          setLoadingStatus(`Model loaded! Found ${successfulMappings}/${totalMappings} bones`);
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
    
    // Add debugging counter
    let debugCounter = 0;
    
    const retargetFromPose2D = (kps) => {
      if (!kps) return;
      
      // Debug every 60 frames (1 second at 60fps)
      debugCounter++;
      const shouldDebug = debugCounter % 60 === 0;
      
      if (shouldDebug) {
        console.log('[MMDPreview] Retarget Debug:', {
          keypointsCount: kps.length,
          sampleKeypoint: kps[11], // Left shoulder
          boneMapExists: !!boneMapRef.current,
          availableBones: boneMapRef.current ? Object.keys(boneMapRef.current).filter(k => boneMapRef.current[k] !== null) : []
        });
      }
      
      // Indices per MediaPipe
      const idx = { LS:11, RS:12, LE:13, RE:14, LW:15, RW:16, LH:23, RH:24, LK:25, RK:26, LA:27, RA:28 };
      const get = (i) => (kps[i] || [0.5,0.5,0.0]);
      const v2 = (a, b) => ({ x: (b[0]-a[0]), y: (a[1]-b[1]) }); // screen up-positive
      const ang = (v, sx=0.35, sy=0.35) => ({ yaw: Math.atan2(v.x, sx), pitch: Math.atan2(v.y, sy) });
      const len = (v) => Math.hypot(v.x, v.y) + 1e-6;
      const crossZ = (a, b) => a.x*b.y - a.y*b.x; // 2D cross product z-component
      // Keep simple baselines to infer foreshortening (depth proxy)
      const baseRef = (name) => {
        if (!boneMapRef.current.__baselines) boneMapRef.current.__baselines = {};
        if (boneMapRef.current.__baselines[name] === undefined) boneMapRef.current.__baselines[name] = 0;
        return boneMapRef.current.__baselines;
      };
      const L = boneMapRef.current;
      // Arms
      const ls = get(idx.LS), le = get(idx.LE), lw = get(idx.LW);
      const rs = get(idx.RS), re = get(idx.RE), rw = get(idx.RW);
      const vLUp = v2(ls, le), aLUp = ang(vLUp);
      const vLLow = v2(le, lw), aLLow = ang(vLLow);
      const vRUp = v2(rs, re), aRUp = ang(vRUp);
      const vRLow = v2(re, rw), aRLow = ang(vRLow);
      // Depth proxies via foreshortening
      const bl = baseRef('armL'); const br = baseRef('armR');
      bl.armL = Math.max(bl.armL||0, len(v2(ls, lw)));
      br.armR = Math.max(br.armR||0, len(v2(rs, rw)));
      const dFL = clamp(((bl.armL||1)-len(v2(ls,lw)))/Math.max(1e-6,(bl.armL||1)), 0, 1); // 0..1 toward camera
      const dFR = clamp(((br.armR||1)-len(v2(rs,rw)))/Math.max(1e-6,(br.armR||1)), 0, 1);
      // Roll from 2D bend sign
      const rollL = clamp(crossZ(vLUp, vLLow)/(len(vLUp)*len(vLLow)), -1, 1) * (0.8 + 0.6*dFL);
      const rollR = clamp(crossZ(vRUp, vRLow)/(len(vRUp)*len(vRLow)), -1, 1) * (0.8 + 0.6*dFR);
      const upScale = 0.9, lowScale = 1.1;
      const qLUp = toQuat(
        clamp(aLUp.pitch*upScale, -1.2, 1.2),
        clamp(aLUp.yaw*upScale, -1.2, 1.2),
        clamp(rollL*0.8, -1.0, 1.0)
      );
      const qLLow = toQuat(
        clamp(aLLow.pitch*lowScale, -1.5, 1.5),
        clamp(aLLow.yaw*0.6, -0.8, 0.8),
        clamp(rollL*0.6, -0.8, 0.8)
      );
      const qRUp = toQuat(
        clamp(aRUp.pitch*upScale, -1.2, 1.2),
        clamp(aRUp.yaw*upScale, -1.2, 1.2),
        clamp(-rollR*0.8, -1.0, 1.0)
      );
      const qRLow = toQuat(
        clamp(aRLow.pitch*lowScale, -1.5, 1.5),
        clamp(aRLow.yaw*0.6, -0.8, 0.8),
        clamp(-rollR*0.6, -0.8, 0.8)
      );
      slerpTo(L.leftUpperArm, qLUp, 0.35);
      slerpTo(L.leftLowerArm, qLLow, 0.35);
      slerpTo(L.rightUpperArm, qRUp, 0.35);
      slerpTo(L.rightLowerArm, qRLow, 0.35);
      // Wrists follow forearm direction lightly
      const qLWrist = toQuat(
        clamp(aLLow.pitch*0.6, -0.8, 0.8),
        clamp(aLLow.yaw*0.5, -0.8, 0.8),
        clamp(rollL*0.5, -0.6, 0.6)
      );
      const qRWrist = toQuat(
        clamp(aRLow.pitch*0.6, -0.8, 0.8),
        clamp(aRLow.yaw*0.5, -0.8, 0.8),
        clamp(-rollR*0.5, -0.6, 0.6)
      );
      slerpTo(L.leftWrist, qLWrist, 0.3);
      slerpTo(L.rightWrist, qRWrist, 0.3);
      // Legs (enhanced with depth and smoothing)
      const lh = get(idx.LH), lk = get(idx.LK), la = get(idx.LA);
      const rh = get(idx.RH), rk = get(idx.RK), ra = get(idx.RA);
      
      // Calculate leg vectors with improved depth awareness
      const vLThigh = v2(lh, lk), aLThigh = ang(vLThigh, 0.5, 0.5);
      const vLShin = v2(lk, la), aLShin = ang(vLShin, 0.45, 0.45);
      const vRThigh = v2(rh, rk), aRThigh = ang(vRThigh, 0.5, 0.5);
      const vRShin = v2(rk, ra), aRShin = ang(vRShin, 0.45, 0.45);
      
      // Enhanced depth estimation for legs
      const blg = baseRef('leg');
      const fullLegL = len(v2(lh, la));
      const fullLegR = len(v2(rh, ra));
      
      // Initialize baselines
      if (!blg.legL) blg.legL = fullLegL;
      if (!blg.legR) blg.legR = fullLegR;
      
      // Update baselines with exponential moving average
      blg.legL = blg.legL * 0.95 + fullLegL * 0.05;
      blg.legR = blg.legR * 0.95 + fullLegR * 0.05;
      
      // Improved depth proxy (0 = away, 1 = toward camera)
      const dLL = clamp((blg.legL - fullLegL) / Math.max(1e-6, blg.legL), 0, 1);
      const dLR = clamp((blg.legR - fullLegR) / Math.max(1e-6, blg.legR), 0, 1);
      
      // Enhanced roll calculation with depth influence
      const thighLen = len(vLThigh), shinLen = len(vLShin);
      const thighLenR = len(vRThigh), shinLenR = len(vRShin);
      
      const rollLegL = clamp(crossZ(vLThigh, vLShin) / Math.max(1e-6, thighLen * shinLen), -1, 1) * (0.8 + 0.7 * dLL);
      const rollLegR = clamp(crossZ(vRThigh, vRShin) / Math.max(1e-6, thighLenR * shinLenR), -1, 1) * (0.8 + 0.7 * dLR);
      
      // Hip movement influence on leg orientation
      const hipCenterY = (lh[1] + rh[1]) * 0.5;
      const hipTilt = Math.atan2(rh[1] - lh[1], rh[0] - lh[0]) * 0.3;
      
      // Enhanced leg rotations with hip influence
      const qLThigh = toQuat(
        clamp(aLThigh.pitch * 1.2, -1.4, 1.4), // Increased range for better movement
        clamp(aLThigh.yaw * 0.6 + hipTilt, -0.8, 0.8),
        clamp(rollLegL * 0.7, -0.9, 0.9)
      );
      
      const qLShin = toQuat(
        clamp(aLShin.pitch * 1.4 + Math.max(0, -aLThigh.pitch) * 0.5, -1.6, 0.2), // More natural knee bend
        clamp(aLShin.yaw * 0.4, -0.6, 0.6),
        clamp(rollLegL * 0.5, -0.7, 0.7)
      );
      
      const qRThigh = toQuat(
        clamp(aRThigh.pitch * 1.2, -1.4, 1.4),
        clamp(aRThigh.yaw * 0.6 - hipTilt, -0.8, 0.8),
        clamp(-rollLegR * 0.7, -0.9, 0.9)
      );
      
      const qRShin = toQuat(
        clamp(aRShin.pitch * 1.4 + Math.max(0, -aRThigh.pitch) * 0.5, -1.6, 0.2),
        clamp(aRShin.yaw * 0.4, -0.6, 0.6),
        clamp(-rollLegR * 0.5, -0.7, 0.7)
      );

      // Apply with higher blend factor for more responsive leg movement
      slerpTo(L.leftUpperLeg, qLThigh, 0.7);
      slerpTo(L.leftLowerLeg, qLShin, 0.7);
      slerpTo(L.rightUpperLeg, qRThigh, 0.7);
      slerpTo(L.rightLowerLeg, qRShin, 0.7);
      
      // Enhanced ankle movement with ground contact awareness
      const groundY = 0.95; // Normalized Y for ground level
      const leftFootOnGround = la[1] > groundY;
      const rightFootOnGround = ra[1] > groundY;
      
      const ankleFactorL = leftFootOnGround ? 0.8 : 0.4;
      const ankleFactorR = rightFootOnGround ? 0.8 : 0.4;
      
      const qLAnkle = toQuat(
        clamp(aLShin.pitch * ankleFactorL, -0.9, 0.9),
        clamp(aLShin.yaw * 0.4, -0.6, 0.6),
        clamp(rollLegL * 0.3, -0.5, 0.5)
      );
      
      const qRAnkle = toQuat(
        clamp(aRShin.pitch * ankleFactorR, -0.9, 0.9),
        clamp(aRShin.yaw * 0.4, -0.6, 0.6),
        clamp(-rollLegR * 0.3, -0.5, 0.5)
      );
      
      slerpTo(L.leftAnkle, qLAnkle, 0.6);
      slerpTo(L.rightAnkle, qRAnkle, 0.6);
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
            // Keep X rotation at 0; apply yaw around Y, add base Y to face camera
            const FACE_CAMERA_BASE_Y = Math.PI;
            group.rotation.set(0, FACE_CAMERA_BASE_Y + yaw, 0);
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

      {/* Bone map debug (legs) */}
      {boneInfo && (
        <div className="absolute bottom-4 left-4 bg-black/60 text-white px-3 py-2 rounded-lg text-xs space-y-1">
          <div className="font-semibold text-primary">Leg bones</div>
          <div>Left Thigh: <span className="text-gray-300">{boneInfo.leftUpperLeg || 'not found'}</span></div>
          <div>Left Shin: <span className="text-gray-300">{boneInfo.leftLowerLeg || 'not found'}</span></div>
          <div>Right Thigh: <span className="text-gray-300">{boneInfo.rightUpperLeg || 'not found'}</span></div>
          <div>Right Shin: <span className="text-gray-300">{boneInfo.rightLowerLeg || 'not found'}</span></div>
        </div>
      )}
    </div>
  );
};

export default MMDPreview;
