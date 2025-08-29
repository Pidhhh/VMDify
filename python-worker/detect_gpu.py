import subprocess
import sys
import re

def detect_gpu():
    """
    Detect NVIDIA GPU and CUDA version to recommend the best PyTorch installation.
    """
    print("🔍 Detecting your system configuration...")
    print("=" * 50)
    
    # Check for NVIDIA GPU
    nvidia_gpu = None
    try:
        # Try nvidia-smi command
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            nvidia_gpu = result.stdout.strip().split('\n')[0]
            print(f"✅ NVIDIA GPU detected: {nvidia_gpu}")
        else:
            print("❌ No NVIDIA GPU detected via nvidia-smi")
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        print("❌ nvidia-smi not found or failed - No NVIDIA GPU detected")
    
    # Check CUDA version if GPU is detected
    cuda_version = None
    if nvidia_gpu:
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                # Extract CUDA version from nvidia-smi output
                cuda_match = re.search(r'CUDA Version: (\d+\.\d+)', result.stdout)
                if cuda_match:
                    cuda_version = cuda_match.group(1)
                    print(f"✅ CUDA Version detected: {cuda_version}")
                else:
                    print("⚠️  Could not detect CUDA version from nvidia-smi")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            print("⚠️  Failed to get CUDA version information")
    
    print("=" * 50)
    
    # Make recommendation
    if nvidia_gpu and cuda_version:
        cuda_major = float(cuda_version)
        print("🎯 RECOMMENDATION:")
        
        if cuda_major >= 12.9:
            print("   Install PyTorch with CUDA 12.9 (Latest)")
            print("   Command: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129")
            return "cuda129"
        elif cuda_major >= 12.8:
            print("   Install PyTorch with CUDA 12.8")
            print("   Command: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
            return "cuda128"
        elif cuda_major >= 12.6:
            print("   Install PyTorch with CUDA 12.6")
            print("   Command: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126")
            return "cuda126"
        else:
            print("   Your CUDA version is older than 12.6")
            print("   Install CPU-only PyTorch for compatibility")
            print("   Command: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
            return "cpu"
    elif nvidia_gpu:
        print("🎯 RECOMMENDATION:")
        print("   NVIDIA GPU detected but CUDA version unclear")
        print("   Try CUDA 12.9 (latest) or fall back to CPU-only if it fails")
        print("   Command: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129")
        return "cuda129"
    else:
        print("🎯 RECOMMENDATION:")
        print("   No NVIDIA GPU detected")
        print("   Install CPU-only PyTorch")
        print("   Command: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
        return "cpu"

def main():
    print("VMDify GPU Detection Tool")
    print("=" * 50)
    
    recommendation = detect_gpu()
    
    print("\n" + "=" * 50)
    print("ℹ️  Additional Information:")
    print("   - CPU-only PyTorch will work but will be slower")
    print("   - CUDA PyTorch requires NVIDIA GPU with compatible drivers")
    print("   - You can always reinstall PyTorch later if needed")
    print("=" * 50)
    
    return recommendation

if __name__ == "__main__":
    main()
