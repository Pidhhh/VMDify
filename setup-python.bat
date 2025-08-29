@echo off
setlocal EnableDelayedExpansion
echo ========================================
echo    VMDify Python Backend Setup
echo ========================================
echo.
echo This will set up the Python environment for VMDify's AI backend.
echo Make sure Python 3.8+ is installed on your system.
echo.

echo.
echo Step 1: Checking python-worker directory...
if not exist "python-worker" (
    echo ERROR: python-worker directory not found!
    echo Make sure you're running this from the main VMDify folder.
    pause
    exit /b 1
)

echo.
echo Step 2: Creating virtual environment in main folder...
if exist "venv" (
    echo Virtual environment already exists! Skipping creation.
) else (
    echo Creating new virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment. Make sure Python is installed.
        pause
        exit /b 1
    )
)

echo.
echo Step 3: Activating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment. Make sure Python is installed.
    pause
    exit /b 1
)

echo.
echo Step 4: Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Verifying Python version in virtual environment...
python --version

echo.
echo Step 5: Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Step 6: Installing basic requirements...
pip install fastapi uvicorn[standard] websockets python-multipart

echo.
echo ========================================
echo      PyTorch Installation Options
echo ========================================

echo Running GPU detection...
python python-worker\detect_gpu.py

echo.
echo Please select your PyTorch installation option:
echo.
echo [1] CPU Only (No GPU acceleration)
echo [2] CUDA 12.6 (For NVIDIA RTX 40 series and newer)
echo [3] CUDA 12.8 (For NVIDIA RTX 40 series and newer)  
echo [4] CUDA 12.9 (Latest CUDA - For NVIDIA RTX 40 series and newer)
echo [5] Skip PyTorch installation (I'll install it manually)
echo [A] Auto-install based on detection above
echo.
set /p choice="Enter your choice (1-5, A): "

if "%choice%"=="1" (
    echo.
    echo Installing PyTorch for CPU only...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
) else if "%choice%"=="2" (
    echo.
    echo Installing PyTorch with CUDA 12.6 support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
) else if "%choice%"=="3" (
    echo.
    echo Installing PyTorch with CUDA 12.8 support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
) else if "%choice%"=="4" (
    echo.
    echo Installing PyTorch with CUDA 12.9 support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
) else if "%choice%"=="5" (
    echo.
    echo Skipping PyTorch installation...
    echo You can install it later using one of these commands:
    echo   CPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo   CUDA 12.6: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    echo   CUDA 12.8: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    echo   CUDA 12.9: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
) else if /i "%choice%"=="A" (
    echo.
    echo Auto-installing based on GPU detection...
    for /f %%i in ('python -c "import sys; sys.path.insert(0, 'python-worker'); from detect_gpu import detect_gpu; print(detect_gpu())"') do set gpu_recommendation=%%i
    if "!gpu_recommendation!"=="cuda129" (
        echo Installing PyTorch with CUDA 12.9...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
    ) else if "!gpu_recommendation!"=="cuda128" (
        echo Installing PyTorch with CUDA 12.8...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    ) else if "!gpu_recommendation!"=="cuda126" (
        echo Installing PyTorch with CUDA 12.6...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    ) else (
        echo Installing PyTorch for CPU only...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    )
) else (
    echo Invalid choice. Installing CPU-only version as default...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

echo.
echo Step 8: Installing other requirements...
pip install opencv-python-headless numpy h5py python-dateutil imageio

echo.
echo ========================================
echo        Python Setup Complete!
echo ========================================
echo.
echo To test the backend server:
echo   1. Run: npm run start (to start Electron frontend)
echo   2. In another terminal:
echo      venv\Scripts\activate.bat  
echo      cd python-worker
echo      python main.py
echo.
echo Then test the connection in the VMDify app UI.
echo.
echo Note: This is early development - many features are still being built!
echo.
pause
