import { app, BrowserWindow, ipcMain, dialog } from 'electron';
import path from 'node:path';
import { spawn } from 'node:child_process';
import started from 'electron-squirrel-startup';

// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (started) {
  app.quit();
}

let pythonProcess = null;
let mainWindow = null;

const createWindow = () => {
  // Create the browser window.
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    title: 'VMDify - AI Motion Capture',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  // and load the index.html of the app.
  if (MAIN_WINDOW_VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(MAIN_WINDOW_VITE_DEV_SERVER_URL);
  } else {
    mainWindow.loadFile(path.join(__dirname, `../renderer/${MAIN_WINDOW_VITE_NAME}/index.html`));
  }

  // Only open DevTools in development
  if (MAIN_WINDOW_VITE_DEV_SERVER_URL) {
    mainWindow.webContents.openDevTools();
  }
};

const startPythonBackend = () => {
  try {
    // Determine Python backend path
    let pythonWorkerPath;
    let pythonExe;
    
    if (app.isPackaged) {
      // In production, Python backend is in extraResources
      pythonWorkerPath = path.join(process.resourcesPath, 'python-worker');
      pythonExe = path.join(process.resourcesPath, 'venv', 'Scripts', 'python.exe');
    } else {
      // In development, use relative paths
      pythonWorkerPath = path.join(__dirname, '..', '..', 'python-worker');
      pythonExe = path.join(__dirname, '..', '..', 'venv', 'Scripts', 'python.exe');
    }

    const mainPy = path.join(pythonWorkerPath, 'main.py');

    console.log('Starting Python backend:', pythonExe, mainPy);

    // Start Python process
    pythonProcess = spawn(pythonExe, [mainPy], {
      cwd: pythonWorkerPath,
      stdio: ['pipe', 'pipe', 'pipe']
    });

    pythonProcess.stdout.on('data', (data) => {
      console.log('Python stdout:', data.toString());
    });

    pythonProcess.stderr.on('data', (data) => {
      console.log('Python stderr:', data.toString());
    });

    pythonProcess.on('close', (code) => {
      console.log(`Python process exited with code ${code}`);
    });

    pythonProcess.on('error', (err) => {
      console.error('Failed to start Python process:', err);
      
      // Show error dialog
      if (mainWindow) {
        dialog.showErrorBox(
          'Python Backend Error',
          `Failed to start the AI backend. Please ensure Python is properly installed.\n\nError: ${err.message}`
        );
      }
    });

  } catch (error) {
    console.error('Error starting Python backend:', error);
  }
};

// IPC Handlers for file dialogs and system operations
ipcMain.handle('show-open-dialog', async (event, options) => {
  const result = await dialog.showOpenDialog(mainWindow, options);
  return result;
});

ipcMain.handle('show-save-dialog', async (event, options) => {
  const result = await dialog.showSaveDialog(mainWindow, options);
  return result;
});

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {
  createWindow();
  
  // Start Python backend after a short delay
  setTimeout(startPythonBackend, 2000);

  // On OS X it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// Quit when all windows are closed, except on macOS.
app.on('window-all-closed', () => {
  // Kill Python process when app closes
  if (pythonProcess) {
    pythonProcess.kill();
  }
  
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// Ensure Python process is killed when app quits
app.on('before-quit', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
});

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and import them here.
