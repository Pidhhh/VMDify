const { FusesPlugin } = require('@electron-forge/plugin-fuses');
const { FuseV1Options, FuseVersion } = require('@electron/fuses');

module.exports = {
  packagerConfig: {
    asar: true,
    name: "VMDify",
    executableName: "VMDify",
    icon: "./assets/icon", // We'll need to add an icon later
    extraResource: [
      "./python-worker" // Include Python backend in the package
    ],
    ignore: [
      /\.git/,
      /node_modules\/.*\/test/,
      /\.vscode/,
      /\.github/,
      /python-worker\/venv/, // Exclude Python virtual environment
      /python-worker\/__pycache__/,
      /python-worker\/\.pytest_cache/
    ]
  },
  rebuildConfig: {},
  makers: [
    {
      name: '@electron-forge/maker-squirrel',
      config: {
        name: 'VMDify',
        setupExe: 'VMDify-Setup.exe',
        setupIcon: './assets/icon.ico', // We'll add this later
        noMsi: true,
        certificateFile: '', // Add code signing certificate if available
        certificatePassword: process.env.CERTIFICATE_PASSWORD || ''
      },
    },
    {
      name: '@electron-forge/maker-zip',
      platforms: ['darwin'],
      config: {
        name: 'VMDify'
      }
    },
    {
      name: '@electron-forge/maker-deb',
      config: {
        options: {
          name: 'vmdify',
          productName: 'VMDify',
          description: 'AI-powered video to MMD motion capture application',
          maintainer: 'Pidhhh',
          homepage: 'https://github.com/Pidhhh/VMDify'
        }
      },
    },
    {
      name: '@electron-forge/maker-rpm',
      config: {
        options: {
          name: 'vmdify',
          productName: 'VMDify',
          description: 'AI-powered video to MMD motion capture application',
          maintainer: 'Pidhhh',
          homepage: 'https://github.com/Pidhhh/VMDify'
        }
      },
    },
  ],
  plugins: [
    {
      name: '@electron-forge/plugin-vite',
      config: {
        // `build` can specify multiple entry builds, which can be Main process, Preload scripts, Worker process, etc.
        // If you are familiar with Vite configuration, it will look really familiar.
        build: [
          {
            // `entry` is just an alias for `build.lib.entry` in the corresponding file of `config`.
            entry: 'src/main.js',
            config: 'vite.main.config.mjs',
            target: 'main',
          },
          {
            entry: 'src/preload.js',
            config: 'vite.preload.config.mjs',
            target: 'preload',
          },
        ],
        renderer: [
          {
            name: 'main_window',
            config: 'vite.renderer.config.mjs',
          },
        ],
      },
    },
    // Fuses are used to enable/disable various Electron functionality
    // at package time, before code signing the application
    new FusesPlugin({
      version: FuseVersion.V1,
      [FuseV1Options.RunAsNode]: false,
      [FuseV1Options.EnableCookieEncryption]: true,
      [FuseV1Options.EnableNodeOptionsEnvironmentVariable]: false,
      [FuseV1Options.EnableNodeCliInspectArguments]: false,
      [FuseV1Options.EnableEmbeddedAsarIntegrityValidation]: true,
      [FuseV1Options.OnlyLoadAppFromAsar]: true,
    }),
  ],
};
