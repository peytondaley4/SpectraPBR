# SpectraPBR

A real-time physically-based rendering application using NVIDIA OptiX for ray tracing, CUDA for GPU computation, and OpenGL for display.

## Prerequisites

Before building, ensure you have the following installed:

1. **CMake** (version 3.18 or higher)
   - Download from: https://cmake.org/download/
   - Or install via: `winget install Kitware.CMake`

2. **CUDA Toolkit** (latest version recommended)
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Ensure CUDA is in your PATH

3. **NVIDIA OptiX SDK** (version 7.0 or higher)
   - Download from: https://developer.nvidia.com/designworks/optix
   - Extract to a location (e.g., `C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.x`)
   - Set one of these environment variables:
     - `OptiX_INSTALL_DIR` - Point to the OptiX SDK root directory
     - `OPTIX_PATH` - Alternative environment variable name
   - Or install to default location: `C:\ProgramData\NVIDIA Corporation\OptiX SDK *`

4. **Visual Studio** (2019 or later) with C++ support
   - Community edition is sufficient
   - Ensure "Desktop development with C++" workload is installed

5. **NVIDIA GPU** with RTX support (Turing, Ampere, or Ada Lovelace architecture)
   - RTX 20-series, 30-series, 40-series, or compatible

## Building the Project

### Step 1: Configure with CMake

Open PowerShell or Command Prompt in the project root directory and run:

```powershell
# Create a build directory
mkdir build
cd build

# Configure the project (Release build by default)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Or for Debug build:
# cmake .. -DCMAKE_BUILD_TYPE=Debug
```

**Note:** If CMake cannot find OptiX, you may need to set the environment variable:
```powershell
$env:OptiX_INSTALL_DIR = "C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.x"
cmake ..
```

### Step 2: Build the Project

```powershell
# Build the project
cmake --build . --config Release

# Or using Visual Studio:
# cmake --build . --config Release --parallel
```

Alternatively, you can open the generated `SpectraPBR.sln` file in Visual Studio and build from there.

### Step 3: Verify Build Output

After building, you should find:
- `build/Release/SpectraPBR.exe` (or `build/Debug/SpectraPBR.exe` for Debug)
- `build/Release/shaders/` directory with shader files
- `build/Release/optix_programs/` directory with `.ptx` files

## Running and Testing

### Basic Execution

From the build directory:

```powershell
# Run from Release build directory
cd Release
.\SpectraPBR.exe
```

Or from the project root:

```powershell
# Run from project root (if shaders and PTX files are in source directory)
cd build/Release
.\SpectraPBR.exe
```

### Controls

Once the application is running, you can use the following controls:

- **ESC** - Quit the application
- **V** - Toggle VSync on/off
- **F** - Toggle Fullscreen mode
- **1** - Switch to 720p resolution (1280x720)
- **2** - Switch to 1080p resolution (1920x1080)
- **3** - Switch to 1440p resolution (2560x1440)
- **4** - Switch to 4K resolution (3840x2160)
- **T** - Print timing information (frame time and FPS)
- **G** - Print GPU information and memory usage

### Testing Checklist

1. **Initialization Test**
   - Launch the application
   - Verify the window opens without errors
   - Check console output for successful initialization messages

2. **Rendering Test**
   - Verify the window displays rendered content
   - Check that frames are being rendered (watch console for any errors)

3. **Performance Test**
   - Press **T** to view frame timing and FPS
   - Verify reasonable frame rates (should be > 30 FPS on modern hardware)
   - Test different resolutions (1-4 keys) and verify performance scales appropriately

4. **GPU Information Test**
   - Press **G** to view GPU device information
   - Verify CUDA device is detected correctly
   - Check memory usage statistics

5. **Window Management Test**
   - Test fullscreen toggle (F key)
   - Test VSync toggle (V key)
   - Resize the window and verify rendering adapts

6. **Stress Test**
   - Run the application for several minutes
   - Monitor for memory leaks or crashes
   - Check console for any warnings or errors

### Troubleshooting

**CMake cannot find OptiX:**
- Set the `OptiX_INSTALL_DIR` environment variable to your OptiX SDK path
- Or ensure OptiX is installed in the default location: `C:\ProgramData\NVIDIA Corporation\OptiX SDK *`

**CUDA not found:**
- Ensure CUDA Toolkit is installed and in your PATH
- Verify with: `nvcc --version`

**Shader files not found:**
- Ensure shaders are copied to the build directory (CMake should do this automatically)
- Check that `shaders/display.vert` and `shaders/display.frag` exist in the executable directory

**PTX files not found:**
- Ensure OptiX programs compiled successfully
- Check that `optix_programs/raygen.ptx` and `optix_programs/miss.ptx` exist in the executable directory

**Application crashes on startup:**
- Verify your GPU supports OptiX (RTX series or compatible)
- Check that NVIDIA drivers are up to date
- Run in Debug mode to see detailed error messages

**Low performance:**
- Check GPU utilization with NVIDIA System Monitor or Task Manager
- Verify CUDA architecture matches your GPU (see CMakeLists.txt line 35)
- Try different resolutions to find optimal performance

## Project Structure

```
SpectraPBR/
├── CMakeLists.txt          # Main build configuration
├── cmake/
│   └── FindOptiX.cmake     # OptiX SDK finder module
├── src/                    # C++ source files
│   ├── main.cpp           # Application entry point
│   ├── gl_context.cpp     # OpenGL context management
│   ├── cuda_interop.cpp   # CUDA-OpenGL interop
│   ├── optix_engine.cpp   # OptiX ray tracing engine
│   └── shader_utils.cpp   # Shader loading utilities
├── optix_programs/         # OptiX CUDA programs
│   ├── raygen.cu          # Ray generation program
│   └── miss.cu            # Miss program
└── shaders/                # OpenGL shaders
    ├── display.vert        # Vertex shader
    └── display.frag        # Fragment shader
```

## Development Notes

- The project uses C++17 standard
- CUDA programs are compiled to PTX for OptiX
- Default CUDA architectures: 75 (Turing), 86 (Ampere), 89 (Ada Lovelace)
- Build type defaults to Release if not specified

## License

[Add your license information here]
