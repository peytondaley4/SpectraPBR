#pragma once

#include <string>
#include <vector>

namespace spectra {

// Forward declarations
class SceneManager;
class Camera;

//------------------------------------------------------------------------------
// Scene Serializer - JSON save/load for scene state
//------------------------------------------------------------------------------
class SceneSerializer {
public:
    SceneSerializer() = default;
    ~SceneSerializer() = default;

    // Scene file structure:
    // {
    //   "version": 1,
    //   "camera": { "position": [x,y,z], "yaw": 0, "pitch": 0, "fov": 60 },
    //   "models": [{ "path": "...", "instances": [{ "transform": [...] }] }],
    //   "lights": { "directional": [...], "point": [...], "area": [...] },
    //   "settings": { "qualityMode": 1, "theme": "dark" }
    // }

    // Save scene to JSON file
    bool saveScene(const std::string& filepath,
                   const Camera* camera,
                   const SceneManager* sceneManager,
                   uint32_t qualityMode,
                   bool darkTheme);

    // Load scene from JSON file
    // Returns true if successful
    // Loaded data is stored in member variables, call getters to retrieve
    bool loadScene(const std::string& filepath);

    // Getters for loaded data
    bool hasLoadedCamera() const { return m_hasCameraData; }
    float getCameraPositionX() const { return m_cameraPosition[0]; }
    float getCameraPositionY() const { return m_cameraPosition[1]; }
    float getCameraPositionZ() const { return m_cameraPosition[2]; }
    float getCameraYaw() const { return m_cameraYaw; }
    float getCameraPitch() const { return m_cameraPitch; }
    float getCameraFov() const { return m_cameraFov; }

    uint32_t getQualityMode() const { return m_qualityMode; }
    bool isDarkTheme() const { return m_darkTheme; }

    const std::vector<std::string>& getModelPaths() const { return m_modelPaths; }

    // Get last error message
    const std::string& getLastError() const { return m_lastError; }

    // Auto-save/load path (in user's home directory or app directory)
    static std::string getAutoSavePath();

private:
    // Loaded camera data
    bool m_hasCameraData = false;
    float m_cameraPosition[3] = {0.0f, 1.0f, 5.0f};
    float m_cameraYaw = 0.0f;
    float m_cameraPitch = 0.0f;
    float m_cameraFov = 60.0f;

    // Loaded settings
    uint32_t m_qualityMode = 1;
    bool m_darkTheme = true;

    // Loaded model paths
    std::vector<std::string> m_modelPaths;

    // Error handling
    std::string m_lastError;
};

} // namespace spectra
