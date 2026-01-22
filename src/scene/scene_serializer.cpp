#include "scene_serializer.h"
#include "camera.h"
#include "scene_manager.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>

namespace spectra {

using json = nlohmann::json;

bool SceneSerializer::saveScene(const std::string& filepath,
                                 const Camera* camera,
                                 const SceneManager* sceneManager,
                                 uint32_t qualityMode,
                                 bool darkTheme) {
    try {
        json root;
        root["version"] = 1;

        // Camera
        if (camera) {
            json cameraJson;
            glm::vec3 pos = camera->getPosition();
            cameraJson["position"] = { pos.x, pos.y, pos.z };
            cameraJson["yaw"] = camera->getYaw();
            cameraJson["pitch"] = camera->getPitch();
            cameraJson["fov"] = camera->getFOV();
            root["camera"] = cameraJson;
        }

        // Models
        json modelsJson = json::array();
        if (sceneManager) {
            const auto& modelPaths = sceneManager->getLoadedModelPaths();
            for (const auto& path : modelPaths) {
                json modelJson;
                modelJson["path"] = path;
                modelsJson.push_back(modelJson);
            }
        }
        root["models"] = modelsJson;

        // Settings
        json settingsJson;
        settingsJson["qualityMode"] = qualityMode;
        settingsJson["theme"] = darkTheme ? "dark" : "light";
        root["settings"] = settingsJson;

        // Write to file
        std::ofstream file(filepath);
        if (!file.is_open()) {
            m_lastError = "Failed to open file for writing: " + filepath;
            return false;
        }

        file << root.dump(2);  // Pretty print with 2-space indent
        file.close();

        std::cout << "[SceneSerializer] Saved scene to: " << filepath << "\n";
        return true;

    } catch (const std::exception& e) {
        m_lastError = std::string("Exception while saving: ") + e.what();
        std::cerr << "[SceneSerializer] " << m_lastError << "\n";
        return false;
    }
}

bool SceneSerializer::loadScene(const std::string& filepath) {
    // Reset state
    m_hasCameraData = false;
    m_modelPaths.clear();
    m_qualityMode = 1;
    m_darkTheme = true;

    try {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            m_lastError = "Failed to open file: " + filepath;
            return false;
        }

        json root = json::parse(file);
        file.close();

        // Version check
        int version = root.value("version", 0);
        if (version < 1) {
            m_lastError = "Unsupported scene file version";
            return false;
        }

        // Load camera
        if (root.contains("camera")) {
            const auto& cameraJson = root["camera"];
            if (cameraJson.contains("position") && cameraJson["position"].is_array()) {
                const auto& pos = cameraJson["position"];
                if (pos.size() >= 3) {
                    m_cameraPosition[0] = pos[0].get<float>();
                    m_cameraPosition[1] = pos[1].get<float>();
                    m_cameraPosition[2] = pos[2].get<float>();
                }
            }
            m_cameraYaw = cameraJson.value("yaw", 0.0f);
            m_cameraPitch = cameraJson.value("pitch", 0.0f);
            m_cameraFov = cameraJson.value("fov", 60.0f);
            m_hasCameraData = true;
        }

        // Load models
        if (root.contains("models") && root["models"].is_array()) {
            for (const auto& modelJson : root["models"]) {
                if (modelJson.contains("path")) {
                    m_modelPaths.push_back(modelJson["path"].get<std::string>());
                }
            }
        }

        // Load settings
        if (root.contains("settings")) {
            const auto& settings = root["settings"];
            m_qualityMode = settings.value("qualityMode", 1u);
            std::string theme = settings.value("theme", "dark");
            m_darkTheme = (theme == "dark");
        }

        std::cout << "[SceneSerializer] Loaded scene from: " << filepath << "\n";
        std::cout << "[SceneSerializer] Models: " << m_modelPaths.size()
                  << ", Quality: " << m_qualityMode
                  << ", Theme: " << (m_darkTheme ? "dark" : "light") << "\n";

        return true;

    } catch (const json::parse_error& e) {
        m_lastError = std::string("JSON parse error: ") + e.what();
        std::cerr << "[SceneSerializer] " << m_lastError << "\n";
        return false;
    } catch (const std::exception& e) {
        m_lastError = std::string("Exception while loading: ") + e.what();
        std::cerr << "[SceneSerializer] " << m_lastError << "\n";
        return false;
    }
}

std::string SceneSerializer::getAutoSavePath() {
    // Try to get user's home directory
    const char* homeDir = std::getenv("HOME");
    if (!homeDir) {
        homeDir = std::getenv("USERPROFILE");  // Windows fallback
    }

    if (homeDir) {
        std::filesystem::path path(homeDir);
        path /= ".spectrapbr";

        // Create directory if it doesn't exist
        if (!std::filesystem::exists(path)) {
            std::filesystem::create_directories(path);
        }

        path /= "autosave.json";
        return path.string();
    }

    // Fallback to current directory
    return "spectrapbr_autosave.json";
}

} // namespace spectra
