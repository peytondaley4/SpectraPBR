#pragma once

#include "shared_types.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace spectra {

class Camera {
public:
    Camera();
    ~Camera() = default;

    // Set camera position
    void setPosition(const glm::vec3& pos);
    glm::vec3 getPosition() const { return m_position; }

    // Set camera orientation (yaw = horizontal, pitch = vertical)
    void setYawPitch(float yaw, float pitch);
    void setYaw(float yaw) { setYawPitch(yaw, m_pitch); }
    void setPitch(float pitch) { setYawPitch(m_yaw, pitch); }
    float getYaw() const { return m_yaw; }
    float getPitch() const { return m_pitch; }

    // Set field of view (vertical, in degrees)
    void setFOV(float fovDegrees);
    float getFOV() const { return m_fovDegrees; }

    // Set aspect ratio
    void setAspectRatio(float aspect);
    float getAspectRatio() const { return m_aspectRatio; }

    // Set near/far planes
    void setClipPlanes(float nearPlane, float farPlane);

    // Movement (FPS-style controls)
    // Call per-frame with deltaTime in seconds
    void processKeyboard(float forward, float right, float up, float deltaTime, bool sprint);

    // Mouse look
    // deltaX/deltaY are mouse movement in pixels
    void processMouseMovement(float deltaX, float deltaY);

    // Scroll to adjust FOV
    void processMouseScroll(float deltaY);

    // Get direction vectors
    glm::vec3 getForward() const { return m_forward; }
    glm::vec3 getRight() const { return m_right; }
    glm::vec3 getUp() const { return m_up; }

    // Get view matrix
    glm::mat4 getViewMatrix() const;

    // Get projection matrix
    glm::mat4 getProjectionMatrix() const;

    // Fill CameraParams struct for GPU
    CameraParams getCameraParams() const;

    // Movement speed settings
    void setMoveSpeed(float speed) { m_moveSpeed = speed; }
    void setSprintMultiplier(float mult) { m_sprintMultiplier = mult; }
    void setMouseSensitivity(float sens) { m_mouseSensitivity = sens; }

    float getMoveSpeed() const { return m_moveSpeed; }
    float getSprintMultiplier() const { return m_sprintMultiplier; }
    float getMouseSensitivity() const { return m_mouseSensitivity; }

private:
    void updateVectors();

    // Position
    glm::vec3 m_position;

    // Orientation (Euler angles in degrees)
    float m_yaw;      // Horizontal rotation (around Y axis)
    float m_pitch;    // Vertical rotation (around X axis)

    // Direction vectors (computed from yaw/pitch)
    glm::vec3 m_forward;
    glm::vec3 m_right;
    glm::vec3 m_up;

    // World up (constant)
    static constexpr glm::vec3 WORLD_UP = glm::vec3(0.0f, 1.0f, 0.0f);

    // Projection parameters
    float m_fovDegrees;
    float m_aspectRatio;
    float m_nearPlane;
    float m_farPlane;

    // Control settings
    float m_moveSpeed;
    float m_sprintMultiplier;
    float m_mouseSensitivity;

    // Constraints
    static constexpr float MIN_PITCH = -89.0f;
    static constexpr float MAX_PITCH = 89.0f;
    static constexpr float MIN_FOV = 10.0f;
    static constexpr float MAX_FOV = 120.0f;
};

} // namespace spectra
