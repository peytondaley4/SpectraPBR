#include "camera.h"
#include <glm/gtc/constants.hpp>
#include <algorithm>
#include <cmath>

namespace spectra {

Camera::Camera()
    : m_position(0.0f, 0.0f, 5.0f)
    , m_yaw(-90.0f)          // Looking along -Z
    , m_pitch(0.0f)
    , m_forward(0.0f, 0.0f, -1.0f)
    , m_right(1.0f, 0.0f, 0.0f)
    , m_up(0.0f, 1.0f, 0.0f)
    , m_fovDegrees(60.0f)
    , m_aspectRatio(16.0f / 9.0f)
    , m_nearPlane(0.01f)
    , m_farPlane(1000.0f)
    , m_moveSpeed(5.0f)
    , m_sprintMultiplier(3.0f)
    , m_mouseSensitivity(0.1f)
{
    updateVectors();
}

void Camera::setPosition(const glm::vec3& pos) {
    m_position = pos;
}

void Camera::setYawPitch(float yaw, float pitch) {
    m_yaw = yaw;
    m_pitch = std::clamp(pitch, MIN_PITCH, MAX_PITCH);
    updateVectors();
}

void Camera::setFOV(float fovDegrees) {
    m_fovDegrees = std::clamp(fovDegrees, MIN_FOV, MAX_FOV);
}

void Camera::setAspectRatio(float aspect) {
    m_aspectRatio = aspect;
}

void Camera::setClipPlanes(float nearPlane, float farPlane) {
    m_nearPlane = nearPlane;
    m_farPlane = farPlane;
}

void Camera::processKeyboard(float forward, float right, float up, float deltaTime, bool sprint) {
    float speed = m_moveSpeed * deltaTime;
    if (sprint) {
        speed *= m_sprintMultiplier;
    }

    // Move relative to camera orientation (XZ plane for forward/right, world Y for up/down)
    glm::vec3 moveForward = glm::normalize(glm::vec3(m_forward.x, 0.0f, m_forward.z));
    glm::vec3 moveRight = glm::normalize(glm::vec3(m_right.x, 0.0f, m_right.z));

    m_position += moveForward * forward * speed;
    m_position += moveRight * right * speed;
    m_position.y += up * speed;
}

void Camera::processMouseMovement(float deltaX, float deltaY) {
    m_yaw += deltaX * m_mouseSensitivity;
    m_pitch -= deltaY * m_mouseSensitivity;  // Inverted for natural feel

    // Clamp pitch to prevent gimbal lock
    m_pitch = std::clamp(m_pitch, MIN_PITCH, MAX_PITCH);

    // Keep yaw in reasonable range
    if (m_yaw > 360.0f) m_yaw -= 360.0f;
    if (m_yaw < -360.0f) m_yaw += 360.0f;

    updateVectors();
}

void Camera::processMouseScroll(float deltaY) {
    m_fovDegrees -= deltaY * 2.0f;
    m_fovDegrees = std::clamp(m_fovDegrees, MIN_FOV, MAX_FOV);
}

void Camera::updateVectors() {
    // Calculate forward direction from yaw and pitch
    float yawRad = glm::radians(m_yaw);
    float pitchRad = glm::radians(m_pitch);

    m_forward.x = std::cos(yawRad) * std::cos(pitchRad);
    m_forward.y = std::sin(pitchRad);
    m_forward.z = std::sin(yawRad) * std::cos(pitchRad);
    m_forward = glm::normalize(m_forward);

    // Recalculate right and up
    m_right = glm::normalize(glm::cross(m_forward, WORLD_UP));
    m_up = glm::normalize(glm::cross(m_right, m_forward));
}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(m_position, m_position + m_forward, m_up);
}

glm::mat4 Camera::getProjectionMatrix() const {
    return glm::perspective(glm::radians(m_fovDegrees), m_aspectRatio, m_nearPlane, m_farPlane);
}

CameraParams Camera::getCameraParams() const {
    CameraParams params = {};

    params.position = make_float3(m_position.x, m_position.y, m_position.z);
    params.forward = make_float3(m_forward.x, m_forward.y, m_forward.z);
    params.right = make_float3(m_right.x, m_right.y, m_right.z);
    params.up = make_float3(m_up.x, m_up.y, m_up.z);

    params.fovY = glm::radians(m_fovDegrees);
    params.aspectRatio = m_aspectRatio;
    params.nearPlane = m_nearPlane;
    params.farPlane = m_farPlane;

    return params;
}

} // namespace spectra
