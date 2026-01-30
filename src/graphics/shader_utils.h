#pragma once

#include <glad/glad.h>
#include <string>
#include <filesystem>

namespace spectra {

// Compile a shader from source code
// Returns shader ID on success, 0 on failure
GLuint compileShader(GLenum type, const char* source);

// Compile a shader from file
// Returns shader ID on success, 0 on failure
GLuint compileShaderFromFile(GLenum type, const std::filesystem::path& path);

// Link shaders into a program
// Returns program ID on success, 0 on failure
// Shaders are detached and deleted after linking
GLuint linkProgram(GLuint vertexShader, GLuint fragmentShader);

// Load and create a complete shader program from files
// Returns program ID on success, 0 on failure
GLuint createProgramFromFiles(const std::filesystem::path& vertPath,
                               const std::filesystem::path& fragPath);

// Read entire file contents into string
// Returns empty string on failure
std::string readFile(const std::filesystem::path& path);

} // namespace spectra
