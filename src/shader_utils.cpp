#include "shader_utils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

namespace spectra {

std::string readFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file) {
        std::cerr << "[Shader] Failed to open file: " << path << "\n";
        return "";
    }

    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (!success) {
        GLint logLength;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);

        std::vector<char> log(logLength);
        glGetShaderInfoLog(shader, logLength, nullptr, log.data());

        const char* typeStr = (type == GL_VERTEX_SHADER) ? "vertex" :
                              (type == GL_FRAGMENT_SHADER) ? "fragment" : "unknown";
        std::cerr << "[Shader] Failed to compile " << typeStr << " shader:\n"
                  << log.data() << "\n";

        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

GLuint compileShaderFromFile(GLenum type, const std::filesystem::path& path) {
    std::string source = readFile(path);
    if (source.empty()) {
        return 0;
    }

    GLuint shader = compileShader(type, source.c_str());
    if (shader == 0) {
        std::cerr << "[Shader] File: " << path << "\n";
    }
    return shader;
}

GLuint linkProgram(GLuint vertexShader, GLuint fragmentShader) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (!success) {
        GLint logLength;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);

        std::vector<char> log(logLength);
        glGetProgramInfoLog(program, logLength, nullptr, log.data());

        std::cerr << "[Shader] Failed to link program:\n" << log.data() << "\n";

        glDeleteProgram(program);
        program = 0;
    }

    // Detach and delete shaders regardless of link success
    glDetachShader(program, vertexShader);
    glDetachShader(program, fragmentShader);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

GLuint createProgramFromFiles(const std::filesystem::path& vertPath,
                               const std::filesystem::path& fragPath) {
    GLuint vertShader = compileShaderFromFile(GL_VERTEX_SHADER, vertPath);
    if (vertShader == 0) {
        return 0;
    }

    GLuint fragShader = compileShaderFromFile(GL_FRAGMENT_SHADER, fragPath);
    if (fragShader == 0) {
        glDeleteShader(vertShader);
        return 0;
    }

    return linkProgram(vertShader, fragShader);
}

} // namespace spectra
