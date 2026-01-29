cmake_minimum_required(VERSION 3.18)

# Enable CUDA language
project(SpectraPBR LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add cmake modules path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Build type defaults
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Debug definitions
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(_DEBUG)
endif()

#------------------------------------------------------------------------------
# CUDA Configuration
#------------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Include: ${CUDAToolkit_INCLUDE_DIRS}")

# Set CUDA architectures (RTX cards: 75=Turing, 86=Ampere, 89=Ada Lovelace)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
endif()
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Enable separable compilation for OptiX
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

#------------------------------------------------------------------------------
# OptiX Configuration
#------------------------------------------------------------------------------
find_package(OptiX REQUIRED)
message(STATUS "OptiX Version: ${OptiX_VERSION}")
message(STATUS "OptiX Include: ${OptiX_INCLUDE_DIRS}")

#------------------------------------------------------------------------------
# OpenGL and GLFW
#------------------------------------------------------------------------------
find_package(OpenGL REQUIRED)
message(STATUS "OpenGL Found: ${OPENGL_FOUND}")

# GLFW - try to find system package, fall back to FetchContent
find_package(glfw3 3.3 QUIET)
if(NOT glfw3_FOUND)
    message(STATUS "GLFW not found, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG        3.3.8
    )
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(glfw)
else()
    message(STATUS "GLFW Found: ${glfw3_VERSION}")
endif()

#------------------------------------------------------------------------------
# GLAD - OpenGL loader
#------------------------------------------------------------------------------
# We'll use FetchContent to get GLAD
include(FetchContent)
FetchContent_Declare(
    glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG        v0.1.36
)
FetchContent_GetProperties(glad)
if(NOT glad_POPULATED)
    FetchContent_Populate(glad)

    # GLAD needs to be configured for OpenGL 4.5 Core
    set(GLAD_PROFILE "core" CACHE STRING "" FORCE)
    set(GLAD_API "gl=4.5" CACHE STRING "" FORCE)
    set(GLAD_GENERATOR "c" CACHE STRING "" FORCE)
    set(GLAD_EXTENSIONS "" CACHE STRING "" FORCE)
    set(GLAD_SPEC "gl" CACHE STRING "" FORCE)
    set(GLAD_NO_LOADER OFF CACHE BOOL "" FORCE)

    add_subdirectory(${glad_SOURCE_DIR} ${glad_BINARY_DIR})
endif()

#------------------------------------------------------------------------------
# GLM - Math library (header only)
#------------------------------------------------------------------------------
find_package(glm QUIET)
if(NOT glm_FOUND)
    message(STATUS "GLM not found, fetching from GitHub...")
    FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG        0.9.9.8
    )
    FetchContent_MakeAvailable(glm)
else()
    message(STATUS "GLM Found")
endif()

#------------------------------------------------------------------------------
# tinygltf - glTF 2.0 loader (header only)
#------------------------------------------------------------------------------
FetchContent_Declare(
    tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG        v2.8.18
)
FetchContent_MakeAvailable(tinygltf)
message(STATUS "tinygltf fetched")

#------------------------------------------------------------------------------
# nlohmann/json - JSON library for scene serialization
#------------------------------------------------------------------------------
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(json)
message(STATUS "nlohmann/json fetched")

#------------------------------------------------------------------------------
# stb - Single-file public domain libraries (stb_truetype for font rendering)
#------------------------------------------------------------------------------
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
)
FetchContent_GetProperties(stb)
if(NOT stb_POPULATED)
    FetchContent_Populate(stb)
endif()
message(STATUS "stb fetched: ${stb_SOURCE_DIR}")

#------------------------------------------------------------------------------
# OptiX PTX Compilation
#------------------------------------------------------------------------------
# Function to compile CUDA files to PTX for OptiX
function(compile_optix_ptx target_name cuda_file output_ptx)
    get_filename_component(cuda_file_abs ${cuda_file} ABSOLUTE)
    get_filename_component(cuda_file_name ${cuda_file} NAME_WE)

    set(ptx_file "${CMAKE_BINARY_DIR}/optix_programs/${cuda_file_name}.ptx")

    # Determine include paths
    set(include_flags
        -I${OptiX_INCLUDE_DIRS}
        -I${CUDAToolkit_INCLUDE_DIRS}
        -I${CMAKE_CURRENT_SOURCE_DIR}/src
        -I${CMAKE_CURRENT_SOURCE_DIR}/optix_programs
    )

    # NVCC flags for PTX compilation
    set(nvcc_flags
        --ptx
        -O3
        --use_fast_math
        -lineinfo
        --machine=64
    )

    # Add architecture flags
    foreach(arch ${CMAKE_CUDA_ARCHITECTURES})
        list(APPEND nvcc_flags --gpu-architecture=compute_${arch})
    endforeach()

    # Use the first architecture for PTX output (PTX is architecture-agnostic)
    list(GET CMAKE_CUDA_ARCHITECTURES 0 first_arch)

    # Get first include directory from each (they may be lists)
    list(GET OptiX_INCLUDE_DIRS 0 optix_include)
    list(GET CUDAToolkit_INCLUDE_DIRS 0 cuda_include)

    add_custom_command(
        OUTPUT ${ptx_file}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/optix_programs"
        COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
                --ptx
                -O3
                --use_fast_math
                -lineinfo
                --machine=64
                --gpu-architecture=compute_${first_arch}
                -I${optix_include}
                -I${cuda_include}
                -o
                ${ptx_file}
                ${cuda_file_abs}
        DEPENDS ${cuda_file_abs}
        COMMENT "Compiling OptiX PTX: ${cuda_file_name}.cu -> ${cuda_file_name}.ptx"
        VERBATIM
    )

    set(${output_ptx} ${ptx_file} PARENT_SCOPE)
endfunction()

# Compile OptiX programs
compile_optix_ptx(spectra_ptx
    "${CMAKE_CURRENT_SOURCE_DIR}/optix_programs/raygen.cu"
    RAYGEN_PTX
)

compile_optix_ptx(spectra_ptx
    "${CMAKE_CURRENT_SOURCE_DIR}/optix_programs/miss.cu"
    MISS_PTX
)

compile_optix_ptx(spectra_ptx
    "${CMAKE_CURRENT_SOURCE_DIR}/optix_programs/closesthit.cu"
    CLOSESTHIT_PTX
)

compile_optix_ptx(spectra_ptx
    "${CMAKE_CURRENT_SOURCE_DIR}/optix_programs/anyhit.cu"
    ANYHIT_PTX
)

# Custom target for PTX files
add_custom_target(optix_ptx ALL DEPENDS ${RAYGEN_PTX} ${MISS_PTX} ${CLOSESTHIT_PTX} ${ANYHIT_PTX})

#------------------------------------------------------------------------------
# Main Executable
#------------------------------------------------------------------------------
add_executable(SpectraPBR
    src/main.cpp
    src/gl_context.cpp
    src/cuda_interop.cpp
    src/optix_engine.cpp
    src/shader_utils.cpp
    src/camera.cpp
    src/model_loader.cpp
    src/geometry_manager.cpp
    src/texture_manager.cpp
    src/material_manager.cpp
    src/scene_manager.cpp
    # Phase 4: UI System
    src/text/font_atlas.cpp
    src/text/text_layout.cpp
    src/ui/widget.cpp
    src/ui/panel.cpp
    src/ui/button.cpp
    src/ui/label.cpp
    src/ui/tree_node.cpp
    src/ui/scroll_view.cpp
    src/ui/slider.cpp
    src/ui/color_picker.cpp
    src/ui/property_panel.cpp
    src/ui/input_handler.cpp
    src/ui/ui_manager.cpp
    src/ui/ui_renderer.cu
    src/ui/texture_preview_cache.cu
    # Phase 4: Scene Management
    src/scene/selection_manager.cpp
    src/scene/scene_serializer.cpp
    src/scene/scene_hierarchy.cpp
)

# Include directories
target_include_directories(SpectraPBR PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OptiX_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${tinygltf_SOURCE_DIR}
    ${stb_SOURCE_DIR}
)

# Link libraries
target_link_libraries(SpectraPBR PRIVATE
    OpenGL::GL
    glfw
    glad
    glm::glm
    CUDA::cudart
    CUDA::cuda_driver
    nlohmann_json::nlohmann_json
)

# Ensure PTX is built before the main target
add_dependencies(SpectraPBR optix_ptx)

#------------------------------------------------------------------------------
# Copy resources to build directory
#------------------------------------------------------------------------------
# Copy shaders
add_custom_command(TARGET SpectraPBR POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:SpectraPBR>/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/display.vert"
            "$<TARGET_FILE_DIR:SpectraPBR>/shaders/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/display.frag"
            "$<TARGET_FILE_DIR:SpectraPBR>/shaders/"
    COMMENT "Copying shaders to build directory"
)

# Copy PTX files
add_custom_command(TARGET SpectraPBR POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:SpectraPBR>/optix_programs"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/optix_programs/raygen.ptx"
            "$<TARGET_FILE_DIR:SpectraPBR>/optix_programs/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/optix_programs/miss.ptx"
            "$<TARGET_FILE_DIR:SpectraPBR>/optix_programs/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/optix_programs/closesthit.ptx"
            "$<TARGET_FILE_DIR:SpectraPBR>/optix_programs/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/optix_programs/anyhit.ptx"
            "$<TARGET_FILE_DIR:SpectraPBR>/optix_programs/"
    COMMENT "Copying PTX files to build directory"
)

# Copy fonts for UI
add_custom_command(TARGET SpectraPBR POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:SpectraPBR>/assets/fonts"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/assets/fonts"
            "$<TARGET_FILE_DIR:SpectraPBR>/assets/fonts"
    COMMENT "Copying fonts to build directory"
)

#------------------------------------------------------------------------------
# Installation
#------------------------------------------------------------------------------
install(TARGETS SpectraPBR RUNTIME DESTINATION bin)
install(DIRECTORY shaders/ DESTINATION bin/shaders)
install(FILES
    ${CMAKE_BINARY_DIR}/optix_programs/raygen.ptx
    ${CMAKE_BINARY_DIR}/optix_programs/miss.ptx
    ${CMAKE_BINARY_DIR}/optix_programs/closesthit.ptx
    ${CMAKE_BINARY_DIR}/optix_programs/anyhit.ptx
    DESTINATION bin/optix_programs
)

#------------------------------------------------------------------------------
# Summary
#------------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== SpectraPBR Configuration ===")
message(STATUS "Build Type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Toolkit:     ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Archs:       ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "OptiX Version:    ${OptiX_VERSION}")
message(STATUS "C++ Standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "================================")
message(STATUS "")
