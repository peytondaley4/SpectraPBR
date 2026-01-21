# FindOptiX.cmake
# Finds the NVIDIA OptiX SDK
#
# This module defines:
#   OptiX_FOUND        - True if OptiX was found
#   OptiX_INCLUDE_DIRS - OptiX include directories
#   OptiX_VERSION      - OptiX version string (if available)
#
# The following paths are searched:
#   - OptiX_INSTALL_DIR environment variable
#   - OPTIX_PATH environment variable
#   - Common installation paths on Windows/Linux

# Check environment variables first
set(_optix_search_paths)

if(DEFINED ENV{OptiX_INSTALL_DIR})
    list(APPEND _optix_search_paths "$ENV{OptiX_INSTALL_DIR}")
endif()

if(DEFINED ENV{OPTIX_PATH})
    list(APPEND _optix_search_paths "$ENV{OPTIX_PATH}")
endif()

# Add common installation paths
if(WIN32)
    # Windows default paths
    file(GLOB _optix_win_paths "C:/ProgramData/NVIDIA Corporation/OptiX SDK *")
    list(APPEND _optix_search_paths ${_optix_win_paths})
    list(APPEND _optix_search_paths
        "C:/Program Files/NVIDIA GPU Computing Toolkit/OptiX"
        "$ENV{PROGRAMFILES}/NVIDIA GPU Computing Toolkit/OptiX"
    )
else()
    # Linux default paths
    list(APPEND _optix_search_paths
        "/opt/optix"
        "/usr/local/optix"
        "$ENV{HOME}/NVIDIA-OptiX-SDK"
    )
    # Check for versioned directories
    file(GLOB _optix_linux_paths "$ENV{HOME}/NVIDIA-OptiX-SDK-*")
    list(APPEND _optix_search_paths ${_optix_linux_paths})
endif()

# Find optix.h
find_path(OptiX_INCLUDE_DIR
    NAMES optix.h
    PATHS ${_optix_search_paths}
    PATH_SUFFIXES include
)

# Extract version from optix.h if found
if(OptiX_INCLUDE_DIR AND EXISTS "${OptiX_INCLUDE_DIR}/optix.h")
    file(STRINGS "${OptiX_INCLUDE_DIR}/optix.h" _optix_version_line
         REGEX "^#define OPTIX_VERSION [0-9]+")
    if(_optix_version_line)
        string(REGEX REPLACE "^#define OPTIX_VERSION ([0-9]+).*$" "\\1"
               _optix_version_num "${_optix_version_line}")
        # Version format: major * 10000 + minor * 100 + micro
        math(EXPR OptiX_VERSION_MAJOR "${_optix_version_num} / 10000")
        math(EXPR OptiX_VERSION_MINOR "(${_optix_version_num} % 10000) / 100")
        math(EXPR OptiX_VERSION_MICRO "${_optix_version_num} % 100")
        set(OptiX_VERSION "${OptiX_VERSION_MAJOR}.${OptiX_VERSION_MINOR}.${OptiX_VERSION_MICRO}")
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiX
    REQUIRED_VARS OptiX_INCLUDE_DIR
    VERSION_VAR OptiX_VERSION
)

if(OptiX_FOUND)
    set(OptiX_INCLUDE_DIRS ${OptiX_INCLUDE_DIR})

    if(NOT TARGET OptiX::OptiX)
        add_library(OptiX::OptiX INTERFACE IMPORTED)
        set_target_properties(OptiX::OptiX PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${OptiX_INCLUDE_DIRS}"
        )
    endif()
endif()

mark_as_advanced(OptiX_INCLUDE_DIR)
