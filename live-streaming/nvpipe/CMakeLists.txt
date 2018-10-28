# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.2)
project(NvPipe VERSION 1.0.0 LANGUAGES CXX)

SET(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
        STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(CUDA REQUIRED)

set (CMAKE_CXX_STANDARD 11)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11 -g ")

# Options
option(NVPIPE_WITH_ENCODER "Enables the NvPipe encoding interface." ON)
option(NVPIPE_WITH_DECODER "Enables the NvPipe decoding interface." ON)
option(NVPIPE_WITH_OPENGL "Enables the NvPipe OpenGL interface." ON)
option(NVPIPE_BUILD_EXAMPLES "Builds the NvPipe example applications (requires both encoder and decoder)." ON)

# Header
configure_file(src/NvPipe.h.in include/NvPipe.h @ONLY)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
link_directories(/usr/lib/nvidia-396)


# NvPipe shared library
list(APPEND NVPIPE_SOURCES
    src/NvPipe.cu
    src/NvCodec/Utils/ColorSpace.cu
    )
list(APPEND NVPIPE_LIBRARIES
    ${CMAKE_DL_LIBS}
    ${CUDA_LIBRARIES}
    cuda
    )

if (NVPIPE_WITH_ENCODER)
    list(APPEND NVPIPE_SOURCES
        src/NvCodec/NvEncoder/NvEncoder.cpp
        src/NvCodec/NvEncoder/NvEncoderCuda.cpp
        )
endif()

if (NVPIPE_WITH_DECODER)
    list(APPEND NVPIPE_SOURCES
        src/NvCodec/NvDecoder/NvDecoder.cpp
        )
    list(APPEND NVPIPE_LIBRARIES
        nvcuvid
        )
endif()

include(GNUInstallDirs)

cuda_add_library(${PROJECT_NAME} SHARED ${NVPIPE_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>
    )
target_include_directories(${PROJECT_NAME} PRIVATE
    $<BUILD_INTERFACE:src/NvCodec ${CUDA_INCLUDE_DIRS}>
    )
target_link_libraries(${PROJECT_NAME} ${NVPIPE_LIBRARIES})

set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1)

install(TARGETS ${PROJECT_NAME} EXPORT NvPipeConfig
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES ${CMAKE_BINARY_DIR}/include/NvPipe.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT NvPipeConfig DESTINATION share/NvPipe/cmake)

export(TARGETS ${PROJECT_NAME} FILE NvPipeConfig.cmake)

# Examples
if (NVPIPE_BUILD_EXAMPLES)
    # Encode to / decode from file
    add_executable(nvpExampleFile examples/file.cpp)
    target_link_libraries(nvpExampleFile PRIVATE ${PROJECT_NAME})

    if (NVPIPE_WITH_ENCODER AND NVPIPE_WITH_DECODER)
        # Host/device memory comparison
        add_executable(nvpExampleMemory examples/memory.cpp)
        target_link_libraries(nvpExampleMemory PRIVATE ${PROJECT_NAME})

        # Lossless test
        add_executable(nvpExampleLossless examples/lossless.cpp)
        target_link_libraries(nvpExampleLossless PRIVATE ${PROJECT_NAME})

        # EGL demo
        if (NVPIPE_WITH_OPENGL)
            list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/examples/cmake)

            find_package(EGL)
            find_package(GLEW)

            if (EGL_FOUND AND GLEW_FOUND)
                add_executable(nvpExampleEGL examples/egl.cpp)
                target_include_directories(nvpExampleEGL PRIVATE ${EGL_INCLUDE_DIR} ${GLEW_INCLUDE_DIR})
                target_link_libraries(nvpExampleEGL PRIVATE ${PROJECT_NAME} ${EGL_LIBRARIES} ${GLEW_LIBRARIES})
            endif()
        endif()
    endif()
endif()
