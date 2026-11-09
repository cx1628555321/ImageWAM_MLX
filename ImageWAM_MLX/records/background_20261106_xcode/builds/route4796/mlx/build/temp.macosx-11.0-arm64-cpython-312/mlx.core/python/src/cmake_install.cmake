# Install script for directory: /Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_xcode/builds/route4796/mlx/python/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_xcode/builds/route4796/mlx/build/lib.macosx-11.0-arm64-cpython-312/mlx")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "core_stub")
  set(CMD "/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_xcode/builds/route4796/venv/bin/python;/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_xcode/builds/route4796/nanobind-3.1.0/src/stubgen.py;-q;-r;-i;/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_xcode/builds/route4796/mlx/build/lib.macosx-11.0-arm64-cpython-312/mlx/..;-i;/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_xcode/builds/route4796/mlx/python/src/..;-p;/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_xcode/builds/route4796/mlx/python/src/../mlx/_stub_patterns.txt;-m;mlx.core;-O;/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_xcode/builds/route4796/mlx/python/src/../mlx")
execute_process(
 COMMAND ${CMD}
 WORKING_DIRECTORY "${CMAKE_INSTALL_PREFIX}"
)
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_xcode/builds/route4796/mlx/build/temp.macosx-11.0-arm64-cpython-312/mlx.core/python/src/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
