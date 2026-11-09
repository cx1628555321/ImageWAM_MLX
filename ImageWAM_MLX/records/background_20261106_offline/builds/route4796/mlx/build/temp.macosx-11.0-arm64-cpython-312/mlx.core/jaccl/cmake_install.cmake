# Install script for directory: /Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_offline/builds/route4796/mlx/mlx/distributed/jaccl/lib

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_offline/builds/route4796/mlx/build/lib.macosx-11.0-arm64-cpython-312/mlx")
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

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY MESSAGE_NEVER FILES "/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_offline/builds/route4796/mlx/build/temp.macosx-11.0-arm64-cpython-312/mlx.core/jaccl/libjaccl.dylib")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libjaccl.dylib" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libjaccl.dylib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -x "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libjaccl.dylib")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/jaccl" TYPE DIRECTORY MESSAGE_NEVER FILES "/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_offline/builds/route4796/mlx/mlx/distributed/jaccl/lib/jaccl/" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/jaccl/jacclTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/jaccl/jacclTargets.cmake"
         "/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_offline/builds/route4796/mlx/build/temp.macosx-11.0-arm64-cpython-312/mlx.core/jaccl/CMakeFiles/Export/6791f8f5f8f252b33a313899a9482ce4/jacclTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/jaccl/jacclTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/jaccl/jacclTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/jaccl" TYPE FILE MESSAGE_NEVER FILES "/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_offline/builds/route4796/mlx/build/temp.macosx-11.0-arm64-cpython-312/mlx.core/jaccl/CMakeFiles/Export/6791f8f5f8f252b33a313899a9482ce4/jacclTargets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/jaccl" TYPE FILE MESSAGE_NEVER FILES "/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_offline/builds/route4796/mlx/build/temp.macosx-11.0-arm64-cpython-312/mlx.core/jaccl/CMakeFiles/Export/6791f8f5f8f252b33a313899a9482ce4/jacclTargets-release.cmake")
  endif()
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_offline/builds/route4796/mlx/build/temp.macosx-11.0-arm64-cpython-312/mlx.core/jaccl/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
