# Install script for directory: /Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_offline/builds/affine4802/mlx

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_offline/builds/affine4802/mlx/build/lib.macosx-11.0-arm64-cpython-312/mlx")
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

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_offline/builds/affine4802/mlx/build/temp.macosx-11.0-arm64-cpython-312/mlx.core/_deps/json-build/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_offline/builds/affine4802/mlx/build/temp.macosx-11.0-arm64-cpython-312/mlx.core/jaccl/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_offline/builds/affine4802/mlx/build/temp.macosx-11.0-arm64-cpython-312/mlx.core/mlx/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_offline/builds/affine4802/mlx/build/temp.macosx-11.0-arm64-cpython-312/mlx.core/python/src/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY MESSAGE_NEVER FILES "/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_offline/builds/affine4802/mlx/build/temp.macosx-11.0-arm64-cpython-312/mlx.core/libmlx.dylib")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmlx.dylib" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmlx.dylib")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_offline/builds/affine4802/mlx/build/temp.macosx-11.0-arm64-cpython-312/mlx.core/jaccl"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmlx.dylib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -x "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmlx.dylib")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "headers" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY MESSAGE_NEVER FILES "/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_offline/builds/affine4802/mlx/mlx" FILES_MATCHING REGEX "/[^/]*\\.h$" REGEX "/backend\\/metal\\/kernels\\.h$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "metal_cpp_source" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/metal_cpp" TYPE DIRECTORY MESSAGE_NEVER FILES "/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261105/builds/route4796/mlx/build/temp.macosx-11.0-arm64-cpython-312/mlx.core/_deps/metal_cpp-src/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/MLX/MLXTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/MLX/MLXTargets.cmake"
         "/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_offline/builds/affine4802/mlx/build/temp.macosx-11.0-arm64-cpython-312/mlx.core/CMakeFiles/Export/1e8da368a82b8fcebd3a63675e0335e4/MLXTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/MLX/MLXTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/MLX/MLXTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/MLX" TYPE FILE MESSAGE_NEVER FILES "/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_offline/builds/affine4802/mlx/build/temp.macosx-11.0-arm64-cpython-312/mlx.core/CMakeFiles/Export/1e8da368a82b8fcebd3a63675e0335e4/MLXTargets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/MLX" TYPE FILE MESSAGE_NEVER FILES "/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_offline/builds/affine4802/mlx/build/temp.macosx-11.0-arm64-cpython-312/mlx.core/CMakeFiles/Export/1e8da368a82b8fcebd3a63675e0335e4/MLXTargets-release.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/MLX" TYPE FILE MESSAGE_NEVER FILES
    "/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_offline/builds/affine4802/mlx/build/temp.macosx-11.0-arm64-cpython-312/mlx.core/MLXConfig.cmake"
    "/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_offline/builds/affine4802/mlx/build/temp.macosx-11.0-arm64-cpython-312/mlx.core/MLXConfigVersion.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/MLX" TYPE DIRECTORY MESSAGE_NEVER FILES "/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_offline/builds/affine4802/mlx/cmake/")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_offline/builds/affine4802/mlx/build/temp.macosx-11.0-arm64-cpython-312/mlx.core/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
if(CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_COMPONENT MATCHES "^[a-zA-Z0-9_.+-]+$")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
  else()
    string(MD5 CMAKE_INST_COMP_HASH "${CMAKE_INSTALL_COMPONENT}")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INST_COMP_HASH}.txt")
    unset(CMAKE_INST_COMP_HASH)
  endif()
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/qklj/work/cx/ImageWAM_MLX/records/background_20261106_offline/builds/affine4802/mlx/build/temp.macosx-11.0-arm64-cpython-312/mlx.core/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
