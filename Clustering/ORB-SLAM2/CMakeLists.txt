cmake_minimum_required(VERSION 3.10)
project(ORB_SLAM2)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-Wno-deprecated-declarations)

set(CMAKE_C_FLAGS_DEBUG "-O2 -g")
set(CMAKE_CXX_FLAGS_DEBUG "-O2 -g")

set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

LIST(APPEND CMAKE_C_FLAGS "-Wall -Wextra -DUSE_NVTX")
LIST(APPEND CMAKE_CXX_FLAGS "-Wall -Wextra -DUSE_NVTX")

find_package(CUDA QUIET REQUIRED)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
SET(CUDA_HOST_COMPILER /usr/bin/g++)
LIST(APPEND CUDA_NVCC_FLAGS "--compiler-options -fno-strict-aliasing -use_fast_math")

# Ensure correct architecture
LIST(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_87,code=sm_87")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
   set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
endif()

find_package(Pangolin REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${Pangolin_INCLUDE_DIRS}
)

# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

cuda_include_directories(
  ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
)

cuda_add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/fast.cu
src/orientation.cu
src/resize.cu
src/gaussian_blur.cu
src/descriptor.cu
src/k_means.cu
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
nvToolsExt
Eigen3::Eigen
)

option(VACCEL "Enable VACCEL support" OFF)
option(CPUONLY "Enable CPU-only execution" OFF)

if (VACCEL)
  message(STATUS "Building with VACCEL support")

  target_link_libraries(${PROJECT_NAME} vaccel)

  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

  set(SOURCES
      Examples/Stereo/stereo_kitti.cc
      src/wrap/wrapper.cpp
      src/wrap/utils.hpp
  )

  if (CPUONLY)
    message(STATUS "Building CPU-only executable")
    add_library(liborb-cpu SHARED ${SOURCES})

    target_link_libraries(liborb-cpu
      ${OpenCV_LIBS}
      ${Pangolin_LIBRARIES}
      ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
      ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
      Eigen3::Eigen
      vaccel
      pthread
    )

    set_target_properties(liborb-cpu PROPERTIES
      OUTPUT_NAME "orb-cpu"
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    )

    add_compile_definitions(VACCEL)
    add_compile_definitions(CPUONLY)

    target_compile_options(liborb-cpu PRIVATE -O3)
    add_executable(cputest ${SOURCES})

    target_link_libraries(cputest PRIVATE
      ${PROJECT_NAME}
      vaccel -pthread
    )
  else()
    message(STATUS "Building GPU-only executable")
    add_library(liborb-gpu SHARED ${SOURCES})

    target_link_libraries(liborb-gpu
      ${OpenCV_LIBS}
      ${Pangolin_LIBRARIES}
      ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
      ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
      Eigen3::Eigen
      vaccel
      nvToolsExt
      pthread
    )

    set_target_properties(liborb-gpu PROPERTIES
      OUTPUT_NAME "orb-gpu"
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    )

    add_compile_definitions(VACCEL)
    # add_compile_definitions(CPUONLY)

    target_compile_options(liborb-gpu PRIVATE -O3)
    add_executable(gputest ${SOURCES})

    target_link_libraries(gputest PRIVATE
      ${PROJECT_NAME}
      vaccel -pthread
    )

  endif()


else()

  # Build examples
  if(NO_EXAMPLES)
    message(STATUS "Don't build examples")
  else ()
    if(CPUONLY)
      message(STATUS "Build cpuonly: stereo_kitti_cpu")

      add_compile_definitions(CPUONLY)

      add_executable(stereo_kitti_cpu
              Examples/Stereo/stereo_kitti.cc)

      target_link_libraries(stereo_kitti_cpu ${PROJECT_NAME})
    else()
      message(STATUS "Build GPU examples")

      set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

      add_executable(rgbd_tum
              Examples/RGB-D/rgbd_tum.cc)
      target_link_libraries(rgbd_tum ${PROJECT_NAME})

      set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

      add_executable(stereo_kitti
              Examples/Stereo/stereo_kitti.cc)
      target_link_libraries(stereo_kitti ${PROJECT_NAME})

      add_executable(stereo_euroc
              Examples/Stereo/stereo_euroc.cc)
      target_link_libraries(stereo_euroc ${PROJECT_NAME})


      set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

      add_executable(mono_tum
              Examples/Monocular/mono_tum.cc)
      target_link_libraries(mono_tum ${PROJECT_NAME})

      add_executable(mono_kitti
              Examples/Monocular/mono_kitti.cc)
      target_link_libraries(mono_kitti ${PROJECT_NAME})

      add_executable(mono_euroc
              Examples/Monocular/mono_euroc.cc)
      target_link_libraries(mono_euroc ${PROJECT_NAME})
    endif()
  endif() # Build Examples
endif()

unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY)

