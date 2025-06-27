cmake_minimum_required(VERSION 3.12)
project(onnx_inference_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Debug")

option(BUILD_SHARED_LIBS "Whether to build shared libraries" OFF)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# 查找ONNX Runtime
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(onnxruntime)
include(kaldi-native-fbank)
include(libsamplerate)
set(ONNXRUNTIME_DIR ${onnxruntime_SOURCE_DIR})
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")
message(STATUS "ONNXRUNTIME_INCLUDE_DIRS: ${ONNXRUNTIME_INCLUDE_DIRS}")
include_directories(${CMAKE_SOURCE_DIR})

find_package(ALSA REQUIRED)

# 可执行文件
add_executable(infer
    clog.cpp
    asr.cpp
    main.cc
    sense_voice.cpp
    vad.cpp
)

# 链接库
target_link_libraries(infer
   ${onnxruntime_lib_files} 
   kaldi-native-fbank-core
   samplerate
)

# 可执行文件
add_executable(stream
    clog.cpp
    asr.cpp
    sense_voice.cpp
    vad.cpp
    resample.cc
    stream.cc
    alsa.cc
)

# 链接库
target_link_libraries(stream
   ${onnxruntime_lib_files} 
   kaldi-native-fbank-core
   samplerate
   ALSA::ALSA
)
