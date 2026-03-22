# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.11)
project(_turbomind LANGUAGES CXX CUDA)

find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
    execute_process(COMMAND "pybind11-config" "--cmakedir"
                    RESULT_VARIABLE _COMMAND_SUCCESS
                    OUTPUT_VARIABLE pybind11_DIR
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    find_package(pybind11 CONFIG)
endif()

if(LMDEPLOY_SPLIT_DEBUG_INFO)
  # pybind11_add_module strips Release modules unless NO_EXTRAS is set.
  # Disable the built-in strip pass so detached symbols keep full debug info.
  set(_lmdeploy_pybind_no_extras NO_EXTRAS)
endif()

pybind11_add_module(${PROJECT_NAME} ${_lmdeploy_pybind_no_extras} bind.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE turbomind xgrammar)

pybind11_add_module(_xgrammar ${_lmdeploy_pybind_no_extras} xgrammar_bind.cpp)
target_link_libraries(_xgrammar PRIVATE core xgrammar)
target_compile_features(_xgrammar PRIVATE cxx_std_14)

if(UNIX AND NOT APPLE)
  find_program(LMDEPLOY_OBJCOPY_EXECUTABLE NAMES objcopy llvm-objcopy)
  find_program(LMDEPLOY_STRIP_EXECUTABLE NAMES strip llvm-strip)
endif()

function(lmdeploy_optimize_binary target_name)
  if(NOT UNIX OR APPLE)
    return()
  endif()

  if(LMDEPLOY_SPLIT_DEBUG_INFO)
    if(NOT LMDEPLOY_OBJCOPY_EXECUTABLE OR NOT LMDEPLOY_STRIP_EXECUTABLE)
      message(WARNING
              "LMDEPLOY_SPLIT_DEBUG_INFO is ON but objcopy/strip not found. "
              "Skip split-debug optimization for target ${target_name}.")
      return()
    endif()
    set(_debug_dir "${CMAKE_BINARY_DIR}/debug_symbols")
    add_custom_command(TARGET ${target_name} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory "${_debug_dir}"
      COMMAND ${LMDEPLOY_OBJCOPY_EXECUTABLE}
              --only-keep-debug
              $<TARGET_FILE:${target_name}>
              "${_debug_dir}/$<TARGET_FILE_NAME:${target_name}>.debug"
      COMMAND ${LMDEPLOY_STRIP_EXECUTABLE}
              --strip-unneeded
              $<TARGET_FILE:${target_name}>
      COMMAND ${LMDEPLOY_OBJCOPY_EXECUTABLE}
              --add-gnu-debuglink=${_debug_dir}/$<TARGET_FILE_NAME:${target_name}>.debug
              $<TARGET_FILE:${target_name}>
      COMMENT "Split debug symbols and strip ${target_name}"
      VERBATIM)
    return()
  endif()

  if(LMDEPLOY_STRIP_BINARIES)
    if(NOT LMDEPLOY_STRIP_EXECUTABLE)
      message(WARNING
              "LMDEPLOY_STRIP_BINARIES is ON but strip not found. "
              "Skip strip optimization for target ${target_name}.")
      return()
    endif()
    add_custom_command(TARGET ${target_name} POST_BUILD
      COMMAND ${LMDEPLOY_STRIP_EXECUTABLE}
              --strip-unneeded
              $<TARGET_FILE:${target_name}>
      COMMENT "Strip ${target_name}"
      VERBATIM)
  endif()
endfunction()

lmdeploy_optimize_binary(${PROJECT_NAME})
lmdeploy_optimize_binary(_xgrammar)
unset(_lmdeploy_pybind_no_extras)

if (CALL_FROM_SETUP_PY)
  string(REPLACE "." ";" _ver ${CMAKE_CUDA_COMPILER_VERSION})
  list(GET _ver 0 CUDA_MAJOR)

  if(CUDA_MAJOR GREATER_EQUAL "13")
    set(_INSTALL_CUDA_RPATH
        "\$ORIGIN"
        "\$ORIGIN/../../nvidia/nccl/lib/"
        "\$ORIGIN/../../nvidia/cu${CUDA_MAJOR}/lib/"
    )
  else()
    set(_INSTALL_CUDA_RPATH
        "\$ORIGIN"
        "\$ORIGIN/../../nvidia/nccl/lib/"
        "\$ORIGIN/../../nvidia/cuda_runtime/lib/"
        "\$ORIGIN/../../nvidia/cublas/lib/"
        "\$ORIGIN/../../nvidia/curand/lib/"
    )
  endif()
  set_target_properties(${PROJECT_NAME} PROPERTIES
      BUILD_RPATH "\$ORIGIN"
      INSTALL_RPATH "${_INSTALL_CUDA_RPATH}"
  )
endif ()
