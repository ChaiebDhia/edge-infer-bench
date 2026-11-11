cmake_minimum_required(VERSION 3.20)
project(edge_infer_bench VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(EIB_WERROR "Treat warnings as errors" ON)
option(EIB_SANITIZE "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
option(EIB_BUILD_TESTS "Build unit and integration tests" ON)
set(ORT_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH "ONNX Runtime prebuilt root (see scripts/fetch_ort.sh)")

add_library(eib_warnings INTERFACE)
target_compile_options(eib_warnings INTERFACE -Wall -Wextra -Wpedantic -Wshadow)
if(EIB_WERROR)
  target_compile_options(eib_warnings INTERFACE -Werror)
endif()
if(EIB_SANITIZE)
  target_compile_options(eib_warnings INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
  target_link_options(eib_warnings INTERFACE -fsanitize=address,undefined)
endif()

add_library(eib INTERFACE)
target_include_directories(eib INTERFACE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(eib INTERFACE eib_warnings)

find_library(ORT_LIB onnxruntime PATHS "${ORT_ROOT}/lib" NO_DEFAULT_PATH)
if(NOT ORT_LIB)
  message(FATAL_ERROR "ONNX Runtime not found under ${ORT_ROOT}. Run scripts/fetch_ort.sh first.")
endif()

add_executable(edge_bench src/bench.cpp)
target_include_directories(edge_bench SYSTEM PRIVATE "${ORT_ROOT}/include")
target_link_libraries(edge_bench PRIVATE eib ${ORT_LIB})
set_target_properties(edge_bench PROPERTIES BUILD_RPATH "${ORT_ROOT}/lib")

if(EIB_BUILD_TESTS)
  enable_testing()
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_executable(unit_tests tests/test_units.cpp)
  target_link_libraries(unit_tests PRIVATE eib GTest::gtest_main)
  add_test(NAME unit_tests COMMAND unit_tests)

  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  set(TINY_MODEL "${CMAKE_BINARY_DIR}/tiny.onnx")
  set(TINY_RESULT "${CMAKE_BINARY_DIR}/tiny_result.json")
  add_test(NAME make_tiny_model
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/python/make_tiny_model.py --out ${TINY_MODEL})
  add_test(NAME bench_end_to_end
           COMMAND edge_bench --model ${TINY_MODEL} --label tiny --threads 2 --iters 20 --warmup 2 --out ${TINY_RESULT})
  add_test(NAME validate_result
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/python/check_result.py ${TINY_RESULT})
  set_tests_properties(make_tiny_model PROPERTIES FIXTURES_SETUP tiny_model)
  set_tests_properties(bench_end_to_end PROPERTIES FIXTURES_REQUIRED tiny_model FIXTURES_SETUP tiny_result)
  set_tests_properties(validate_result PROPERTIES FIXTURES_REQUIRED tiny_result)
endif()
