cmake_minimum_required(VERSION 3.15)
project(pygalago VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Optional: AddressSanitizer in Debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
  endif()
endif()

# ── pybind11 ──────────────────────────────────────────────────────────────────
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  # Fall back to pip-installed pybind11
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pb11_result
  )
  if(_pb11_result EQUAL 0)
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
    find_package(pybind11 CONFIG REQUIRED)
  else()
    message(STATUS "pybind11 not found — C++ extension will not be built")
    set(BUILD_BINDINGS OFF)
  endif()
endif()

# ── Core C++ library (libgalago) ──────────────────────────────────────────────
add_library(galago STATIC
  src/galago/compression/vbyte.cpp
  src/galago/io/file_stream.cpp
  src/galago/btree/vocabulary_reader.cpp
  src/galago/btree/disk_btree_iterator.cpp
  src/galago/btree/disk_btree_reader.cpp
)
target_include_directories(galago PUBLIC src)

# ── Python extension module ───────────────────────────────────────────────────
if(pybind11_FOUND AND NOT DEFINED BUILD_BINDINGS OR BUILD_BINDINGS)
  pybind11_add_module(_galago src/bindings/bindings.cpp)
  target_link_libraries(_galago PRIVATE galago)
  install(TARGETS _galago DESTINATION pygalago)
endif()

# ── Tests (opt-in) ───────────────────────────────────────────────────────────
option(BUILD_TESTS "Build C++ unit tests" OFF)
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()
