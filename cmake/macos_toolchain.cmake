# macOS toolchain — fixes C++ standard library include path for macOS 26+ SDK
# where libc++ headers moved into the SDK instead of the CLT tools directory.
#
# Usage:  cmake -DCMAKE_TOOLCHAIN_FILE=cmake/macos_toolchain.cmake ...

if(APPLE)
  execute_process(
    COMMAND xcrun --sdk macosx --show-sdk-path
    OUTPUT_VARIABLE MACOS_SDK_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(CMAKE_OSX_SYSROOT "${MACOS_SDK_PATH}" CACHE STRING "macOS SDK" FORCE)

  # On macOS 26+ the C++ stdlib headers are inside the SDK; the CLT's
  # usr/include/c++/v1 directory is empty.  Tell clang where to look.
  set(CXX_INC "${MACOS_SDK_PATH}/usr/include/c++/v1")
  if(EXISTS "${CXX_INC}")
    add_compile_options(-I${CXX_INC})
  endif()
endif()
