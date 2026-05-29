cmake_minimum_required(VERSION 3.15)

# ── Catch2 (fetch at configure time) ─────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.6.0
)
FetchContent_MakeAvailable(Catch2)

# ── VByte tests ───────────────────────────────────────────────────────────────
add_executable(test_vbyte test_vbyte.cpp)
target_link_libraries(test_vbyte PRIVATE galago Catch2::Catch2WithMain)
add_test(NAME vbyte COMMAND test_vbyte)

# ── FileStream + BTree tests ──────────────────────────────────────────────────
add_executable(test_btree test_btree.cpp)
target_link_libraries(test_btree PRIVATE galago Catch2::Catch2WithMain)
add_test(NAME btree COMMAND test_btree)
