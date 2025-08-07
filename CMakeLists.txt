# Define the following variables to use the CMake build system:
# - BLT_SOURCE_DIR : BLT install directory
# - MFEM_DIR : MFEM install directory

cmake_minimum_required(VERSION 3.14)

project(continuation LANGUAGES CXX C)

set(ENABLE_MPI ON CACHE BOOL "")

if (NOT DEFINED BLT_SOURCE_DIR)
  message(FATAL_ERROR "BLT_SOURCE_DIR required")
endif()
if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
  message(FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
endif()
if (NOT BLT_CXX_STD)
  set(BLT_CXX_STD "c++14" CACHE STRING "")
endif()
# These BLT tools are not used in the continuation solver, turn them off
set(_unused_blt_tools
  CLANGQUERY
  VALGRIND
  ASTYLE
  CMAKEFORMAT
  CLANGTIDY
  CPPCHECK
  DOXYGEN
  SPHINX
  UNCRUSTIFY
  YAPF)
foreach(_tool ${_unused_blt_tools})
  set(ENABLE_${_tool} OFF CACHE BOOL "")
endforeach()
include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

#if (NOT TARGET mfem)
#   include(${PROJECT_SOURCE_DIR}/../cmake/thirdparty/FindMFEM.cmake)
#endif()

set(continuation_sources
  problems/NLMCProblems.cpp
  problems/OptProblems.cpp
  solvers/HomotopySolver.cpp
  solvers/CondensedHomotopySolver.cpp
  solvers/IPSolver.cpp
  solvers/AMGF.cpp
  utilities.cpp)

set(continuation_headers
  problems/NLMCProblems.hpp
  problems/OptProblems.hpp
  problems/Problems.hpp
  solvers/Solvers.hpp
  solvers/HomotopySolver.hpp
  solvers/CondensedHomotopySolver.hpp
  solvers/IPSolver.hpp
  solvers/AMGF.hpp
  utilities.hpp)

set(continuation_depends
  blt::mpi
  mfem)

blt_add_library(
  NAME       continuation_solver
  SOURCES    ${continuation_sources}
  HEADERS    ${continuation_headers}
  DEPENDS_ON ${continuation_depends})

target_include_directories(continuation_solver PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/continuationsolver>)

set(continuation_problems_headers
  problems/NLMCProblems.hpp
  problems/OptProblems.hpp
  problems/Problems.hpp)

install(FILES ${continuation_problems_headers} DESTINATION include/continuationsolver/problems)

set(continuation_solvers_headers
  solvers/Solvers.hpp
  solvers/HomotopySolver.hpp
  solvers/IPSolver.hpp)

install(FILES ${continuation_solvers_headers} DESTINATION include/continuationsolver/solvers)

install(FILES utilities.hpp DESTINATION include/continuationsolver)

install(
  TARGETS     continuation_solver
  EXPORT      continuation-targets
  DESTINATION lib)
