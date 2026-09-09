cmake_minimum_required(VERSION 3.16)

project(phire_mt_matching_extractor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Host numerical-distance environment:
#   VTK 9.6 under /usr/local
#   TTK under /usr/local
#
# Request only the non-Qt VTK modules this extractor needs.
find_package(
  VTK 9.6 CONFIG REQUIRED
  COMPONENTS
    CommonCore
    CommonDataModel
    IOXML
)

find_package(TTKBase CONFIG REQUIRED)

add_executable(
  extract_ttk_mt_matching
  extract_ttk_mt_matching.cpp
)

target_include_directories(
  extract_ttk_mt_matching
  PRIVATE
    /usr/local/include/ttk/vtk
)

set(_ttk_targets "")

foreach(_candidate
    ttk::base::mergeTreeDistanceMatrix
    ttk::base::mergeTreeClustering
    ttk::base::ftmTreePP
    ttk::base::ftmTree
    ttk::base::assignmentSolver
    ttk::base::common
)
  if(TARGET "${_candidate}")
    list(APPEND _ttk_targets "${_candidate}")
  endif()
endforeach()

if(NOT _ttk_targets)
  message(FATAL_ERROR
    "No expected TTKBase imported targets were found. "
    "Inspect /usr/local/lib/cmake/ttkBase/TTKBaseTargets.cmake.")
endif()

message(STATUS "Using TTK base targets: ${_ttk_targets}")

target_link_libraries(
  extract_ttk_mt_matching
  PRIVATE
    ${_ttk_targets}
    VTK::CommonCore
    VTK::CommonDataModel
    VTK::IOXML
)
