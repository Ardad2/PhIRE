cmake_minimum_required(VERSION 3.16)

project(phire_mt_matching_extractor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# IMPORTANT:
# TTK in phire-ttk is linked against the distro VTK 9.1 runtime.
# Request only the non-Qt VTK modules needed by this executable, rather than
# loading the full TTKVTK package (which pulls Qt-enabled VTK dependencies).
find_package(
  VTK 9.1 CONFIG REQUIRED
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

# ttkMergeTreeUtils.h is an installed header-only VTK utility used to turn the
# original TTK node/arc VTU files into the same ftm::MergeTree representation.
target_include_directories(
  extract_ttk_mt_matching
  PRIVATE
    /usr/local/include/ttk/vtk
)

# TTK's exact exported base-target spelling can vary across installs.
# Select the installed merge-tree base targets defensively.
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
