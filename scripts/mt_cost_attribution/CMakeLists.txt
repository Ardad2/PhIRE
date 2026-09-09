cmake_minimum_required(VERSION 3.16)

project(phire_mt_cost_attribution LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(
  VTK 9.6 CONFIG REQUIRED
  COMPONENTS
    CommonCore
    CommonDataModel
    IOXML
)

find_package(TTKBase CONFIG REQUIRED)

add_executable(
  extract_ttk_mt_cost_attribution
  extract_ttk_mt_cost_attribution.cpp
)

target_include_directories(
  extract_ttk_mt_cost_attribution
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
  message(FATAL_ERROR "No expected TTKBase imported targets were found.")
endif()

target_link_libraries(
  extract_ttk_mt_cost_attribution
  PRIVATE
    ${_ttk_targets}
    VTK::CommonCore
    VTK::CommonDataModel
    VTK::IOXML
)
