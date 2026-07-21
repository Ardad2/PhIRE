# - Config file for the GUDHI package
# It defines the following variables
#  GUDHI_INCLUDE_DIRS - include directories for GUDHI
#
# Order is :
# 1. user defined GUDHI_INCLUDE_DIRS
# 2. ${CMAKE_SOURCE_DIR}/include     => Where the 'cmake' has been done
# 3. ${CMAKE_INSTALL_PREFIX}/include => Where the 'make install' has been performed

# Compute paths
set(GUDHI_INCLUDE_DIRS "${GUDHI_INCLUDE_DIRS};/home/conda/feedstock_root/build_artifacts/gudhi_1774940357458/work/build/version/include;/home/adadhwal/PhIRE/.mamba_candidateD_pd/include")

