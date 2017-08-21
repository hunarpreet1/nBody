cmake_minimum_required(VERSION 3.5)
project( nbody-gpu )
find_package( OpenCV REQUIRED )
find_package(CUDA QUIET REQUIRED)
include(FindCUDA)

include_directories(/usr/local/cuda/include)
CUDA_ADD_LIBRARY(cudalib
    SHARED
    nbody.cu
)
TARGET_LINK_LIBRARIES(cudalib
    ${CUDA_LIBRARIES}
)
cuda_add_executable(nbody nbody.cu)
target_link_libraries(nbody cudalib ${OpenCV_LIBS})

