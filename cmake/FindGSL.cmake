set(GSL_FOUND FALSE)

find_path(GSL_INCLUDE_PATH gsl/gsl_cblas.h PATHS "C:/Program Files/GnuWin32" PATH_SUFFIXES include)

if(GSL_INCLUDE_PATH)

set(GSL_FOUND TRUE)

get_filename_component(GSL_LIB_PATH ${GSL_INCLUDE_PATH}/../lib ABSOLUTE)
find_library(GSL_GSL_A gsl PATHS ${GSL_LIB_PATH})
find_library(GSL_GSLCBLAS_A gslcblas PATHS ${GSL_LIB_PATH})
set(GSL_LIBRARIES ${GSL_GSL_A} ${GSL_GSLCBLAS_A})

if(NOT GSL_FIND_QUIETLY)
	message(STATUS "Found GSL: " ${GSL_INCLUDE_PATH})
endif(NOT GSL_FIND_QUIETLY)

endif(GSL_INCLUDE_PATH)

