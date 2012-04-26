#
#	Locates the AMD Core Math Library
#
#	Output:
#
#	ACML_FOUND - TRUE/FALSE - Whether the library has been found.
#		If FALSE, all other output variables are not defined.
#
#	ACML_PATH         - Library home.
#	ACML_INCLUDE_PATH - Path to the library's header files.
#	ACML_LIBRARY_PATH - Path to the library's binaries.
#	ACML_LIBRARIES    - Line for the linker.
#
#	The following locations are searched:
#	1. CMake ACMLROOT
#	2. Environment ACMLROOT
#	4. Default ACML installation directories
#	
set(ACML_FOUND FALSE)

if(ACMLROOT)
	find_path(ACML_PATH_ACMLROOT acml.h PATHS ${ACMLROOT}/include)
endif(ACMLROOT)
if(NOT ACML_PATH_ACMLROOT)
	find_path(ACML_PATH_ACMLROOT acml.h PATHS $ENV{ACMLROOT}/include)
endif(NOT ACML_PATH_ACMLROOT)

find_path(ACML_PATH_GUESS include/acml.h
	PATHS
	/opt/acml4.4.0/open64_64)

if(ACML_PATH_ACMLROOT)
	get_filename_component(ACML_PATH ${ACML_PATH_ACMLROOT}/.. ABSOLUTE)
	set(ACML_FOUND TRUE)
endif(ACML_PATH_ACMLROOT)
if(NOT ACML_FOUND AND ACML_PATH_GUESS)
	get_filename_component(ACML_PATH ${ACML_PATH_GUESS} ABSOLUTE)
	set(ACML_FOUND TRUE)
endif(NOT ACML_FOUND AND ACML_PATH_GUESS)

if(ACML_FOUND)

set(ACML_INCLUDE_PATH ${ACML_PATH}/include)
set(ACML_LIBRARY_PATH ${ACML_PATH}/lib)
set(ACML_LIBRARIES -L${ACML_LIBRARY_PATH} acml fortran ffio)

if(NOT ACML_FIND_QUIETLY)
	message(STATUS "Found ACML: " ${ACML_PATH})
endif(NOT ACML_FIND_QUIETLY)

else(ACML_FOUND)

if(ACML_FIND_REQUIRED)
	message(FATAL_ERROR "Could not find ACML")
endif(ACML_FIND_REQUIRED)

endif(ACML_FOUND)

