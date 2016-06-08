#===============ds_opencv top-level CMakeLists.txt==========================#
#
#  Created on: Dec 8, 2014
#      Author: Gregory Kramida
#     License: Apache v2
#   Copyright: (c) Gregory Kramida 2016
#
# EDITING GUIDELINES:
# 1) Use lowercase CMake commands
# 2) Succinct > descriptive > abbreviated
# 3) Less = more
# 4) Hard now = easy later
# 5) Freedom = slavery
#

cmake_minimum_required(VERSION 3.4)
#----------------------------PROJECT NAME--------------------------------------#
project(cve)
set(PYTHON_MODULE_NAME cve)

#----------------------------CMAKE & GLOBAL PROPERTIES-------------------------#
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_INCLUDE_CURRENT_DIR ON)

#----------------------------BUILD CONFIGURATIONS------------------------------#
set(DEBUG_BUILD_TYPE_OPTIONS Debug RelWithDebInfo)
set(RELEASE_BUILD_TYPE_OPTIONS Release MinSizeRel)
set(BUILD_TYPE_OPTIONS ${DEBUG_BUILD_TYPE_OPTIONS} ${RELEASE_BUILD_TYPE_OPTIONS})

#set build type
if (NOT MSVC)#TODO: also don't do for XCode generator?
    #work around CMakes strange CACHE setting system
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
        set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: ${BUILD_TYPE_OPTIONS}." FORCE)
    else ()
        set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: ${BUILD_TYPE_OPTIONS}.")
    endif ()
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${BUILD_TYPE_OPTIONS})
    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})
endif ()

#----------------------------COMPILER FLAGS-----------------------------------#
if (MSVC)
    #ADD_DEFINITIONS(-DUNICODE -D_UNICODE) #TODO: not sure about those
else ()
    set(CMAKE_CXX_FLAGS "-Wall")
    if (DEFINED CMAKE_BUILD_TYPE)
        if (${CMAKE_BUILD_TYPE} STREQUAL "Debug" OR ${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
            add_definitions(-D_DEBUG) #add the debug flag
        else ()
            add_definitions(-DQT_NO_DEBUG_OUTPUT) #add the qt debug output flag
        endif ()
    endif ()

    ### C++11 support
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
    CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
    if (COMPILER_SUPPORTS_CXX11)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    elseif (COMPILER_SUPPORTS_CXX0X)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    else ()
        message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
    endif ()
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#----------------------------PYTHON OPTION-------------------------------------#
set(PYTHON_OPTIONS "2.X" "3.X")
set(PYTHON_DESIRED_VERSION "3.X" CACHE STRING "Choose which python version to use, options are: ${PYTHON_OPTIONS}.")
set_property(CACHE PYTHON_DESIRED_VERSION PROPERTY STRINGS ${PYTHON_OPTIONS})


#--------------------------FIND PACKAGES----------------------------------------#

find_package(OpenCV COMPONENTS core imgproc highgui REQUIRED)
include("DetectPython")
if (${PYTHON_DESIRED_VERSION} STREQUAL "2.X")
    set(Python_ADDITIONAL_VERSIONS ${PYTHON2_VERSION_MAJOR}.${PYTHON2_VERSION_MINOR})
    find_package(Boost COMPONENTS python-py${PYTHON2_VERSION_MAJOR}${PYTHON2_VERSION_MINOR} REQUIRED)
else ()
    set(Python_ADDITIONAL_VERSIONS ${PYTHON3_VERSION_MAJOR}.${PYTHON3_VERSION_MINOR})
    find_package(Boost COMPONENTS python-py${PYTHON3_VERSION_MAJOR}${PYTHON3_VERSION_MINOR} REQUIRED)
endif ()

#========pick python stuff========================================
if(${PYTHON_DESIRED_VERSION} STREQUAL "2.X")
    SET(PYTHON_INCLUDE_DIRS ${PYTHON2_INCLUDE_DIR} ${PYTHON2_INCLUDE_DIR2} ${PYTHON2_NUMPY_INCLUDE_DIRS})
    SET(PYTHON_LIBRARIES ${PYTHON2_LIBRARY})
    SET(PYTHON_EXECUTABLE ${PYTHON2_EXECUTABLE})
    SET(PYTHON_PACKAGES_PATH ${PYTHON2_PACKAGES_PATH})
else()
    SET(PYTHON_INCLUDE_DIRS ${PYTHON3_INCLUDE_DIR} ${PYTHON3_INCLUDE_DIR2} ${PYTHON3_NUMPY_INCLUDE_DIRS})
    SET(PYTHON_LIBRARIES ${PYTHON3_LIBRARY})
    SET(PYTHON_EXECUTABLE ${PYTHON3_EXECUTABLE})
    SET(PYTHON_PACKAGES_PATH ${PYTHON3_PACKAGES_PATH})
endif()

file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/include/pyboostcvconverter/pyboostcvconverter.hpp)
target_include_directories(${PROJECT_NAME} PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ${Boost_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS}
        )

target_link_libraries(${PROJECT_NAME}
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${PYTHON_LIBRARIES}
        )

#---------------------------   INSTALLATION    -----------------------------------------------------
#-get proper extension for python binary shared object on this platform


execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('SO'))"
        RESULT_VARIABLE PYTHON_${PROJECT_NAME}_PY_PROCESS
        OUTPUT_VARIABLE ${PROJECT_NAME}_PY_SUFFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE)


set_target_properties(${PROJECT_NAME} PROPERTIES
        ARCHIVE_OUTPUT_NAME ${PROJECT_NAME}  # prevent name conflict for python2/3 outputs
        PREFIX ""
        OUTPUT_NAME ${PROJECT_NAME}
        SUFFIX ${${PROJECT_NAME}_PY_SUFFIX})

if(MSVC AND NOT BUILD_SHARED_LIBS)
    set_target_properties(${the_module} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /DEBUG")
endif()

if(MSVC AND NOT PYTHON_DEBUG_LIBRARIES)
    set(PYTHON_INSTALL_CONFIGURATIONS CONFIGURATIONS Release)
else()
    set(PYTHON_INSTALL_CONFIGURATIONS "")
endif()

if(WIN32)
    set(PYTHON_INSTALL_ARCHIVE "")
else()
    set(PYTHON_INSTALL_ARCHIVE ARCHIVE DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT python)
endif()

install(TARGETS ${PROJECT_NAME}
        ${PYTHON_INSTALL_CONFIGURATIONS}
        RUNTIME DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT python
        LIBRARY DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT python
        ${PYTHON_INSTALL_ARCHIVE}
        )
