set(MRT_PKG_VERSION 3.0.1)
# Modify only if you know what you are doing!
cmake_minimum_required(VERSION 3.5.1)
project(votof_tool)

###################
## Find packages ##
###################
find_package(mrt_cmake_modules REQUIRED)
include(UseMrtStdCompilerFlags)

include(GatherDeps)
# Remove libs which cannot be found automatically
#list(REMOVE_ITEM DEPENDEND_PACKAGES <package name 1> <package name 2> ...)
find_package(AutoDeps REQUIRED COMPONENTS ${DEPENDEND_PACKAGES})

# Manually resolve removed dependend packages
#find_package(...)

########################
## Add python modules ##
########################
# This adds a python module if located under src/{PROJECT_NAME)
mrt_python_module_setup()

###################################
## Catkin specific configuration ##
###################################
catkin_package()

###########
## Build ##
###########
# Add include and library directories
include_directories(
    ${mrt_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    )
link_directories(
    ${mrt_LIBRARY_DIRS}
    )

mrt_glob_folders(SRC_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/src")
if (SRC_DIRECTORIES)
    # Found subfolders, add executable for each subfolder
    foreach(SRC_DIR ${SRC_DIRECTORIES})
        mrt_add_executable(${SRC_DIR} FOLDER "src/${SRC_DIR}")
    endforeach()
else()
    # No subfolder found, add executable and python modules for src folder
    mrt_add_executable(${PROJECT_NAME} FOLDER "src")
endif()

#############
## Install ##
#############
# Install all targets, headers by default and scripts and other files if specified (folders or files)
mrt_install(PROGRAMS scripts FILES res data)

#############
## Testing ##
#############
# Add test targets for cpp and python tests
if (CATKIN_ENABLE_TESTING)
    mrt_add_tests(test)
    mrt_add_nosetests(test)
endif()
