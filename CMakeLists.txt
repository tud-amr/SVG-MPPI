cmake_minimum_required(VERSION 3.0.2)
project(SVG-MPPI)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  std_msgs
  rospy
)

## Uncomment if package has a setup.py for Python module installation
catkin_python_setup()

#############################################
## Declare ROS messages, services, and actions
#############################################

## Uncomment and list every .msg/.srv/.action file to be processed
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate messages/services/actions with dependencies
# generate_messages(
#   DEPENDENCIES
#   std_msgs
# )

##############################################
## Declare ROS dynamic reconfigure parameters
##############################################

## Generate dynamic reconfigure parameters
generate_dynamic_reconfigure_options(
  config/weights/DingoCost.cfg
)

###################################
## Catkin specific configuration ##
###################################

## Declare catkin-specific configuration to be passed to dependent projects
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES semantic_namo
#  CATKIN_DEPENDS message_runtime
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Include directories
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library (optional)
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/semantic_namo.cpp
# )

## Add CMake target dependencies for generated code
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
# add_executable(${PROJECT_NAME}_node src/semantic_namo_node.cpp)

## Rename C++ executable (optional)
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add dependencies for the executable
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Link libraries to executable
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

## Install Python scripts
catkin_install_python(PROGRAMS
  scripts/visualisation
  scripts/environment
  scripts/controller
  tests/setup_0
  tests/setup_1
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install other files (e.g., launch, config, assets)
install(DIRECTORY config assets launch
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
)

#############
## Testing ##
#############

## Add gtest-based cpp test
# catkin_add_gtest(${PROJECT_NAME}-test test/test_semantic_namo.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
