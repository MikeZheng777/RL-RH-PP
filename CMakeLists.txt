cmake_minimum_required(VERSION 3.12)

# Project information
project(warehouse_sim LANGUAGES CXX)

# Set C++ Standard to match your working command
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_BUILD_TYPE "Debug")

# Ensure that you use Debug flags for debugging, otherwise Release
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "-shared -O0 -g -fPIC")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-shared -O3 -fPIC")
endif()


# Optionally set PYTHON_EXECUTABLE to pin a specific Python interpreter:
#   cmake -DPYTHON_EXECUTABLE=/path/to/python ..
if(NOT DEFINED PYTHON_EXECUTABLE)
    find_program(PYTHON_EXECUTABLE python3)
endif()

# Set Python root directory
get_filename_component(PYTHON_ROOT "${PYTHON_EXECUTABLE}" DIRECTORY)
get_filename_component(PYTHON_ROOT "${PYTHON_ROOT}" DIRECTORY)

# Set hints for finding Python
set(Python_ROOT_DIR "${PYTHON_ROOT}")
set(Python_FIND_STRATEGY LOCATION)

# Find Python (Interpreter and Development headers)
find_package(Python 3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python_FOUND: ${Python_FOUND}")
message(STATUS "Python_VERSION: ${Python_VERSION}")
message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
message(STATUS "Python_LIBRARIES: ${Python_LIBRARIES}")


execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(pybind11_DIR "${PYBIND11_CMAKE_DIR}")

# Find pybind11 package
find_package(pybind11 REQUIRED)

# Find Pybind11 (use system-installed version if available)
# find_package(pybind11 REQUIRED)

# Find Boost
find_package(Boost REQUIRED COMPONENTS program_options system filesystem)
message(STATUS "Boost_FOUND: ${Boost_FOUND}")
message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")

# Include directories
include_directories("inc")
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${Python_INCLUDE_DIRS})
include_directories(${PYBIND11_INCLUDE_DIRS})

# Source files (assuming source files are in src folder)
file(GLOB SOURCES "src/*.cpp")

# Create Python module
pybind11_add_module(warehouse_sim ${SOURCES})

# Link libraries (Boost and Python)
target_link_libraries(warehouse_sim PRIVATE
    ${Boost_LIBRARIES}
    Boost::filesystem
    Boost::system
    Python::Python
)

# Include directories for module
target_include_directories(warehouse_sim PRIVATE
    ${Python_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

# Output directory for Python module
set_target_properties(warehouse_sim PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python
)

# Add rpath for Boost libraries (important for runtime linking)
set_target_properties(warehouse_sim PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
    INSTALL_RPATH "${Boost_LIBRARY_DIRS}"
    BUILD_WITH_INSTALL_RPATH TRUE
)
