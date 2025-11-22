# Generated Cmake Pico project file

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Initialise pico_sdk from installed location
# (note this can come from environment, CMake cache etc)

# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.2.0)
set(toolchainVersion 14_2_Rel1)
set(picotoolVersion 2.2.0-a4)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
set(PICO_BOARD pico CACHE STRING "Board type")

# Pull in Raspberry Pi Pico SDK (must be before project)
include(pico_sdk_import.cmake)

project(Miko C CXX ASM)

# Initialise the Raspberry Pi Pico SDK
pico_sdk_init()

# ============================================================================
# Miko - Custom Neural Network Implementation
# Pure C++ implementation with zero external ML library dependencies
# ============================================================================

add_executable(Miko
    Miko.cpp
    neural_network.cpp
    temp_sensor.cpp
)

pico_set_program_name(Miko "Miko")
pico_set_program_version(Miko "1.0")

# Set compiler flags for C++ compatibility
set_target_properties(
    Miko
    PROPERTIES
    COMPILE_FLAGS "-fno-exceptions"
)

# Enable USB output for serial communication
pico_enable_stdio_uart(Miko 0)
pico_enable_stdio_usb(Miko 1)

# Link libraries: Pico standard library + hardware ADC for temperature sensor
target_link_libraries(Miko
    pico_stdlib
    hardware_adc
)

# Add include directories
target_include_directories(Miko PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

pico_add_extra_outputs(Miko)

