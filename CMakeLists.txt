cmake_minimum_required(VERSION 3.13)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LIBNYQUIST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_MODULE_PATH ${LIBNYQUIST_ROOT}/cmake)

include(CXXhelpers)
include(GNUInstallDirs)

option (USE_CUDA "USE CUDA to compile the library")
if(USE_CUDA)
  message(STATUS "CUDA ENABLED")
else()
  message(STATUS "CUDA DISABLED")
endif()


if (CMAKE_OSX_ARCHITECTURES)
    if(CMAKE_OSX_SYSROOT MATCHES ".*iphoneos.*")
        # RtAudio is not portable to ios currently
        option(LIBNYQUIST_BUILD_EXAMPLE "Build example application" OFF)
    else()
        option(LIBNYQUIST_BUILD_EXAMPLE "Build example application" ON)
    endif()
else()
    option(LIBNYQUIST_BUILD_EXAMPLE "Build example application" ON)
endif()


# libnyquist static library
if(USE_CUDA)
    project(libnyquist LANGUAGES C CXX CUDA)
else()
    project(libnyquist LANGUAGES C CXX)
    add_compile_definitions(ENABLE_FFT_TEST)
endif()

file(GLOB nyquist_include "${LIBNYQUIST_ROOT}/include/libnyquist/*")
file(GLOB nyquist_src     "${LIBNYQUIST_ROOT}/src/*")

add_library(libnyquist STATIC
    ${nyquist_include}
    ${nyquist_src}
)

set_cxx_version(libnyquist)
_set_compile_options(libnyquist)

if (WIN32)
    _disable_warning(4244)
    _disable_warning(4018)
endif()

target_include_directories(libnyquist
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${LIBNYQUIST_ROOT}/include>
  PRIVATE
    ${LIBNYQUIST_ROOT}/include/libnyquist
    ${LIBNYQUIST_ROOT}/third_party
    ${LIBNYQUIST_ROOT}/third_party/FLAC/src/include
    ${LIBNYQUIST_ROOT}/third_party/libogg/include
    ${LIBNYQUIST_ROOT}/third_party/libvorbis/include
    ${LIBNYQUIST_ROOT}/third_party/libvorbis/src
    ${LIBNYQUIST_ROOT}/third_party/musepack/include
    ${LIBNYQUIST_ROOT}/third_party/opus/celt
    ${LIBNYQUIST_ROOT}/third_party/opus/libopus/include
    ${LIBNYQUIST_ROOT}/third_party/opus/opusfile/include
    ${LIBNYQUIST_ROOT}/third_party/opus/opusfile/src/include
    ${LIBNYQUIST_ROOT}/third_party/opus/silk
    ${LIBNYQUIST_ROOT}/third_party/opus/silk/float
    ${LIBNYQUIST_ROOT}/third_party/wavpack/include
    ${LIBNYQUIST_ROOT}/src
)

if (USE_CUDA)
  add_subdirectory(cuda)
  target_compile_definitions(libnyquist PUBLIC USE_CUDA)
endif()

if (ENABLE_CPU_PROFILING)
    message(STATUS "CPU profiling enabled")
    target_compile_definitions(libnyquist PRIVATE MDCT_PROFILE)
endif()


if (MSVC_IDE)
    # hack to get around the "Debug" and "Release" directories cmake tries to add on Windows
    #set_target_properties(libnyquist PROPERTIES PREFIX "../")
    set_target_properties(libnyquist PROPERTIES IMPORT_PREFIX "../")
endif()

set_target_properties(libnyquist PROPERTIES OUTPUT_NAME_DEBUG libnyquist_d)

set_target_properties(libnyquist
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_INSTALL_BINDIR}"
)

add_library(libnyquist::libnyquist ALIAS libnyquist)

# install the libnyquist binaries
install(TARGETS libnyquist
        EXPORT libnyquist
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

# install the cmake configuration files
install(EXPORT libnyquist
    FILE libnyquistConfig.cmake
    NAMESPACE libnyquist::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/libnyquist"
)

message(STATUS "CMAKE_INSTALL_INCLUDEDIR: ${CMAKE_INSTALL_INCLUDEDIR}")

# todo, install the headers

# export the targets for use when libnyquist is included as a subproject
export(TARGETS libnyquist
       NAMESPACE libnyquist::
       FILE "${CMAKE_CURRENT_BINARY_DIR}/libnyquistConfig.cmake"
)

# folders
source_group(src FILES ${nyquist_src})

#-------------------------------------------------------------------------------

# libnyquist-examples

if(LIBNYQUIST_BUILD_EXAMPLE)

    set(NQR_EXAMPLE_APP_NAME "libnyquist-examples")

    set(
        EXAMPLE_SOURCES
        ${LIBNYQUIST_ROOT}/examples/src/Main.cpp
    )

    add_executable(${NQR_EXAMPLE_APP_NAME} ${EXAMPLE_SOURCES})

    if(WIN32)
        target_compile_definitions(${NQR_EXAMPLE_APP_NAME} PRIVATE __WINDOWS_WASAPI__)
    elseif(APPLE)
        target_compile_definitions(${NQR_EXAMPLE_APP_NAME} PRIVATE __MACOSX_CORE__)
    elseif(LIBNYQUIST_JACK)
        target_compile_definitions(${NQR_EXAMPLE_APP_NAME} PRIVATE __UNIX_JACK__)
        target_link_libraries(${NQR_EXAMPLE_APP_NAME} PRIVATE jack pthread)
    elseif(LIBNYQUIST_PULSE)
        find_library(PULSESIMPLE_LIB pulse-simple)
        target_compile_definitions(${NQR_EXAMPLE_APP_NAME} PRIVATE __LINUX_PULSE__)
        target_link_libraries(${NQR_EXAMPLE_APP_NAME} PRIVATE pulse ${PULSESIMPLE_LIB} pthread)
    elseif(LIBNYQUIST_ASOUND)
        target_compile_definitions(${NQR_EXAMPLE_APP_NAME} PRIVATE __LINUX_ALSA__)
        target_link_libraries(${NQR_EXAMPLE_APP_NAME} PRIVATE asound pthread)
    else()
        # message(FATAL, "On Linux, one of LIBNYQUIST_JACK, LIBNYQUIST_PULSE, or LIBNYQUIST_ASOUND must be set.")
    endif()

    target_include_directories(${NQR_EXAMPLE_APP_NAME} PRIVATE
        ${LIBNYQUIST_ROOT}/examples/src
        ${LIBNYQUIST_ROOT}/third_party
    )
    target_link_libraries(${NQR_EXAMPLE_APP_NAME} PRIVATE libnyquist)

    if(USE_CUDA)
      target_link_libraries(${NQR_EXAMPLE_APP_NAME} PRIVATE libnyquist_cuda)
      target_link_libraries(${NQR_EXAMPLE_APP_NAME} PRIVATE ${CUDA_LIBRARIES})
    endif()

    set_target_properties(${NQR_EXAMPLE_APP_NAME}
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    if(APPLE)
        target_link_libraries(${NQR_EXAMPLE_APP_NAME} PRIVATE
            "-framework AudioToolbox"
            "-framework AudioUnit"
            "-framework Accelerate"
            "-framework CoreAudio"
            "-framework Cocoa"
        )
    ENDIF(APPLE)

endif()


enable_testing()

# Add test for libnyquist-examples
if (LIBNYQUIST_BUILD_EXAMPLE)
    add_test(NAME libnyquist_example_test
            COMMAND $<TARGET_FILE:libnyquist-examples>
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            )

    add_test(NAME libnyquist_example_test_8c
            COMMAND $<TARGET_FILE:libnyquist-examples> ./test_data/Rachel8ch.opus
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            )
endif()