cmake_minimum_required(VERSION 3.10)

project(myColorCorrectApp)

if(WIN32)
    set(OpenCV_DIR "D:/DevTools/OpenCV/4.5.5/x64/mingw/lib/cmake/opencv4")
endif()

find_package(OpenCV REQUIRED)

add_executable(myColorCorrect_app
    example.cpp
    myColorCorrect.cpp)

target_include_directories(myColorCorrect_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS})

target_link_libraries(myColorCorrect_app PRIVATE ${OpenCV_LIBS})
