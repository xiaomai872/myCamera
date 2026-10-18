
cmake_minimum_required(VERSION 3.10)

project(CameraApp)

if(WIN32)
    set(OpenCV_DIR "D:/DevTools/OpenCV/4.5.5/x64/mingw/lib/cmake/opencv4")
endif()

find_package(OpenCV REQUIRED)

add_executable(camera_app io/example.cpp myCamera.cpp)

target_link_libraries(camera_app PRIVATE ${OpenCV_LIBS})

