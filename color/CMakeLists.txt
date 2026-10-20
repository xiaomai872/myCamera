add_executable(myColorCorrect_app
    example.cpp
    myColorCorrect.cpp)

target_include_directories(myColorCorrect_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS})

target_link_libraries(myColorCorrect_app PRIVATE ${OpenCV_LIBS})