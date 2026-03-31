set(TARGET bs-deltacalc)
add_executable(${TARGET} bs-deltacalc.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
endif()

# Standalone test for the three-tier ray march kernel
add_executable(test-ray-march test-ray-march.cpp)
target_link_libraries(test-ray-march PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(test-ray-march PRIVATE cxx_std_17)
target_include_directories(test-ray-march PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)
