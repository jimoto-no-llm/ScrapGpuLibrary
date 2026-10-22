import sys
root = sys.argv[1]
def edit(path, old, new):
    p = root + '/' + path
    s = open(p).read()
    assert old in s, (path, old[:60])
    open(p, 'w').write(s.replace(old, new, 1))

edit('adapters/ggml/composite/composite.h', '''// Called once by the CL11 registry''', '''// an NVIDIA GPU through the CUDA backend of src/backends/cuda (pdev_cuda.cpp); `index` is the number in "CUDA<index>". nullptr when the plugin is
// built without CL11_WITH_CUDA, there is no libcuda, or there is no such device.
std::unique_ptr<pdev> make_pdev_cuda(int index);

// Called once by the CL11 registry''')
edit('adapters/ggml/composite/composite_device.cpp', '''        } else {
            ggml_backend_dev_t d = find_ggml_device(s.first.c_str());''', '''        } else if (s.first.size() > 4 && s.first.rfind("CUDA", 0) == 0 && s.first.find_first_not_of("0123456789", 4) == std::string::npos) {
            p = make_pdev_cuda(atoi(s.first.c_str() + 4));
            if (!p) {
                SGLX_LOG("GGML_SGLX_SPLIT: no CUDA device %s (plugin built without CL11_WITH_CUDA, no libcuda, or not in SCRAPGPU_CUDA_DEVICES)\\n", s.first.c_str());
                return false;
            }
        } else {
            ggml_backend_dev_t d = find_ggml_device(s.first.c_str());''')
edit('adapters/ggml/CMakeLists.txt', '''    composite/pdev_cl11.cpp
''', '''    composite/pdev_cl11.cpp
    composite/pdev_cuda.cpp
''')
edit('adapters/ggml/CMakeLists.txt', '''target_link_libraries(${TARGET_NAME} PRIVATE scrapgpu_opencl11 scrapgpu_tensor ${OpenCL_LIBRARIES})''', '''# NVIDIA GPUs as physical devices of the SGLX composite device through the CUDA Driver API ("CUDA0", "CUDA1", ... in GGML_SGLX_SPLIT, see
# composite/pdev_cuda.cpp): the CUDA backend of src/backends/cuda, with libcuda loaded at run time. No CUDA toolkit and no libcuda are needed to
# build or to load the plugin (without libcuda there are 0 CUDA devices). Out-of-tree builds only.
option(CL11_WITH_CUDA "SGLX: NVIDIA GPUs through the CUDA Driver API (composite/pdev_cuda.cpp, src/backends/cuda)" OFF)
if (CL11_WITH_CUDA)
    if (NOT EXISTS "${CL11_SRC_DIR}/backends/cuda/device.cpp")
        message(FATAL_ERROR "CL11_WITH_CUDA needs the repository tree (src/backends/cuda), not the in-tree copy")
    endif()
    target_sources(${TARGET_NAME} PRIVATE
        "${CL11_SRC_DIR}/backends/cuda/driver.cpp"
        "${CL11_SRC_DIR}/backends/cuda/device.cpp"
        "${CL11_SRC_DIR}/backends/cuda/r32_transfer.cpp")
    target_include_directories(${TARGET_NAME} PRIVATE "${CL11_SRC_DIR}/backends")
    target_compile_definitions(${TARGET_NAME} PRIVATE SGLX_WITH_CUDA)
    target_link_libraries(${TARGET_NAME} PRIVATE ${CMAKE_DL_LIBS})
endif()

target_link_libraries(${TARGET_NAME} PRIVATE scrapgpu_opencl11 scrapgpu_tensor ${OpenCL_LIBRARIES})''')
