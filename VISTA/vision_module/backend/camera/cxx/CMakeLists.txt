cmake_minimum_required(VERSION 3.4)
project(fast_cam)

# 告诉 CMake 这是一个 C++11 项目
set(CMAKE_CXX_STANDARD 11)

# 寻找 Python 和 Pybind11 包
find_package(PythonLibs REQUIRED)
# 寻找 pybind11，如果在 AidLux 报错找不到，我们等下手动安装
find_package(pybind11 REQUIRED)

# 寻找 GStreamer 库
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-app-1.0)

# 包含头文件路径
include_directories(${PYTHON_INCLUDE_DIRS} ${GST_INCLUDE_DIRS})

# 创建 Python 扩展模块 (名字必须和 PYBIND11_MODULE 里的一致)
pybind11_add_module(fast_cam cam_gst.cpp)

# 链接 GStreamer 库
target_link_libraries(fast_cam PRIVATE ${GST_LIBRARIES})
