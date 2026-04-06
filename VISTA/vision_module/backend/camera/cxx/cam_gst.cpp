#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <iostream>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace aidlux_cam {

struct BufferData {
    GstSample* sample;
    GstMapInfo map;
};

class HardwareCamera {
private:
    GstElement *pipeline = nullptr;
    GstElement *appsink = nullptr;
    int out_width, out_height;
    int channels;

public:
    // 构造函数新增 fps 参数
    HardwareCamera(const std::string& device, 
                   int in_w, int in_h, 
                   int out_w, int out_h, 
                   int fps = 30,                         // 新增：帧率控制
                   const std::string& in_format="YUY2", 
                   const std::string& out_format="RGB",  
                   bool flip_h=false, bool flip_v=false, 
                   int rotate=0,
                   int crop_x=0, int crop_y=0, int crop_w=0, int crop_h=0) 
                   : out_width(out_w), out_height(out_h) {
        
        if (!gst_is_initialized()) {
            gst_init(nullptr, nullptr);
        }

        // 卸载深度图逻辑后，通道判断变得非常纯粹
        channels = (out_format == "RGB" || out_format == "BGR") ? 3 : 1;

        // 1. 动态拼接硬件转换属性
        std::string transform_props = "";
        if (flip_h) transform_props += " flip-horizontal=true";
        if (flip_v) transform_props += " flip-vertical=true";
        if (rotate != 0) transform_props += " rotate=" + std::to_string(rotate);

        // 2. 拼接硬件裁剪属性
        if (crop_w > 0 && crop_h > 0) {
            transform_props += " crop=\"<" + 
                               std::to_string(crop_x) + "," +
                               std::to_string(crop_y) + "," +
                               std::to_string(crop_w) + "," +
                               std::to_string(crop_h) + ">\"";
        }

        // 3. 构建纯净的硬件加速管道，加入 framerate 控制
        // 注意：GStreamer 的帧率格式必须是 分数形式，例如 30/1
        std::string pipe_str = 
            "v4l2src device=" + device + " ! " +
            "video/x-raw,format=" + in_format + ",width=" + std::to_string(in_w) + ",height=" + std::to_string(in_h) + ",framerate=" + std::to_string(fps) + "/1 ! " +
            "qtivtransform" + transform_props + " ! " +
            "video/x-raw,format=" + out_format + ",width=" + std::to_string(out_w) + ",height=" + std::to_string(out_h) + " ! " +
            "appsink name=mysink drop=true max-buffers=2 sync=false";

        std::cout << "🔗 [AidLux Cam] 启动硬件加速管道: " << pipe_str << std::endl;

        GError *error = nullptr;
        pipeline = gst_parse_launch(pipe_str.c_str(), &error);
        if (error) {
            std::string err_msg = "Pipeline 编译失败: " + std::string(error->message);
            g_clear_error(&error);
            throw std::runtime_error(err_msg);
        }

        appsink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
        gst_element_set_state(pipeline, GST_STATE_PLAYING);
    }

    ~HardwareCamera() {
        if (pipeline) {
            gst_element_set_state(pipeline, GST_STATE_NULL);
            gst_object_unref(pipeline);
        }
    }

    py::array read_frame() {
        GstSample *sample = nullptr;

        {
            py::gil_scoped_release release; 
            sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink));
        }

        if (!sample) return py::array();

        BufferData* data = new BufferData();
        data->sample = sample;
        GstBuffer *buffer = gst_sample_get_buffer(sample);
        
        if (!gst_buffer_map(buffer, &data->map, GST_MAP_READ)) {
            gst_sample_unref(sample);
            delete data;
            return py::array();
        }

        py::capsule free_when_done(data, [](void *p) {
            BufferData* d = reinterpret_cast<BufferData*>(p);
            GstBuffer* buf = gst_sample_get_buffer(d->sample);
            gst_buffer_unmap(buf, &d->map); 
            gst_sample_unref(d->sample);    
            delete d;
        });

        // 纯净的 8 位图像返回逻辑 (支持 RGB/BGR 3通道，或 GRAY8 单通道)
        if (channels == 3) {
            return py::array_t<uint8_t>(
                {out_height, out_width, channels}, 
                {out_width * channels, channels, 1}, 
                data->map.data,              
                free_when_done                
            );
        } else {
            return py::array_t<uint8_t>(
                {out_height, out_width}, 
                {out_width, 1}, 
                data->map.data,              
                free_when_done                
            );
        }
    }
};

} // namespace aidlux_cam

// ==========================================
// Python 绑定部分
// ==========================================
PYBIND11_MODULE(fast_cam, m) {
    m.doc() = "AidLux Hardware Accelerated Camera Module (Optimized for RGB/IR)";
    
    py::class_<aidlux_cam::HardwareCamera>(m, "Camera")
        .def(py::init<const std::string&, int, int, int, int, int, const std::string&, const std::string&, bool, bool, int, int, int, int, int>(),
             py::arg("device"), 
             py::arg("in_w"), py::arg("in_h"), 
             py::arg("out_w"), py::arg("out_h"),
             py::arg("fps") = 30,             // 新增的 fps 参数绑定
             py::arg("in_format") = "YUY2",   
             py::arg("out_format") = "RGB",   
             py::arg("flip_h") = false, 
             py::arg("flip_v") = false, 
             py::arg("rotate") = 0,
             py::arg("crop_x") = 0, py::arg("crop_y") = 0, 
             py::arg("crop_w") = 0, py::arg("crop_h") = 0) 
        .def("read_frame", &aidlux_cam::HardwareCamera::read_frame);
}