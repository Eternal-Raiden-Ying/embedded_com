#include <iostream>
#include <string>
#include <algorithm>
#include <cctype>
#include <cstring> // 用于 memcpy
#include <opencv2/opencv.hpp>
#include <aidlux/aidlite/aidlite.hpp>
#include <vector>
#include <numeric>

using namespace cv;
using namespace std;
using namespace Aidlux::Aidlite;

const int out_size = 8400;
const float CONF_T = 0.25f;
const float SCORE_T = 0.25f;
const float NMS_CONF_T = 0.25f;
const float NMS_IOU_T = 0.45f;

const std::vector<std::string> class_list = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"};

struct Args
{
    std::string target_model = "../../models/cutoff_yolov8s-seg_qcs6490_w8a8.qnn236.ctx.bin";
    std::string imgs = "../bus.jpg";
    int invoke_nums = 1;
    std::string model_type = "QNN";
};

Args parse_args(int argc, char *argv[])
{
    Args args;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--target_model" && i + 1 < argc)
        {
            args.target_model = argv[++i];
        }
        else if (arg == "--imgs" && i + 1 < argc)
        {
            args.imgs = argv[++i];
        }
        else if (arg == "--invoke_nums" && i + 1 < argc)
        {
            args.invoke_nums = std::stoi(argv[++i]);
        }
        else if (arg == "--model_type" && i + 1 < argc)
        {
            args.model_type = argv[++i];
        }
    }
    return args;
}

std::string to_lower(const std::string &str)
{
    std::string lower_str = str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), [](unsigned char c)
                   { return std::tolower(c); });
    return lower_str;
}

// concatenate_3(qnn4, qnn80 , qnn32, 1, 8400, 4, 80,32, qnn_concat);
void concatenate_3(float *qnn_4, float *qnn_80, float *qnn_32, int batch, int num_elements, int dim4, int dim80, int dim32, std::vector<float> &output)
{
    int out_dim = dim4 + dim80 + dim32; // 116
    output.resize(batch * num_elements * out_dim);
    for (int i = 0; i < batch * num_elements; ++i)
    {
        std::memcpy(&output[i * out_dim], &qnn_4[i * dim4], dim4 * sizeof(float));
        std::memcpy(&output[i * out_dim + dim4], &qnn_80[i * dim80], dim80 * sizeof(float));
        std::memcpy(&output[i * out_dim + dim4 + dim80], &qnn_32[i * dim32], dim32 * sizeof(float));
    }
}

// 4*8400
void transformData(const float *input, float *output, int C, int N)
{
    for (int c = 0; c < C; ++c)
    {
        for (int n = 0; n < N; ++n)
        {
            output[n * C + c] = input[c * N + n];
        }
    }
}

double img_process(cv::Mat frame, cv::Mat &img_input, int size)
{
    cv::Mat img_processed = frame.clone();
    int height = img_processed.rows;
    int width = img_processed.cols;
    int length = std::max(height, width);
    double scala = static_cast<double>(length) / size;

    cv::Mat image = cv::Mat::zeros(cv::Size(length, length), CV_8UC3);
    img_processed.copyTo(image(cv::Rect(0, 0, width, height)));

    cv::cvtColor(image, img_input, cv::COLOR_BGR2RGB);
    cv::resize(img_input, img_input, cv::Size(size, size));

    cv::Mat mean_data = cv::Mat::zeros(img_input.size(), CV_32FC3);
    cv::Mat std_data(img_input.size(), CV_32FC3, cv::Scalar(255, 255, 255));
    img_input.convertTo(img_input, CV_32FC3);
    img_input = (img_input - mean_data) / std_data;
    return scala;
}

// -------------------- process_mask_vector --------------------
// protos: CV_32F (C x HW) e.g. 32 x 25600
// masks_in: CV_32F (N x C) e.g. N x 32
// boxes: vector<Rect> length N (in original image coords)
// im0_size: original image size (width,height)
static std::vector<cv::Mat> process_mask_vector(const cv::Mat &protos,
                                                const cv::Mat &masks_in,
                                                const std::vector<cv::Rect> &boxes,
                                                const cv::Size &im0_size)
{
    CV_Assert(protos.type() == CV_32F && masks_in.type() == CV_32F);
    int N = masks_in.rows;
    int C = masks_in.cols;
    int HW = protos.cols; // mh*mw
    int mh = (int)std::round(std::sqrt((double)HW));
    int mw = HW / mh;

    // gemm: (N x C) * (C x HW) = N x HW
    cv::Mat mask_flat; // N x HW, CV_32F
    cv::gemm(masks_in, protos, 1.0, cv::Mat(), 0.0, mask_flat);

    std::vector<cv::Mat> masks_out;
    masks_out.reserve(N);

    for (int i = 0; i < N; ++i)
    {
        // 1) row i -> reshape to mh x mw (float)
        cv::Mat row = mask_flat.row(i);      // 1 x HW
        cv::Mat mask_f = row.reshape(1, mh); // mh x mw (shared if continuous)

        // 2) resize to original image size
        cv::Mat mask_resized;
        cv::resize(mask_f, mask_resized, im0_size, 0, 0, cv::INTER_LINEAR); // float

        // 3) threshold > 0.5 (同 Python greater)
        cv::Mat mask_bin;
        cv::threshold(mask_resized, mask_bin, 0.5, 255.0, cv::THRESH_BINARY);

        // 4) convert to CV_8U
        mask_bin.convertTo(mask_bin, CV_8U);

        // 5) crop to the bounding box: zero-out outside the box
        cv::Mat final_mask = cv::Mat::zeros(im0_size, CV_8U);
        cv::Rect roi = boxes[i] & cv::Rect(0, 0, im0_size.width, im0_size.height);
        if (roi.width > 0 && roi.height > 0)
        {
            // copy corresponding region (mask_bin and final_mask sizes are equal, so direct roi)
            mask_bin(roi).copyTo(final_mask(roi));
        }
        // push
        masks_out.push_back(final_mask);
    }

    return masks_out; // each element CV_8U HxW
}

// -------------------- draw detections + masks --------------------
static cv::Mat draw_detect_res_with_masks(const cv::Mat &src_img,
                                          const std::vector<cv::Rect> &boxes,
                                          const std::vector<int> &class_ids,
                                          const std::vector<cv::Mat> &masks, // per-instance mask CV_8U
                                          const std::vector<std::string> &class_name)
{
    CV_Assert(boxes.size() == class_ids.size());
    CV_Assert(boxes.size() == masks.size());

    cv::Mat img = src_img.clone();
    cv::Mat canvas = src_img.clone();

    int num_classes = std::max(1, (int)class_name.size());
    int color_step = int(255.0 / num_classes);

    for (size_t i = 0; i < boxes.size(); ++i)
    {
        const cv::Rect &box = boxes[i];
        int cls_id = class_ids[i];
        cv::Scalar color(0, int(cls_id * color_step), int(255 - cls_id * color_step));

        // draw box & label
        cv::rectangle(img, box, color, 2);
        if (!class_name.empty() && cls_id >= 0 && cls_id < (int)class_name.size())
        {
            cv::putText(img, class_name[cls_id], cv::Point(box.x, std::max(0, box.y - 6)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }

        // draw mask if valid
        if (!masks[i].empty() && cv::countNonZero(masks[i]) > 0)
        {
            // find contours for the mask
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(masks[i], contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            if (!contours.empty())
            {
                // fill contours on overlay
                cv::Mat overlay = img.clone();
                cv::fillPoly(overlay, contours, color);
                // blend overlay to img
                double alpha = 0.6; // mask alpha
                cv::addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img);
            }
        }
    }

    // slight global blend like Python code
    cv::addWeighted(canvas, 0.3, img, 0.7, 0, img);
    return img;
}

// -------------------- post_process --------------------
cv::Mat post_process_fixed(cv::Mat &frame, std::vector<float> &outputs,
                           const std::vector<std::string> &class_name,
                           float ratio, float *protos_data)
{
    cv::Mat input_image = frame.clone();

    // 1. protos matrix (32 x 160*160) CV_32F
    const int C = 32;
    const int MH = 160;
    const int MW = 160;
    const int HW = MH * MW;
    cv::Mat protos(C, HW, CV_32F);
    for (int c = 0; c < C; ++c)
    {
        float *rowPtr = protos.ptr<float>(c);
        int col = 0;
        for (int y = 0; y < MH; ++y)
        {
            for (int x = 0; x < MW; ++x)
            {
                int idx = (y * MW + x) * C + c; // use this for HWC protos_data
                rowPtr[col++] = protos_data[idx];
            }
        }
    }

    // 2. parse detections and collect mask coeffs
    std::vector<int> class_ids_all;
    std::vector<float> confidences_all;
    std::vector<cv::Rect> boxes_all;
    std::vector<cv::Mat> coeffs_all; // each 1x32 CV_32F

    int stride = 116;
    int total = (int)outputs.size();
    for (int i = 0; i + stride <= total; i += stride)
    {
        // find max class score in outputs[i+4 : i+84)
        auto it_begin = outputs.begin() + i + 4;
        auto it_end = outputs.begin() + i + 84;
        auto max_iter = std::max_element(it_begin, it_end);
        float max_score = (max_iter == it_end) ? 0.0f : *max_iter;
        if (max_score < SCORE_T)
            continue;

        int cls_id = (int)std::distance(it_begin, max_iter);
        float confidence = max_score;

        // box center cx,cy and w,h are in outputs[i+0..3]
        float cx = outputs[i + 0];
        float cy = outputs[i + 1];
        float w = outputs[i + 2];
        float h = outputs[i + 3];

        int left = int((cx - w / 2.0f) * ratio);
        int top = int((cy - h / 2.0f) * ratio);
        int width = int(w * ratio);
        int height = int(h * ratio);
        cv::Rect rect(left, top, width, height);
        // normalize/cap the rect to image bounds later

        boxes_all.push_back(rect);
        class_ids_all.push_back(cls_id);
        confidences_all.push_back(confidence);

        // mask coeffs: outputs[i+84 .. i+115] (32 values)
        cv::Mat coeff(1, C, CV_32F);
        for (int k = 0; k < C; ++k)
            coeff.at<float>(0, k) = outputs[i + 84 + k];
        coeffs_all.push_back(coeff);
    }

    if (boxes_all.empty())
        return frame;

    // 3. NMS (use confidences_all)
    std::vector<int> keep_idx;
    cv::dnn::NMSBoxes(boxes_all, confidences_all, NMS_CONF_T, NMS_IOU_T, keep_idx);
    printf("Detected {%ld} targets.\n", keep_idx.size());

    if (keep_idx.empty())
        return frame;

    // build final vectors in NMS order
    std::vector<cv::Rect> nms_boxes;
    std::vector<int> nms_classes;
    std::vector<cv::Mat> nms_coeffs;
    for (int id : keep_idx)
    {
        nms_boxes.push_back(boxes_all[id]);
        nms_classes.push_back(class_ids_all[id]);
        nms_coeffs.push_back(coeffs_all[id]);
    }

    // 4. build masks_in matrix (N x 32) CV_32F, continuous
    int N = (int)nms_coeffs.size();
    cv::Mat masks_in(N, C, CV_32F);
    for (int i = 0; i < N; ++i)
    {
        CV_Assert(nms_coeffs[i].cols == C && nms_coeffs[i].type() == CV_32F);
        nms_coeffs[i].copyTo(masks_in.row(i));
    }

    // 5. decode masks -> vector<Mat> of CV_8U HxW per-instance
    std::vector<cv::Rect> boxes_for_masks = nms_boxes; // same order
    std::vector<cv::Mat> masks_decoded = process_mask_vector(protos, masks_in, boxes_for_masks, frame.size());

    // debug: print non-zero counts
    for (size_t i = 0; i < masks_decoded.size(); ++i)
    {
        int nz = cv::countNonZero(masks_decoded[i]);
        std::cout << "[mask debug] idx=" << i << " nonzero=" << nz << " box=" << nms_boxes[i] << std::endl;
    }

    // 6. draw boxes + masks
    cv::Mat result = draw_detect_res_with_masks(frame, nms_boxes, nms_classes, masks_decoded, class_name);
    return result;
}

int invoke(const Args &args)
{
    std::cout << "Start main ... ... Model Path: " << args.target_model << "\n"
              << "Image Path: " << args.imgs << "\n"
              << "Inference Nums: " << args.invoke_nums << "\n"
              << "Model Type: " << args.model_type << "\n";
    Model *model = Model::create_instance(args.target_model);
    if (model == nullptr)
    {
        printf("Create model failed !\n");
        return EXIT_FAILURE;
    }
    Config *config = Config::create_instance();
    if (config == nullptr)
    {
        printf("Create config failed !\n");
        return EXIT_FAILURE;
    }
    config->implement_type = ImplementType::TYPE_LOCAL;
    std::string model_type_lower = to_lower(args.model_type);
    if (model_type_lower == "qnn")
    {
        config->framework_type = FrameworkType::TYPE_QNN;
    }
    else if (model_type_lower == "snpe2" || model_type_lower == "snpe")
    {
        config->framework_type = FrameworkType::TYPE_SNPE2;
    }
    config->accelerate_type = AccelerateType::TYPE_DSP;
    config->is_quantify_model = 1;

    std::vector<std::vector<uint32_t>> input_shapes = {{1, 640, 640, 3}};
    std::vector<std::vector<uint32_t>> output_shapes = {{1, 32, 8400}, {1, 4, 8400}, {1, 80, 8400}, {1, 160, 160, 32}};
    model->set_model_properties(input_shapes, Aidlux::Aidlite::DataType::TYPE_FLOAT32, output_shapes, Aidlux::Aidlite::DataType::TYPE_FLOAT32);
    std::unique_ptr<Interpreter> fast_interpreter = InterpreterBuilder::build_interpretper_from_model_and_config(model, config);
    if (fast_interpreter == nullptr)
    {
        printf("build_interpretper_from_model_and_config failed !\n");
        return EXIT_FAILURE;
    }
    int result = fast_interpreter->init();
    if (result != EXIT_SUCCESS)
    {
        printf("interpreter->init() failed !\n");
        return EXIT_FAILURE;
    }
    // load model
    fast_interpreter->load_model();
    if (result != EXIT_SUCCESS)
    {
        printf("interpreter->load_model() failed !\n");
        return EXIT_FAILURE;
    }
    printf("detect model load success!\n");

    cv::Mat frame = cv::imread(args.imgs);
    if (frame.empty())
    {
        printf("detect image load failed!\n");
        return 1;
    }
    printf("img_src cols: %d, img_src rows: %d\n", frame.cols, frame.rows);
    cv::Mat input_img;
    double scale = img_process(frame, input_img, 640);
    if (input_img.empty())
    {
        printf("detect input_img load failed!\n");
        return 1;
    }

    float *qnn_seg_data = nullptr;
    float *qnn_trans_data = nullptr;
    float *qnn_mul_data = nullptr;
    float *qnn_canc_data = nullptr;

    std::vector<float> invoke_time;
    for (int i = 0; i < args.invoke_nums; ++i)
    {
        result = fast_interpreter->set_input_tensor(0, input_img.data);
        if (result != EXIT_SUCCESS)
        {
            printf("interpreter->set_input_tensor() failed !\n");
            return EXIT_FAILURE;
        }
        // 开始计时
        auto t1 = std::chrono::high_resolution_clock::now();
        result = fast_interpreter->invoke();
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cost_time = t2 - t1;
        invoke_time.push_back(cost_time.count() * 1000);
        if (result != EXIT_SUCCESS)
        {
            printf("interpreter->invoke() failed !\n");
            return EXIT_FAILURE;
        }
        uint32_t out_data_32 = 0;
        result = fast_interpreter->get_output_tensor(1, (void **)&qnn_canc_data, &out_data_32);
        if (result != EXIT_SUCCESS)
        {
            printf("interpreter->get_output_tensor() 1 failed !\n");
            return EXIT_FAILURE;
        }
        std::cout << "out 1 length: " << out_data_32 / 4 << std::endl; // 26800

        uint32_t out_data_4 = 0;
        result = fast_interpreter->get_output_tensor(2, (void **)&qnn_trans_data, &out_data_4);
        if (result != EXIT_SUCCESS)
        {
            printf("interpreter->get_output_tensor() 2 failed !\n");
            return EXIT_FAILURE;
        }
        std::cout << "out 2 length: " << out_data_4 / 4 << std::endl; // 33600

        uint32_t out_data_2 = 0;
        result = fast_interpreter->get_output_tensor(0, (void **)&qnn_seg_data, &out_data_2);
        if (result != EXIT_SUCCESS)
        {
            printf("interpreter->get_output_tensor() 2 failed !\n");
            return EXIT_FAILURE;
        }
        std::cout << "out 0 length: " << out_data_2 / 4 << std::endl; // 819200

        uint32_t out_data_80 = 0;
        result = fast_interpreter->get_output_tensor(3, (void **)&qnn_mul_data, &out_data_80);
        if (result != EXIT_SUCCESS)
        {
            printf("interpreter->get_output_tensor() 2 failed !\n");
            return EXIT_FAILURE;
        }
        std::cout << "out 3 length: " << out_data_80 / 4 << std::endl; // 67200
    }

    float max_invoke_time = *std::max_element(invoke_time.begin(), invoke_time.end());
    float min_invoke_time = *std::min_element(invoke_time.begin(), invoke_time.end());
    float mean_invoke_time = std::accumulate(invoke_time.begin(), invoke_time.end(), 0.0f) / args.invoke_nums;
    float var_invoketime = 0.0f;
    for (auto time : invoke_time)
    {
        var_invoketime += (time - mean_invoke_time) * (time - mean_invoke_time);
    }
    var_invoketime /= args.invoke_nums;
    printf("=======================================\n");
    printf("QNN inference %d times :\n --mean_invoke_time is %f \n --max_invoke_time is %f \n --min_invoke_time is %f \n --var_invoketime is %f\n",
           args.invoke_nums, mean_invoke_time, max_invoke_time, min_invoke_time, var_invoketime);
    printf("=======================================\n");

    float *pos_data = new float[4 * out_size];
    float *class_data = new float[80 * out_size];
    float *canc_data = new float[32 * out_size];
    transformData(qnn_trans_data, pos_data, 4, out_size);
    transformData(qnn_mul_data, class_data, 80, out_size);
    transformData(qnn_canc_data, canc_data, 32, out_size);

    // post process
    std::vector<float> qnn_concat;
    concatenate_3(pos_data, class_data, canc_data, 1, out_size, 4, 80, 32, qnn_concat);
    cv::Mat img = post_process_fixed(frame, qnn_concat, class_list, scale, qnn_seg_data);
    cv::imwrite("./results.jpg", img);
    fast_interpreter->destory();
    return 0;
}

int main(int argc, char *argv[])
{
    Args args = parse_args(argc, argv);
    return invoke(args);
}
