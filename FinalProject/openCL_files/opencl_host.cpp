#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <OpenCL/opencl.h>
#include <opencv2/opencv.hpp>

// Function to load kernel source code from file
std::string loadKernelSource(const std::string &filePath)
{
    std::ifstream file(filePath);
    if (!file.is_open())
    {
        std::cerr << "Failed to open kernel source file: " << filePath << std::endl;
        exit(EXIT_FAILURE);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

int main()
{
    // Load input images
    cv::Mat left_img = cv::imread("../images/im0.png", cv::IMREAD_GRAYSCALE);
    cv::Mat right_img = cv::imread("../images/im1.png", cv::IMREAD_GRAYSCALE);

    if (left_img.empty() || right_img.empty())
    {
        std::cerr << "Error loading images" << std::endl;
        return -1;
    }

    int width = left_img.cols;
    int height = left_img.rows;
    int image_downsample_factor = 4; // Example resize image_downsample_factor
    int resized_width = width / image_downsample_factor;
    int resized_height = height / image_downsample_factor;
    int winSize = 21;             // Example window size
    int maxDisp = 260;            // Example maximum disparity
    int threshold = 8;            // Example threshold for cross-checking
    int max_search_distance = 10; // Example max search distance for occlusion filling

    maxDisp = int(maxDisp / image_downsample_factor);

    // Initialize OpenCL
    cl_int err;
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error getting number of platforms: " << err << std::endl;
        return -1;
    }
    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error getting platforms: " << err << std::endl;
        return -1;
    }

    cl_platform_id platform = platforms.front();
    cl_uint numDevices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error getting number of devices: " << err << std::endl;
        return -1;
    }
    std::vector<cl_device_id> devices(numDevices);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error getting devices: " << err << std::endl;
        return -1;
    }

    cl_device_id device = devices.front();
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error creating context: " << err << std::endl;
        return -1;
    }

    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error creating command queue: " << err << std::endl;
        return -1;
    }

    // Load resize kernel source code from file
    std::string resizeKernelSource = loadKernelSource("resize_image_kernel.cl");
    const char *resizeKernelSourceCStr = resizeKernelSource.c_str();
    cl_program resizeProgram = clCreateProgramWithSource(context, 1, &resizeKernelSourceCStr, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error creating resize program: " << err << std::endl;
        return -1;
    }

    err = clBuildProgram(resizeProgram, 1, &device, nullptr, nullptr, nullptr);
    size_t resizeLogSize;
    clGetProgramBuildInfo(resizeProgram, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &resizeLogSize);
    std::vector<char> resizeLog(resizeLogSize);
    clGetProgramBuildInfo(resizeProgram, device, CL_PROGRAM_BUILD_LOG, resizeLogSize, resizeLog.data(), nullptr);
    std::cerr << "Resize Build log: " << resizeLog.data() << std::endl;

    if (err != CL_SUCCESS)
    {
        std::cerr << "Error building resize program: " << resizeLog.data() << std::endl;
        return -1;
    }

    cl_kernel resizeKernel = clCreateKernel(resizeProgram, "resize_image_kernel", &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error creating resize kernel: " << err << std::endl;
        return -1;
    }

    // Create buffers for resizing
    cl_mem leftImgBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, left_img.total() * left_img.elemSize(), left_img.data, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error creating left image buffer: " << err << std::endl;
        return -1;
    }
    cl_mem rightImgBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, right_img.total() * right_img.elemSize(), right_img.data, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error creating right image buffer: " << err << std::endl;
        return -1;
    }
    cl_mem resizedLeftImgBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, resized_width * resized_height * sizeof(uchar), nullptr, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error creating resized left image buffer: " << err << std::endl;
        return -1;
    }
    cl_mem resizedRightImgBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, resized_width * resized_height * sizeof(uchar), nullptr, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error creating resized right image buffer: " << err << std::endl;
        return -1;
    }

    // Set resize kernel arguments for left image
    err = clSetKernelArg(resizeKernel, 0, sizeof(cl_mem), &leftImgBuffer);
    err |= clSetKernelArg(resizeKernel, 1, sizeof(cl_mem), &resizedLeftImgBuffer);
    err |= clSetKernelArg(resizeKernel, 2, sizeof(int), &width);
    err |= clSetKernelArg(resizeKernel, 3, sizeof(int), &height);
    err |= clSetKernelArg(resizeKernel, 4, sizeof(int), &resized_width);
    err |= clSetKernelArg(resizeKernel, 5, sizeof(int), &resized_height);
    err |= clSetKernelArg(resizeKernel, 6, sizeof(int), &image_downsample_factor);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error setting resize kernel arguments for left image: " << err << std::endl;
        return -1;
    }

    // Execute the resize kernel for left image
    size_t resizeGlobalSize[] = {static_cast<size_t>(resized_width), static_cast<size_t>(resized_height)};
    cl_event resizeKernelEventLeft;
    err = clEnqueueNDRangeKernel(queue, resizeKernel, 2, nullptr, resizeGlobalSize, nullptr, 0, nullptr, &resizeKernelEventLeft);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error enqueuing resize kernel for left image: " << err << std::endl;
        return -1;
    }

    // Set resize kernel arguments for right image
    err = clSetKernelArg(resizeKernel, 0, sizeof(cl_mem), &rightImgBuffer);
    err |= clSetKernelArg(resizeKernel, 1, sizeof(cl_mem), &resizedRightImgBuffer);
    err |= clSetKernelArg(resizeKernel, 2, sizeof(int), &width);
    err |= clSetKernelArg(resizeKernel, 3, sizeof(int), &height);
    err |= clSetKernelArg(resizeKernel, 4, sizeof(int), &resized_width);
    err |= clSetKernelArg(resizeKernel, 5, sizeof(int), &resized_height);
    err |= clSetKernelArg(resizeKernel, 6, sizeof(int), &image_downsample_factor);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error setting resize kernel arguments for right image: " << err << std::endl;
        return -1;
    }

    // Execute the resize kernel for right image
    cl_event resizeKernelEventRight;
    err = clEnqueueNDRangeKernel(queue, resizeKernel, 2, nullptr, resizeGlobalSize, nullptr, 0, nullptr, &resizeKernelEventRight);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error enqueuing resize kernel for right image: " << err << std::endl;
        return -1;
    }

    // Wait for the resize kernels to finish
    clWaitForEvents(1, &resizeKernelEventLeft);
    clWaitForEvents(1, &resizeKernelEventRight);

    // Profiling information for resize kernels
    cl_ulong resizeStartLeft, resizeEndLeft, resizeStartRight, resizeEndRight;
    clGetEventProfilingInfo(resizeKernelEventLeft, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &resizeStartLeft, nullptr);
    clGetEventProfilingInfo(resizeKernelEventLeft, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &resizeEndLeft, nullptr);
    clGetEventProfilingInfo(resizeKernelEventRight, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &resizeStartRight, nullptr);
    clGetEventProfilingInfo(resizeKernelEventRight, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &resizeEndRight, nullptr);

    double resizeTimeLeft = (resizeEndLeft - resizeStartLeft) * 1.0e-6;    // Convert to milliseconds
    double resizeTimeRight = (resizeEndRight - resizeStartRight) * 1.0e-6; // Convert to milliseconds

    std::cout << "Resize Kernel (Left Image) Time: " << resizeTimeLeft << " ms" << std::endl;
    std::cout << "Resize Kernel (Right Image) Time: " << resizeTimeRight << " ms" << std::endl;

    // Read the resized images from the buffers
    std::vector<uchar> resizedLeftImgData(resized_width * resized_height);
    std::vector<uchar> resizedRightImgData(resized_width * resized_height);
    err = clEnqueueReadBuffer(queue, resizedLeftImgBuffer, CL_TRUE, 0, resizedLeftImgData.size() * sizeof(uchar), resizedLeftImgData.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error reading resized left image buffer: " << err << std::endl;
        return -1;
    }
    err = clEnqueueReadBuffer(queue, resizedRightImgBuffer, CL_TRUE, 0, resizedRightImgData.size() * sizeof(uchar), resizedRightImgData.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error reading resized right image buffer: " << err << std::endl;
        return -1;
    }

    // Release resize kernel and program
    clReleaseKernel(resizeKernel);
    clReleaseProgram(resizeProgram);

    // Create OpenCV matrices for the resized images
    cv::Mat resizedLeftImg(resized_height, resized_width, CV_8U, resizedLeftImgData.data());
    cv::Mat resizedRightImg(resized_height, resized_width, CV_8U, resizedRightImgData.data());

    // Load ZNCC kernel source code from file
    // std::string znccKernelSource = loadKernelSource("zncc_full_kernel.cl");
    // std::string znccKernelSource = loadKernelSource("zncc_kernel_optimized.cl");
    std::string znccKernelSource = loadKernelSource("zncc_kernel_optimized2.cl");

    const char *znccKernelSourceCStr = znccKernelSource.c_str();
    cl_program znccProgram = clCreateProgramWithSource(context, 1, &znccKernelSourceCStr, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error creating zncc program: " << err << std::endl;
        return -1;
    }

    err = clBuildProgram(znccProgram, 1, &device, nullptr, nullptr, nullptr);
    size_t znccLogSize;
    clGetProgramBuildInfo(znccProgram, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &znccLogSize);
    std::vector<char> znccLog(znccLogSize);
    clGetProgramBuildInfo(znccProgram, device, CL_PROGRAM_BUILD_LOG, znccLogSize, znccLog.data(), nullptr);
    std::cerr << "ZNCC Build log: " << znccLog.data() << std::endl;

    if (err != CL_SUCCESS)
    {
        std::cerr << "Error building zncc program: " << znccLog.data() << std::endl;
        return -1;
    }

    cl_kernel znccKernel = clCreateKernel(znccProgram, "zncc_kernel", &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error creating zncc kernel: " << err << std::endl;
        return -1;
    }

    // Create buffers for ZNCC kernel
    cl_mem znccLeftImgBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, resizedLeftImg.total() * resizedLeftImg.elemSize(), resizedLeftImg.data, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error creating zncc left image buffer: " << err << std::endl;
        return -1;
    }
    cl_mem znccRightImgBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, resizedRightImg.total() * resizedRightImg.elemSize(), resizedRightImg.data, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error creating zncc right image buffer: " << err << std::endl;
        return -1;
    }
    cl_mem disparityLeftBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, resizedLeftImg.total() * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error creating disparity left buffer: " << err << std::endl;
        return -1;
    }
    cl_mem disparityRightBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, resizedLeftImg.total() * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error creating disparity right buffer: " << err << std::endl;
        return -1;
    }
    cl_mem crossCheckedDisparityBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, resizedLeftImg.total() * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error creating cross-checked disparity buffer: " << err << std::endl;
        return -1;
    }
    cl_mem filledDisparityBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, resizedLeftImg.total() * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error creating filled disparity buffer: " << err << std::endl;
        return -1;
    }

    // Set ZNCC kernel arguments
    err = clSetKernelArg(znccKernel, 0, sizeof(cl_mem), &znccLeftImgBuffer);
    err |= clSetKernelArg(znccKernel, 1, sizeof(cl_mem), &znccRightImgBuffer);
    err |= clSetKernelArg(znccKernel, 2, sizeof(cl_mem), &disparityLeftBuffer);
    err |= clSetKernelArg(znccKernel, 3, sizeof(cl_mem), &disparityRightBuffer);
    err |= clSetKernelArg(znccKernel, 4, sizeof(cl_mem), &crossCheckedDisparityBuffer);
    err |= clSetKernelArg(znccKernel, 5, sizeof(cl_mem), &filledDisparityBuffer);
    err |= clSetKernelArg(znccKernel, 6, sizeof(int), &resized_width);
    err |= clSetKernelArg(znccKernel, 7, sizeof(int), &resized_height);
    err |= clSetKernelArg(znccKernel, 8, sizeof(int), &winSize);
    err |= clSetKernelArg(znccKernel, 9, sizeof(int), &maxDisp);
    err |= clSetKernelArg(znccKernel, 10, sizeof(int), &threshold);
    err |= clSetKernelArg(znccKernel, 11, sizeof(int), &max_search_distance);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error setting zncc kernel arguments: " << err << std::endl;
        return -1;
    }

    // Execute the ZNCC kernel
    size_t znccGlobalSize = resized_width * resized_height;
    cl_event znccKernelEvent;
    err = clEnqueueNDRangeKernel(queue, znccKernel, 1, nullptr, &znccGlobalSize, nullptr, 0, nullptr, &znccKernelEvent);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error enqueuing zncc kernel: " << err << std::endl;
        return -1;
    }

    // Wait for the ZNCC kernel to finish
    clWaitForEvents(1, &znccKernelEvent);

    // Profiling information for zncc kernel
    cl_ulong znccStart, znccEnd;
    clGetEventProfilingInfo(znccKernelEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &znccStart, nullptr);
    clGetEventProfilingInfo(znccKernelEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &znccEnd, nullptr);

    double znccTime = (znccEnd - znccStart) * 1.0e-6; // Convert to milliseconds

    std::cout << "ZNCC Kernel Time: " << znccTime << " ms" << std::endl;

    // Read the output buffers
    std::vector<float> disparityLeftData(resizedLeftImg.total());
    std::vector<float> disparityRightData(resizedLeftImg.total());
    std::vector<float> crossCheckedDisparityData(resizedLeftImg.total());
    std::vector<float> filledDisparityData(resizedLeftImg.total());

    err = clEnqueueReadBuffer(queue, disparityLeftBuffer, CL_TRUE, 0, disparityLeftData.size() * sizeof(float), disparityLeftData.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error reading disparity left buffer: " << err << std::endl;
        return -1;
    }
    err = clEnqueueReadBuffer(queue, disparityRightBuffer, CL_TRUE, 0, disparityRightData.size() * sizeof(float), disparityRightData.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error reading disparity right buffer: " << err << std::endl;
        return -1;
    }
    err = clEnqueueReadBuffer(queue, crossCheckedDisparityBuffer, CL_TRUE, 0, crossCheckedDisparityData.size() * sizeof(float), crossCheckedDisparityData.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error reading cross-checked disparity buffer: " << err << std::endl;
        return -1;
    }
    err = clEnqueueReadBuffer(queue, filledDisparityBuffer, CL_TRUE, 0, filledDisparityData.size() * sizeof(float), filledDisparityData.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error reading filled disparity buffer: " << err << std::endl;
        return -1;
    }

    // Convert float data to unsigned char for saving as images
    cv::Mat disparityLeftImg(resized_height, resized_width, CV_8U);
    cv::Mat disparityRightImg(resized_height, resized_width, CV_8U);
    cv::Mat crossCheckedDisparityImg(resized_height, resized_width, CV_8U);
    cv::Mat filledDisparityImg(resized_height, resized_width, CV_8U);

    for (int i = 0; i < resized_height * resized_width; ++i)
    {
        disparityLeftImg.data[i] = static_cast<unsigned char>(disparityLeftData[i]);
        disparityRightImg.data[i] = static_cast<unsigned char>(disparityRightData[i]);
        crossCheckedDisparityImg.data[i] = static_cast<unsigned char>(crossCheckedDisparityData[i]);
        filledDisparityImg.data[i] = static_cast<unsigned char>(filledDisparityData[i]);
    }

    double minVal = 0.0;   // Minimum expected disparity value
    double maxVal = 255.0; // Maximum expected disparity value

    // Normalize and save disparity_left_img.png
    cv::Mat normalized_disparityLeftImg;
    cv::normalize(disparityLeftImg, normalized_disparityLeftImg, minVal, maxVal, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite("disparity_left_img.png", normalized_disparityLeftImg);

    // Normalize and save disparity_right_img.png
    cv::Mat normalized_disparityRightImg;
    cv::normalize(disparityRightImg, normalized_disparityRightImg, minVal, maxVal, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite("disparity_right_img.png", normalized_disparityRightImg);

    // Normalize and save cross_checked_disparity_img.png (assuming it has the same data type as others)
    cv::Mat normalized_crossCheckedDisparityImg;
    cv::normalize(crossCheckedDisparityImg, normalized_crossCheckedDisparityImg, minVal, maxVal, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite("cross_checked_disparity_img.png", normalized_crossCheckedDisparityImg);

    // Normalize and save filled_disparity_img.png (assuming it has the same data type as others)
    cv::Mat normalized_filledDisparityImg;
    cv::normalize(filledDisparityImg, normalized_filledDisparityImg, minVal, maxVal, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite("filled_disparity_img.png", normalized_filledDisparityImg);

    // Cleanup
    clReleaseEvent(resizeKernelEventLeft);
    clReleaseEvent(resizeKernelEventRight);
    clReleaseEvent(znccKernelEvent);
    clReleaseMemObject(leftImgBuffer);
    clReleaseMemObject(rightImgBuffer);
    clReleaseMemObject(resizedLeftImgBuffer);
    clReleaseMemObject(resizedRightImgBuffer);
    clReleaseMemObject(znccLeftImgBuffer);
    clReleaseMemObject(znccRightImgBuffer);
    clReleaseMemObject(disparityLeftBuffer);
    clReleaseMemObject(disparityRightBuffer);
    clReleaseMemObject(crossCheckedDisparityBuffer);
    clReleaseMemObject(filledDisparityBuffer);
    clReleaseKernel(znccKernel);
    clReleaseProgram(znccProgram);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
