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
    // Load input grayscale images
    cv::Mat image1 = cv::imread("../images/im0.png", cv::IMREAD_GRAYSCALE);
    cv::Mat image2 = cv::imread("../images/im1.png", cv::IMREAD_GRAYSCALE);

    if (image1.empty() || image2.empty())
    {
        std::cerr << "Error loading images" << std::endl;
        return -1;
    }

    int factor = 4; // Example downscale factor
    int input_width1 = image1.cols;
    int input_height1 = image1.rows;
    int output_width1 = input_width1 / factor;
    int output_height1 = input_height1 / factor;

    int input_width2 = image2.cols;
    int input_height2 = image2.rows;
    int output_width2 = input_width2 / factor;
    int output_height2 = input_height2 / factor;

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

    // Load kernel source code from file
    std::string kernelSource = loadKernelSource("resize_image_kernel.cl");
    const char *kernelSourceCStr = kernelSource.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSourceCStr, nullptr, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error creating program: " << err << std::endl;
        return -1;
    }

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    size_t logSize;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
    std::vector<char> log(logSize);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
    std::cerr << "Build log: " << log.data() << std::endl;

    if (err != CL_SUCCESS)
    {
        std::cerr << "Error building program: " << log.data() << std::endl;
        return -1;
    }

    cl_kernel kernel = clCreateKernel(program, "resize_image_kernel", &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error creating kernel: " << err << std::endl;
        return -1;
    }

    // Create buffers for image1
    cl_mem inputImgBuffer1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, image1.total() * image1.elemSize(), image1.data, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error creating input image buffer 1: " << err << std::endl;
        return -1;
    }
    cl_mem outputImgBuffer1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_width1 * output_height1 * sizeof(uchar), nullptr, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error creating output image buffer 1: " << err << std::endl;
        return -1;
    }

    // Create buffers for image2
    cl_mem inputImgBuffer2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, image2.total() * image2.elemSize(), image2.data, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error creating input image buffer 2: " << err << std::endl;
        return -1;
    }
    cl_mem outputImgBuffer2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_width2 * output_height2 * sizeof(uchar), nullptr, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error creating output image buffer 2: " << err << std::endl;
        return -1;
    }

    // Set kernel arguments for image1
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImgBuffer1);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputImgBuffer1);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &input_width1);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &input_height1);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &output_width1);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &output_height1);
    err |= clSetKernelArg(kernel, 6, sizeof(int), &factor);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error setting kernel arguments for image 1: " << err << std::endl;
        return -1;
    }

    // Execute the kernel for image1
    size_t globalSize1[] = {static_cast<size_t>(output_width1), static_cast<size_t>(output_height1)};
    cl_event kernel_event1;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize1, nullptr, 0, nullptr, &kernel_event1);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error enqueuing kernel for image 1: " << err << std::endl;
        return -1;
    }

    // Set kernel arguments for image2
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImgBuffer2);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputImgBuffer2);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &input_width2);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &input_height2);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &output_width2);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &output_height2);
    err |= clSetKernelArg(kernel, 6, sizeof(int), &factor);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error setting kernel arguments for image 2: " << err << std::endl;
        return -1;
    }

    // Execute the kernel for image2
    size_t globalSize2[] = {static_cast<size_t>(output_width2), static_cast<size_t>(output_height2)};
    cl_event kernel_event2;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize2, nullptr, 0, nullptr, &kernel_event2);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error enqueuing kernel for image 2: " << err << std::endl;
        return -1;
    }

    // Wait for both kernels to finish
    clWaitForEvents(1, &kernel_event1);
    clWaitForEvents(1, &kernel_event2);

    // Read the output buffers
    std::vector<uchar> outputImageData1(output_width1 * output_height1);
    std::vector<uchar> outputImageData2(output_width2 * output_height2);

    err = clEnqueueReadBuffer(queue, outputImgBuffer1, CL_TRUE, 0, outputImageData1.size() * sizeof(uchar), outputImageData1.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error reading output image buffer 1: " << err << std::endl;
        return -1;
    }
    err = clEnqueueReadBuffer(queue, outputImgBuffer2, CL_TRUE, 0, outputImageData2.size() * sizeof(uchar), outputImageData2.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error reading output image buffer 2: " << err << std::endl;
        return -1;
    }

    // Convert output data to OpenCV Mat and save images
    cv::Mat resizedImage1(output_height1, output_width1, CV_8U, outputImageData1.data());
    cv::Mat resizedImage2(output_height2, output_width2, CV_8U, outputImageData2.data());

    cv::imwrite("resized_image1.png", resizedImage1);
    cv::imwrite("resized_image2.png", resizedImage2);

    // Cleanup
    clReleaseEvent(kernel_event1);
    clReleaseEvent(kernel_event2);
    clReleaseMemObject(inputImgBuffer1);
    clReleaseMemObject(outputImgBuffer1);
    clReleaseMemObject(inputImgBuffer2);
    clReleaseMemObject(outputImgBuffer2);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
