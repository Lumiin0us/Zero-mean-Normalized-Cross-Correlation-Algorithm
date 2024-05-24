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
    // Load input RGB image
    cv::Mat image = cv::imread("a.png", cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

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
    std::string kernelSource = loadKernelSource("rgb_to_grayscale.cl");
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

    cl_kernel kernel = clCreateKernel(program, "rgb_to_grayscale", &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error creating kernel: " << err << std::endl;
        return -1;
    }

    // Create buffers
    cl_mem inputImgBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, image.total() * image.elemSize(), image.data, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error creating input image buffer: " << err << std::endl;
        return -1;
    }
    cl_mem outputImgBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * sizeof(uchar), nullptr, &err);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error creating output image buffer: " << err << std::endl;
        return -1;
    }

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImgBuffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputImgBuffer);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &width);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &height);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error setting kernel arguments: " << err << std::endl;
        return -1;
    }

    // Execute the kernel
    size_t globalSize[] = {static_cast<size_t>(width), static_cast<size_t>(height)};
    cl_event kernel_event;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, &kernel_event);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error enqueuing kernel: " << err << std::endl;
        return -1;
    }

    // Wait for the kernel to finish
    clWaitForEvents(1, &kernel_event);

    // Read the output buffer
    std::vector<uchar> outputImageData(width * height);
    err = clEnqueueReadBuffer(queue, outputImgBuffer, CL_TRUE, 0, outputImageData.size() * sizeof(uchar), outputImageData.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error reading output image buffer: " << err << std::endl;
        return -1;
    }

    // Convert output data to OpenCV Mat and save image
    cv::Mat grayImage(height, width, CV_8U, outputImageData.data());
    cv::imwrite("gray_image.png", grayImage);

    // Cleanup
    clReleaseEvent(kernel_event);
    clReleaseMemObject(inputImgBuffer);
    clReleaseMemObject(outputImgBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
