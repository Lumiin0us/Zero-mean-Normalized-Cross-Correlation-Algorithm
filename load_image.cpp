#include <OpenCL/opencl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h> // For chdir() 
#include "/Users/abdurrehman/Downloads/stb_image.h" // stb_image headerfile just downloaded from github

int main() {
    const char* filename = "image.jpg";
    int width, height, channels;
    unsigned char* image_data = stbi_load(filename, &width, &height, &channels, 0);

    if (!image_data) {
        std::cerr << "Failed to load image: " << stbi_failure_reason() << std::endl;
        return 1;
    } 

    std::cout << "Image loaded successfully. Dimensions: " << width << "x" << height << ", Channels: " << channels << std::endl;

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    // Creating OpenCL context
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error creating context" << std::endl;
        return 1;
    }

    // Creating a command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error creating command queue" << std::endl;
        return 1;
    }

    // Create memory objects
    cl_mem imageBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width * height * channels * sizeof(unsigned char), image_data, &err);
    cl_mem resultBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * sizeof(unsigned char), NULL, &err);

    // Load kernel source code
    std::ifstream file("/Users/abdurrehman/Desktop/MicroprosessorProgramming/load_image.cl");
    if (!file.is_open()) {
        std::cerr << "Failed to open file!" << std::endl;
        return 1;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string source = buffer.str();
    const char* source_str = source.c_str();

    // Create program
    program = clCreateProgramWithSource(context, 1, (const char**)&source_str, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error creating program" << std::endl;
        return 1;
    }

    // Build program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cout << "Error building program" << std::endl;
        return 1;
    }

    // Create kernel
    kernel = clCreateKernel(program, "convert_to_grayscale", &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error creating kernel" << std::endl;
        return 1;
    }

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageBuffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &resultBuffer);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &width);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &height);
    if (err != CL_SUCCESS) {
        std::cout << "Error setting kernel arguments" << std::endl;
        return 1;
    }

    // Execute kernel
    size_t globalSize[2] = {width, height};
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cout << "Error enqueueing kernel" << std::endl;
        return 1;
    }

    // Read result from device
    unsigned char* resultData = new unsigned char[width * height];
    err = clEnqueueReadBuffer(queue, resultBuffer, CL_TRUE, 0, width * height * sizeof(unsigned char), resultData, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cout << "Error reading result buffer" << std::endl;
        return 1;
    }

    // Free resources
    clReleaseMemObject(imageBuffer);
    clReleaseMemObject(resultBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // Free image data
    stbi_image_free(image_data);

    return 0;
}
