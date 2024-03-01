#include <OpenCL/opencl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h> // For chdir() 

int main() {
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

    // Changing workdir
    if (chdir("/Users/abdurrehman/Desktop/MicroprosessorProgramming") != 0) {
        std::cout << "Error changing directory" << std::endl;
        return 1;
    }

    // Printing current working directory - to check if prev function was succesful
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        std::cout << "Current working directory: " << cwd << std::endl;
    } else {
        std::cout << "Error getting current working directory" << std::endl;
        return 1;
    }

    // full path to the OpenCL program source-file
    std::string filePath = "/Users/abdurrehman/Desktop/MicroprosessorProgramming/hello_world.cl";
    std::cout << "Attempted file path: " << filePath << std::endl;

    // Loading the source file
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cout << "Failed to open file!" << std::endl;
        std::perror("Error");
        return 1;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string source = buffer.str();
    const char *sources[] = {source.c_str()};
    program = clCreateProgramWithSource(context, 1, sources, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error creating program" << std::endl;
        return 1;
    }

    // Building the program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cout << "Error building program" << std::endl;
        return 1;
    }

    // Creating the kernel
    kernel = clCreateKernel(program, "hello_world", &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error creating kernel" << std::endl;
        return 1;
    }

    // Enqueuing the kernel/function for execution
    size_t global_size = 1; 
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cout << "Error enqueueing kernel" << std::endl;
        return 1;
    }

    // Releasing OpenCL resources
    clFinish(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
