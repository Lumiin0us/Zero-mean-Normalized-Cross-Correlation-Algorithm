#include <OpenCL/opencl.h>
#include <iostream>
#include <vector>

int main()
{
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_int err;

    // Creating OpenCL context
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS)
    {
        std::cout << "Error creating context" << std::endl;
        return 1;
    }

    // Creating a command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS)
    {
        std::cout << "Error creating command queue" << std::endl;
        return 1;
    }

    // Querying and printing device information
    cl_device_local_mem_type localMemType;
    cl_ulong localMemSize;
    cl_uint maxComputeUnits;
    cl_uint maxClockFrequency;
    cl_ulong maxConstantBufferSize;
    size_t maxWorkGroupSize;
    size_t maxWorkItemSizes[3];

    // CL_DEVICE_LOCAL_MEM_TYPE
    err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(localMemType), &localMemType, NULL);
    if (err == CL_SUCCESS)
        std::cout << "CL_DEVICE_LOCAL_MEM_TYPE: " << (localMemType == CL_LOCAL ? "CL_LOCAL" : "CL_GLOBAL") << std::endl;

    // CL_DEVICE_LOCAL_MEM_SIZE
    err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), &localMemSize, NULL);
    if (err == CL_SUCCESS)
        std::cout << "CL_DEVICE_LOCAL_MEM_SIZE: " << localMemSize << " bytes" << std::endl;

    // CL_DEVICE_MAX_COMPUTE_UNITS
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    if (err == CL_SUCCESS)
        std::cout << "CL_DEVICE_MAX_COMPUTE_UNITS: " << maxComputeUnits << std::endl;

    // CL_DEVICE_MAX_CLOCK_FREQUENCY
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(maxClockFrequency), &maxClockFrequency, NULL);
    if (err == CL_SUCCESS)
        std::cout << "CL_DEVICE_MAX_CLOCK_FREQUENCY: " << maxClockFrequency << " MHz" << std::endl;

    // CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(maxConstantBufferSize), &maxConstantBufferSize, NULL);
    if (err == CL_SUCCESS)
        std::cout << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: " << maxConstantBufferSize << " bytes" << std::endl;

    // CL_DEVICE_MAX_WORK_GROUP_SIZE
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
    if (err == CL_SUCCESS)
        std::cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << maxWorkGroupSize << std::endl;

    // CL_DEVICE_MAX_WORK_ITEM_SIZES
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxWorkItemSizes), maxWorkItemSizes, NULL);
    if (err == CL_SUCCESS)
        std::cout << "CL_DEVICE_MAX_WORK_ITEM_SIZES: " << maxWorkItemSizes[0] << " " << maxWorkItemSizes[1] << " " << maxWorkItemSizes[2] << std::endl;

    // Releasing OpenCL resources
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
