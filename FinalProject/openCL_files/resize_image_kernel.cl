__kernel void resize_image_kernel(
    __global const uchar* input_image,
    __global uchar* output_image,
    int input_width,
    int input_height,
    int output_width,
    int output_height,
    int factor
) {
    // Thread indices in the global work-group
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= output_width || y >= output_height) return;

    // Calculate corresponding input pixel coordinates based on scaling factor
    int input_x = x * factor;
    int input_y = y * factor;

    if (input_x >= input_width || input_y >= input_height) return;

    // Calculate memory indices for input and output images
    int input_index = input_y * input_width + input_x;
    int output_index = y * output_width + x;

    output_image[output_index] = input_image[input_index];
}
