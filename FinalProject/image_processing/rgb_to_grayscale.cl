__kernel void rgb_to_grayscale(
    __global const uchar* input_image,
    __global uchar* output_image,
    int width,
    int height
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    int input_index = (y * width + x) * 3;
    int output_index = y * width + x;

    uchar r = input_image[input_index];
    uchar g = input_image[input_index + 1];
    uchar b = input_image[input_index + 2];

    uchar gray = (uchar)(0.299f * r + 0.587f * g + 0.114f * b);

    output_image[output_index] = gray;
}
