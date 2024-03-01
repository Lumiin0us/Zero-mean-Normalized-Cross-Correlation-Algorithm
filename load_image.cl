__kernel void rgb_to_gray(__global uchar4* rgbImage, 
                          __global uchar* grayImage, 
                          const int width, 
                          const int height) {
    int x = get_global_id(0); // Get the global ID in the x-direction
    int y = get_global_id(1); // Get the global ID in the y-direction

    if (x < width && y < height) {
        int index = y * width + x; // Compute 1D index for accessing pixel
        uchar4 pixel = rgbImage[index]; 

        // Converting RGB to grayscale using luminosity method
        uchar grayValue = (uchar)(0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z);

        grayImage[index] = grayValue;
    }
}