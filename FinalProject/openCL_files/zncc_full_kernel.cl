// Function to calculate the mean of a patch in the image
float calculateMean(int x, int y, int d, int width, int height, int winSize, __global const unsigned char* img, bool flag)
{
    int sum = 0;
    int count = 0;

    // Iterate over the window centered around (x, y)
    for (int j = -winSize / 2; j <= winSize / 2; ++j)
        {
            for (int i = -winSize / 2; i <= winSize / 2; ++i)
            {
                int xj = x + i;
                int yj = y + j;

                // Check if the current coordinates are within the image bounds
                if (xj >= 0 && xj < width && yj >= 0 && yj < height)
                {
                    int xjd = flag ? xj + d : xj - d;

                    // Check if the adjusted x coordinate is within the image bounds
                    if (xjd >= 0 && xjd < width)
                    {
                        sum += img[yj * width + xjd];
                        count++;
                    }
                }
            }
        }

    // Return the mean value of the patch, or 0 if no valid pixels were found
    return count == 0 ? 0.0f : (float)sum / (float)count;
}

// Function to calculate the ZNCC values
float calculateZncc(int x, int y, int d, float mean1, float mean2, int width, int height, int winSize, __global const unsigned char* img1, __global const unsigned char* img2, bool flag)
{
    float num = 0.0f;
    float denom1 = 0.0f;
    float denom2 = 0.0f;

    for (int j = -winSize / 2; j <= winSize / 2; ++j)
    {
        for (int i = -winSize / 2; i <= winSize / 2; ++i)
        {
            int xj1 = x + i;
            int yj = y + j;
            int xj2 = flag ? xj1 + d : xj1 - d;

            if (xj1 >= 0 && xj1 < width && xj2 >= 0 && xj2 < width && yj >= 0 && yj < height)
            {
                float val1, val2;
                if (flag)
                {
                    val1 = img2[yj * width + xj2] - mean2;
                    val2 = img1[yj * width + xj1] - mean1;
                }
                else
                {
                    val1 = img1[yj * width + xj1] - mean1;
                    val2 = img2[yj * width + xj2] - mean2;
                }
                num += val1 * val2;
                denom1 += val1 * val1;
                denom2 += val2 * val2;
            }
        }
    }

    float denom = sqrt(denom1 * denom2);
    return denom == 0.0f ? 0.0f : num / denom;
}

// Function to calculate the disparity map with the left image as reference
float calculate_disparity_left(
  __global const unsigned char* leftImage,
  __global const unsigned char* rightImage,
  int x, int y, int width, int height, int winSize, int maxDisp
) {

  bool right = false;
  float maxZncc = -1.0f;
  int bestDisp = 0;

  float mean1 = calculateMean(x, y, 0, width, height, winSize, leftImage, right);

  for (int d = 0; d < maxDisp; ++d) {
    float mean2 = calculateMean(x, y, d, width, height, winSize, rightImage, right);
    float znccVal = calculateZncc(x, y, d, mean1, mean2, width, height, winSize, leftImage, rightImage, right);

    if (znccVal > maxZncc) {
      maxZncc = znccVal;
      bestDisp = d;
    }
  }

  return (float)bestDisp;
}

// Function to calculate the disparity map with the right image as reference
float calculate_disparity_right(
  __global const unsigned char* leftImage,
  __global const unsigned char* rightImage,
  int x, int y, int width, int height, int winSize, int maxDisp
) {

  bool right = true; 
  float maxZncc = -1.0f;
  int bestDisp = 0;

  float mean1 = calculateMean(x, y, 0, width, height, winSize, rightImage, right);

  for (int d = 0; d < maxDisp; ++d) {
    float mean2 = calculateMean(x, y, d, width, height, winSize, leftImage, right);
    float znccVal = calculateZncc(x, y, d, mean1, mean2, width, height, winSize, leftImage, rightImage, right);

    if (znccVal > maxZncc) {
      maxZncc = znccVal;
      bestDisp = d;
    }
  }

  return (float)bestDisp;
}

// Function for occlusion filling
void fill_occlusions(
  int x, int y, int width, int height, int max_search_distance,
  __global float* disparity_map, __global float* filled_disparity
) {
  filled_disparity[y * width + x] = disparity_map[y * width + x];

  if (disparity_map[y * width + x] == 0.0f) { // Check for invalid disparity (occlusion)
    for (int d = 1; d <= max_search_distance; ++d) {
      // Check neighbors in 4 directions
      for (int dir = 0; dir < 4; ++dir) {
        int dx = (dir % 2 == 0) ? 0 : d * ((dir / 2) * 2 - 1);
        int dy = (dir % 2 == 1) ? d * ((dir - 1) / 2) : 0;
        int nx = x + dx;
        int ny = y + dy; // Neighbor coordinates

        if (0 <= nx && nx < width && 0 <= ny && ny < height && disparity_map[ny * width + nx] != 0.0f) {
          filled_disparity[y * width + x] = disparity_map[ny * width + nx]; // Fill with neighbor value
          return; // Stop searching once a valid neighbor is found
        }
      }
    }
  }
}

__kernel void zncc_kernel(
  __global const unsigned char* leftImg,
  __global const unsigned char* rightImg,
  __global float* disparityLeft, // Output for left reference
  __global float* disparityRight, // Output for right reference
  __global float* crossCheckedDisparity, // Output for cross-checked disparity
  __global float* filledDisparity, // Output for filled disparity
  int width, int height, int winSize, int maxDisp, int threshold, int max_search_distance
) {
  // Get global thread ID
  int idx = get_global_id(0);
  int x = idx % width;
  int y = idx / width;

  // Calculate disparity with right image as reference
  float dispRight = calculate_disparity_right(rightImg, leftImg, x, y, width, height, winSize, maxDisp); // Fix: pass true for right disparity
  disparityRight[y * width + x] = dispRight;

  // Calculate disparity with left image as reference
  float dispLeft = calculate_disparity_left(leftImg, rightImg, x, y, width, height, winSize, maxDisp); // Fix: pass false for left disparity
  disparityLeft[y * width + x] = dispLeft;


  // Barrier to ensure all threads have computed disparities
  barrier(CLK_GLOBAL_MEM_FENCE);

  // Perform cross-checking
  float left_disp = disparityLeft[y * width + x];
  float right_disp = disparityRight[y * width + x];

  if (fabs(left_disp - right_disp) > threshold) {
    crossCheckedDisparity[y * width + x] = 0.0f;
  } else {
    crossCheckedDisparity[y * width + x] = left_disp;
  }

  // Barrier to ensure all threads have completed cross-checking
  barrier(CLK_GLOBAL_MEM_FENCE);

  // Fill occlusions
  fill_occlusions(x, y, width, height, max_search_distance, crossCheckedDisparity, filledDisparity);
}
