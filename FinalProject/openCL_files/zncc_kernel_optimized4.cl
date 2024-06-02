__constant int winSize = 21; // Example value, adjust as needed
__constant int halfWinSize = winSize / 2;

float calculateMean(int x, int y, int d, int width, int height, __global const unsigned char* img)
{
    int yy_0 = max(0, y - halfWinSize);
    int yy_1 = min(height, y + halfWinSize + 1);
    int xx_0 = max(0, x - halfWinSize);
    int xx_1 = min(width, x + halfWinSize + 1);

    float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    int count = 0;

    for (int yy = yy_0; yy < yy_1; yy++)
    {
        for (int xx = xx_0; xx < xx_1; xx += 4) // Process 4 pixels at a time
        {
            int xxd = xx - d;
            if (xxd >= 0 && (xxd + 3) < width)
            {
                // Load 4 pixels into a vector
                uchar4 pixelVec = vload4(0, &img[yy * width + xxd]);
                sum += convert_float4(pixelVec);
                count += 4;
            }
        }
    }

    float totalSum = sum.s0 + sum.s1 + sum.s2 + sum.s3;
    return count == 0 ? 0.0f : totalSum / count;
}

float calculateZncc(int x, int y, int d, float mean1, float mean2, int width, int height, __global const unsigned char* img1, __global const unsigned char* img2)
{
    int yy_0 = max(0, y - halfWinSize);
    int yy_1 = min(height, y + halfWinSize + 1);
    int xx_0 = max(0, x - halfWinSize);
    int xx_1 = min(width, x + halfWinSize + 1);

    float4 num = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float4 denom1 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float4 denom2 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    for (int yy = yy_0; yy < yy_1; yy++)
    {
        for (int xx = xx_0; xx < xx_1; xx += 4) // Process 4 pixels at a time
        {
            int xxd = xx - d;
            if (xxd >= 0 && (xxd + 3) < width)
            {
                uchar4 val1Vec = vload4(0, &img1[yy * width + xx]);
                uchar4 val2Vec = vload4(0, &img2[yy * width + xxd]);

                float4 val1 = convert_float4(val1Vec) - (float4)(mean1);
                float4 val2 = convert_float4(val2Vec) - (float4)(mean2);

                num += val1 * val2;
                denom1 += val1 * val1;
                denom2 += val2 * val2;
            }
        }
    }

    float totalNum = num.s0 + num.s1 + num.s2 + num.s3;
    float totalDenom1 = denom1.s0 + denom1.s1 + denom1.s2 + denom1.s3;
    float totalDenom2 = denom2.s0 + denom2.s1 + denom2.s2 + denom2.s3;

    float denom = sqrt(totalDenom1 * totalDenom2);
    return denom == 0.0f ? 0.0f : totalNum / denom;
}


float calculate_disparity_left(
    __global const unsigned char* leftImage,
    __global const unsigned char* rightImage,
    int x, int y, int width, int height, int maxDisp)
{
    float maxZncc = -1.0;
    int bestDisp = 0;

    float mean1 = calculateMean(x, y, 0, width, height, leftImage);

    for (int d = 0; d < maxDisp; d++)
    {
        float mean2 = calculateMean(x, y, d, width, height, rightImage);
        float znccVal = calculateZncc(x, y, d, mean1, mean2, width, height, leftImage, rightImage);

        if (znccVal > maxZncc)
        {
            maxZncc = znccVal;
            bestDisp = d;
        }
    }

    return (float)bestDisp;
}

float calculate_disparity_right(
    __global const unsigned char* leftImage,
    __global const unsigned char* rightImage,
    int x, int y, int width, int height, int maxDisp)
{
    float maxZncc = -1.0;
    int bestDisp = 0;

    float mean1 = calculateMean(x, y, 0, width, height, rightImage);

    for (int d = 0; d < maxDisp; d++)
    {
        float mean2 = calculateMean(x, y, d, width, height, leftImage);
        float znccVal = calculateZncc(x, y, d, mean1, mean2, width, height, rightImage, leftImage);

        if (znccVal > maxZncc)
        {
            maxZncc = znccVal;
            bestDisp = d;
        }
    }

    return (float)bestDisp;
}


void fill_occlusions(
    int x, int y, int width, int height, int max_search_distance,
    __global float* disparity_map, __global float* filled_disparity)
{
    filled_disparity[y * width + x] = disparity_map[y * width + x];

    if (disparity_map[y * width + x] == 0.0f) // Check for invalid disparity (occlusion)
    {
        for (int d = 1; d <= max_search_distance; ++d)
        {
            // Check neighbors in 4 directions
            for (int dir = 0; dir < 4; ++dir)
            {
                int dx = (dir % 2 == 0) ? 0 : d * ((dir / 2) * 2 - 1);
                int dy = (dir % 2 == 1) ? d * ((dir - 1) / 2) : 0;
                int nx = x + dx;
                int ny = y + dy; // Neighbor coordinates

                if (0 <= nx && nx < width && 0 <= ny && ny < height && disparity_map[ny * width + nx] != 0.0f)
                {
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
    int width, int height, int maxDisp, int threshold, int max_search_distance)
{
    // Get global thread ID
    int idx = get_global_id(0);
    int x = idx % width;
    int y = idx / width;

    // Calculate disparity with left image as reference
    disparityLeft[y * width + x] = calculate_disparity_left(leftImg, rightImg, x, y, width, height, maxDisp);

    // Calculate disparity with right image as reference
    disparityRight[y * width + x] = calculate_disparity_right(leftImg, rightImg, x, y, width, height, maxDisp);

    // Barrier to ensure all threads have computed disparities
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Perform cross-checking
    float left_disp = disparityLeft[y * width + x];
    float right_disp = disparityRight[y * width + x];

    if (fabs(left_disp - right_disp) > threshold)
    {
        crossCheckedDisparity[y * width + x] = 0.0f;
    }
    else
    {
        crossCheckedDisparity[y * width + x] = left_disp;
    }

    // Barrier to ensure all threads have completed cross-checking
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Fill occlusions
    fill_occlusions(x, y, width, height, max_search_distance, crossCheckedDisparity, filledDisparity);
}

