#include <cmath>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <limits>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include "./omp/omp.h"

using namespace std;
using namespace std::chrono;

// Function to calculate the ZNCC between two patches
float calculate_zncc(int x, int y, int d, float mean1, float mean2, int width, int height, int winSize, const std::vector<unsigned char> &img1, const std::vector<unsigned char> &img2, bool flag)
{
    float num = 0.0f;
    float denom1 = 0.0f;
    float denom2 = 0.0f;

// Parallelize the loop and accumulate num, denom1, denom2 across threads
#pragma omp parallel for reduction(+ : num, denom1, denom2)
    // Iterate over the patch window
    for (int j = -winSize / 2; j <= winSize / 2; ++j)
    {
        for (int i = -winSize / 2; i <= winSize / 2; ++i)
        {
            int xj1 = x + i;
            int yj = y + j;
            int xj2 = flag ? xj1 + d : xj1 - d;

            // Check for valid coordinates
            if (xj1 >= 0 && xj1 < width && xj2 >= 0 && xj2 < width && yj >= 0 && yj < height)
            {
                float val1, val2;
                if (flag)
                {
                    val1 = img1[yj * width + xj2] - mean2;
                    val2 = img2[yj * width + xj1] - mean1;
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

    float denom = std::sqrt(denom1 * denom2);
    return denom == 0.0f ? 0.0f : num / denom;
}

// Function to calculate the mean of a patch in the image
float calculate_mean(int x, int y, int d, int width, int height, int winSize, const std::vector<unsigned char> &img, bool flag)
{
    int sum = 0;
    int count = 0;

// Parallelize the loop and accumulate sum and count across threads
#pragma omp parallel for reduction(+ : sum, count)
    // Iterate over the patch window
    for (int j = -winSize / 2; j <= winSize / 2; ++j)
    {
        for (int i = -winSize / 2; i <= winSize / 2; ++i)
        {
            int xj = x + i;
            int yj = y + j;

            // Check for valid coordinates
            if (xj >= 0 && xj < width && yj >= 0 && yj < height)
            {
                int xjd = flag ? xj + d : xj - d;

                if (xjd >= 0 && xjd < width)
                {
                    sum += img[yj * width + xjd];
                    count++;
                }
            }
        }
    }

    return count == 0 ? 0.0f : static_cast<float>(sum) / static_cast<float>(count);
}

// Function to compute the disparity map using ZNCC (Left Image as reference)
void calculate_disparity_map_left(const std::vector<unsigned char> &leftImg, const std::vector<unsigned char> &rightImg, std::vector<unsigned char> &disparityImg, int width, int height, int winSize, int maxDisp)
{
    auto left_disparity_start = high_resolution_clock::now();

    bool right = false;

// Parallelize the following loop using OpenMP
#pragma omp parallel for
    // Iterate over each pixel in the image
    for (int y = winSize / 2; y < height - winSize / 2; ++y)
    {
        for (int x = winSize / 2; x < width - winSize / 2; ++x)
        {
            float maxZncc = -1.0f;
            int bestDisp = 0;

            // Calculate mean for the left image patch
            float mean1 = calculate_mean(x, y, 0, width, height, winSize, leftImg, right);

            // Iterate over all possible disparities
            for (int d = 0; d < maxDisp; ++d)
            {
                // Calculate mean for the right image patch
                float mean2 = calculate_mean(x, y, d, width, height, winSize, rightImg, right);

                // Calculate ZNCC value
                float znccVal = calculate_zncc(x, y, d, mean1, mean2, width, height, winSize, leftImg, rightImg, right);

                // Find the best disparity
                if (znccVal > maxZncc)
                {
                    maxZncc = znccVal;
                    bestDisp = d;
                }
            }
            // Set the disparity value for the current pixel
            disparityImg[y * width + x] = static_cast<unsigned char>(bestDisp);
        }
    }

    auto left_disparity_end = high_resolution_clock::now();
    auto left_disparity_duration = duration_cast<milliseconds>(left_disparity_end - left_disparity_start);
    std::cout << "Time taken for calculate_disparity_map_left: " << left_disparity_duration.count() << " milliseconds" << std::endl;
}

// Function to compute the disparity map using ZNCC (Right Image as reference)
void calculate_disparity_map_right(const std::vector<unsigned char> &leftImg, const std::vector<unsigned char> &rightImg, std::vector<unsigned char> &disparityImg, int width, int height, int winSize, int maxDisp)
{
    auto right_disparity_start = high_resolution_clock::now();

    bool right = true;

// Parallelize the following loop using OpenMP
#pragma omp parallel for
    // Iterate over each pixel in the image
    for (int y = winSize / 2; y < height - winSize / 2; ++y)
    {
        for (int x = winSize / 2; x < width - winSize / 2; ++x)
        {
            float maxZncc = -1.0f;
            int bestDisp = 0;

            // Calculate mean for the left image patch
            float mean1 = calculate_mean(x, y, 0, width, height, winSize, rightImg, right);

            // Iterate over all possible disparities
            for (int d = 0; d < maxDisp; ++d)
            {
                // Calculate mean for the right image patch
                float mean2 = calculate_mean(x, y, d, width, height, winSize, leftImg, right);

                // Calculate ZNCC value
                float znccVal = calculate_zncc(x, y, d, mean1, mean2, width, height, winSize, leftImg, rightImg, right);

                // Find the best disparity
                if (znccVal > maxZncc)
                {
                    maxZncc = znccVal;
                    bestDisp = d;
                }
            }

            // Set the disparity value for the current pixel
            disparityImg[y * width + x] = static_cast<unsigned char>(bestDisp);
        }
    }

    auto right_disparity_end = high_resolution_clock::now();
    auto right_disparity_duration = duration_cast<milliseconds>(right_disparity_end - right_disparity_start);
    std::cout << "Time taken for calculate_disparity_map_left: " << right_disparity_duration.count() << " milliseconds" << std::endl;
}

// Function to fill occlusions in the disparity map
std::vector<unsigned char> fill_occlusions(const std::vector<unsigned char> &disparity_map, int width, int height, int max_search_distance)
{
    std::vector<unsigned char> filled_disparity(disparity_map); // Make a copy

// Parallelize the following loop using OpenMP
#pragma omp parallel for
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            if (disparity_map[y * width + x] == 0)
            { // Check for invalid disparity (occlusion)
                for (int d = 1; d <= max_search_distance; ++d)
                {
                    // Check neighbors in 4 directions
                    for (int dir = 0; dir < 4; ++dir)
                    {
                        int dx = (dir % 2 == 0) ? 0 : d * ((dir / 2) * 2 - 1);
                        int dy = (dir % 2 == 1) ? d * ((dir - 1) / 2) : 0;
                        int nx = x + dx;
                        int ny = y + dy; // Neighbor coordinates

                        if (0 <= nx && nx < width && 0 <= ny && ny < height && disparity_map[ny * width + nx] != 0)
                        {
                            filled_disparity[y * width + x] = disparity_map[ny * width + nx]; // Fill with neighbor value
                            break;                                                            // Stop searching once a valid neighbor is found
                        }
                    }
                    if (filled_disparity[y * width + x] != 0)
                    {
                        break; // Exit the outer loop if a valid neighbor was found for this pixel
                    }
                }
            }
        }
    }

    return filled_disparity;
}

// Function to cross-check disparity maps
cv::Mat cross_check_disparity_maps(const cv::Mat &left_disparity, const cv::Mat &right_disparity, int threshold)
{
    int height = left_disparity.rows;
    int width = left_disparity.cols;

    cv::Mat result = left_disparity.clone();

// Parallelize the following loop using OpenMP
#pragma omp parallel for
    // Iterate over each pixel in the image
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            float left_disp = left_disparity.at<unsigned char>(y, x);
            float right_disp = right_disparity.at<unsigned char>(y, x);

            // Check if the disparity difference exceeds the threshold
            if (std::abs(right_disp - left_disp) > threshold)
            {
                result.at<unsigned char>(y, x) = 0;
            }
        }
    }

    return result;
}

cv::Mat resize_image(const cv::Mat &image, int factor)
{
    // Error handling: check for valid image and factor
    if (image.empty() || factor <= 0)
    {
        // Handle error, e.g., throw exception or return empty Mat
        return cv::Mat();
    }

    // Calculate dimensions of resized image
    int resized_width = image.cols / factor;
    int resized_height = image.rows / factor;

    // Create empty output image with same data type as input
    cv::Mat resized_image(resized_height, resized_width, image.type());

// Parallelize the following loop using OpenMP
#pragma omp parallel for
    // Loop through pixels in resized image
    for (int y = 0; y < resized_height; y++)
    {
        for (int x = 0; x < resized_width; x++)
        {
            // Calculate corresponding coordinates in original image
            int input_x = x * factor;
            int input_y = y * factor;

            // Check for boundary conditions (avoid going out of image bounds)
            if (input_x < 0 || input_x >= image.cols || input_y < 0 || input_y >= image.rows)
            {
                // Handle boundary pixels (e.g., set to black, replicate edge pixels)
                continue; // Skip to next iteration
            }

            // Handle different image types
            if (image.type() == CV_8UC3)
            {
                // Access pixel values based on image data type (CV_8UC3 for color images)
                cv::Vec3b original_pixel = image.at<cv::Vec3b>(input_y, input_x);
                // Copy pixel value to corresponding location in resized image
                resized_image.at<cv::Vec3b>(y, x) = original_pixel;
            }
            else if (image.type() == CV_8UC1)
            {
                // Access pixel values based on image data type (CV_8UC1 for grayscale images)
                uchar original_pixel = image.at<uchar>(input_y, input_x);
                // Copy pixel value to corresponding location in resized image
                resized_image.at<uchar>(y, x) = original_pixel;
            }
        }
    }

    return resized_image;
}

// Function to save and display images
void save_and_show(const std::string &output_dir, const std::string &name, const cv::Mat &img, bool show_images = true)
{
    // Save the image
    std::string filename = output_dir + "/" + name + ".png"; // Save to output directory
    cv::imwrite(filename, img);

    // Show the image if desired
    if (show_images)
    {
        cv::imshow(name, img);
        cv::waitKey(0); // Wait for a key press to close the window
    }
}

bool create_directory(const std::string &path)
{
    struct stat info;
    if (stat(path.c_str(), &info) != 0) // Check if path exists
    {
        if (mkdir(path.c_str(), 0777) == 0) // Create directory
        {
            return true;
        }
        else
        {
            std::cerr << "Error creating directory: " << strerror(errno) << std::endl;
            return false;
        }
    }
    else if (info.st_mode & S_IFDIR) // Check if path is a directory
    {
        return true; // Directory already exists
    }
    else
    {
        std::cerr << "Path exists but is not a directory." << std::endl;
        return false;
    }
}

int main()
{
    auto main_start = high_resolution_clock::now();

    cv::Mat left_img = cv::imread("../images/im0.png", cv::IMREAD_GRAYSCALE);
    cv::Mat right_img = cv::imread("../images/im1.png", cv::IMREAD_GRAYSCALE);
    if (left_img.empty() || right_img.empty())
    {
        std::cerr << "Error loading images" << std::endl;
        return -1;
    }

    int patch_size = 9;      // example window size
    int max_disparity = 260; // example maximum disparity
    int image_downsample_factor = 4;
    int cross_checking_threshold = 8;
    int max_search_distance = 16;

    max_disparity = int(max_disparity / image_downsample_factor);

    left_img = resize_image(left_img, image_downsample_factor);
    right_img = resize_image(right_img, image_downsample_factor);

    int width = left_img.size().width;
    int height = left_img.size().height;

    std::cout << "Width: " << left_img.size().width << ", Height: " << left_img.size().height << std::endl;

    // Convert images to vector
    std::vector<unsigned char> leftImg(left_img.begin<unsigned char>(), left_img.end<unsigned char>());
    std::vector<unsigned char> rightImg(right_img.begin<unsigned char>(), right_img.end<unsigned char>());
    std::vector<unsigned char> disparityImgLeft(width * height);
    std::vector<unsigned char> disparityImgRight(width * height);

    // Calling calculate_disparity_map_left for the left disparity map
    calculate_disparity_map_left(leftImg, rightImg, disparityImgLeft, width, height, patch_size, max_disparity);

    // Calling calculate_disparity_map_right for the right disparity map
    calculate_disparity_map_right(leftImg, rightImg, disparityImgRight, width, height, patch_size, max_disparity);

    cv::Mat disparity_mat_left(height, width, CV_8UC1, disparityImgLeft.data());
    cv::Mat disparity_mat_right(height, width, CV_8UC1, disparityImgRight.data());

    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
    std::string timestamp = ss.str();
    std::string output_dir = "disparity_results_" + timestamp;

    if (!create_directory(output_dir))
    {
        return -1;
    }

    // Save and show images
    save_and_show(output_dir, "Left Image", left_img);
    save_and_show(output_dir, "Right Image", right_img);
    save_and_show(output_dir, "Disparity Image Left", disparity_mat_left);
    save_and_show(output_dir, "Disparity Image Right", disparity_mat_right);

    // Normalize and save disparity images
    cv::Mat normalized_left_img;
    cv::normalize(disparity_mat_left, normalized_left_img, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    save_and_show(output_dir, "Normalized Left Image", normalized_left_img);

    cv::Mat normalized_right_img;
    cv::normalize(disparity_mat_right, normalized_right_img, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    save_and_show(output_dir, "Normalized Right Image", normalized_right_img);

    // Cross-check disparity maps
    auto crossCheck_start = high_resolution_clock::now();
    cv::Mat crossCheck_disparity_map = cross_check_disparity_maps(normalized_left_img, normalized_right_img, cross_checking_threshold);
    auto crossCheck_end = high_resolution_clock::now();
    auto crossCheck_duration = duration_cast<milliseconds>(crossCheck_end - crossCheck_start);
    std::cout << "Total CrossChecking time taken: " << crossCheck_duration.count() << " milliseconds" << std::endl;

    cv::Mat normalized_crossCheck;
    cv::normalize(crossCheck_disparity_map, normalized_crossCheck, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    save_and_show(output_dir, "CrossCheck 255", normalized_crossCheck);

    // Convert the cross-checked disparity map to std::vector
    std::vector<unsigned char> crossCheck_disparity_vector;
    crossCheck_disparity_vector.assign(crossCheck_disparity_map.datastart, crossCheck_disparity_map.dataend);

    auto occlusion_start = high_resolution_clock::now();
    std::vector<unsigned char> filled_disparity_vector = fill_occlusions(crossCheck_disparity_vector, width, height, max_search_distance);
    auto occlusion_end = high_resolution_clock::now();
    auto occlusion_duration = duration_cast<milliseconds>(occlusion_end - occlusion_start);
    std::cout << "Total Occlusion time taken: " << occlusion_duration.count() << " milliseconds" << std::endl;

    // Convert the filled disparity back to cv::Mat
    cv::Mat filled_disparity_mat(height, width, CV_8UC1, filled_disparity_vector.data());

    // Normalize the filled disparity map
    cv::Mat normalized_occlusion;
    cv::normalize(filled_disparity_mat, normalized_occlusion, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // Show the normalized filled occlusion disparity map
    save_and_show(output_dir, "Occlusion 255", normalized_occlusion);

    std::ofstream param_file(output_dir + "/params.txt");
    if (param_file.is_open())
    {
        param_file << "\nParameters\n";
        param_file << "-------------\n";
        param_file << "patch_size: " << patch_size << "\n";
        param_file << "image_downsample_factor: " << image_downsample_factor << "\n";
        param_file << "max_disparity: " << max_disparity << "\n";
        param_file << "cross_checking_threshold: " << cross_checking_threshold << "\n";
        param_file << "max_search_distance: " << max_search_distance << "\n";
        param_file.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing parameters." << std::endl;
        return -1;
    }
    auto main_end = high_resolution_clock::now();
    auto main_duration = duration_cast<milliseconds>(main_end - main_start);
    std::cout << "Total Time taken: " << main_duration.count() << " milliseconds" << std::endl;

    return 0;
}
