#include <iostream>
#include <opencv2/opencv.hpp>
#include <numeric> // For std::accumulate
#include <iomanip>
#include <sstream>
#include <string>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

using namespace std;
using namespace std::chrono;

double calculate_zncc(const cv::Mat &patch1, const cv::Mat &patch2)
{
    // auto start = high_resolution_clock::now();

    // if (patch1.size() != patch2.size() || patch1.channels() != 1 || patch2.channels() != 1)
    // {
    //     std::cerr << "Patches must be the same size and single-channel." << std::endl;
    //     return 0;
    // }

    cv::Scalar mean1 = cv::mean(patch1);
    cv::Scalar mean2 = cv::mean(patch2);
    cv::Mat patch1_zero_mean = patch1 - mean1[0];
    cv::Mat patch2_zero_mean = patch2 - mean2[0];

    cv::Scalar std1, std2;
    cv::meanStdDev(patch1_zero_mean, cv::noArray(), std1);
    cv::meanStdDev(patch2_zero_mean, cv::noArray(), std2);

    if (std1[0] == 0 || std2[0] == 0)
    {
        return 0;
    }

    double numerator = patch1_zero_mean.dot(patch2_zero_mean);
    double denominator = std1[0] * std2[0];
    double zncc = numerator / denominator;

    // auto stop = high_resolution_clock::now();                  // Stop measuring time
    // auto duration = duration_cast<microseconds>(stop - start); // Calculate duration
    // cout << "Calculate ZNCC function took: " << duration.count() << " microseconds." << endl;

    return zncc;
}

// cv::Mat resize_image(const cv::Mat &image, int factor)
// {
//     auto start = high_resolution_clock::now();

//     cv::Mat resized_image;
//     cv::resize(image, resized_image, cv::Size(), 1.0 / factor, 1.0 / factor, cv::INTER_NEAREST);

//     auto stop = high_resolution_clock::now();                  // Stop measuring time
//     auto duration = duration_cast<microseconds>(stop - start); // Calculate duration
//     cout << "Image Resize function took: " << duration.count() << " microseconds." << endl;

//     return resized_image;
// }

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

            // Access pixel values based on image data type (assuming CV_8UC3 for color images)
            cv::Vec3b original_pixel = image.at<cv::Vec3b>(input_y, input_x);

            // Copy pixel value to corresponding location in resized image
            resized_image.at<cv::Vec3b>(y, x) = original_pixel;
        }
    }

    return resized_image;
}

cv::Mat calculate_disparity_map_left(const cv::Mat &left_img, const cv::Mat &right_img, int patch_size, int max_disparity)
{
    auto start = high_resolution_clock::now();

    int height = left_img.rows;
    int width = left_img.cols;
    cv::Mat disparity_map = cv::Mat::zeros(height, width, CV_32F);

    for (int y = patch_size / 2; y < height - patch_size / 2; ++y)
    {
        for (int x = patch_size / 2; x < width - patch_size / 2; ++x)
        {
            cv::Mat left_patch = left_img(cv::Rect(x - patch_size / 2, y - patch_size / 2, patch_size, patch_size));

            int best_match = 0;
            double best_zncc = -std::numeric_limits<double>::infinity();

            for (int d = 0; d < std::min(max_disparity, x - patch_size / 2 + 1); ++d)
            {
                int x_right = x - d;
                cv::Mat right_patch = right_img(cv::Rect(x_right - patch_size / 2, y - patch_size / 2, patch_size, patch_size));

                double zncc = calculate_zncc(left_patch, right_patch);
                if (zncc > best_zncc)
                {
                    best_zncc = zncc;
                    best_match = d;
                }
            }
            disparity_map.at<float>(y, x) = best_match;
        }
    }

    auto stop = high_resolution_clock::now();                  // Stop measuring time
    auto duration = duration_cast<microseconds>(stop - start); // Calculate duration
    cout << "Calculation of left dispairty map took: " << duration.count() << " microseconds." << endl;

    return disparity_map;
}

cv::Mat calculate_disparity_map_right(const cv::Mat &right_img, const cv::Mat &left_img, int patch_size, int max_disparity)
{
    auto start = high_resolution_clock::now();

    int height = right_img.rows;
    int width = right_img.cols;
    cv::Mat disparity_map = cv::Mat::zeros(height, width, CV_32F);

    for (int y = patch_size / 2; y < height - patch_size / 2; ++y)
    {
        for (int x = patch_size / 2; x < width - patch_size / 2; ++x)
        {
            cv::Mat right_patch = right_img(cv::Rect(x - patch_size / 2, y - patch_size / 2, patch_size, patch_size));

            int best_match = 0;
            double best_zncc = -std::numeric_limits<double>::infinity();

            for (int d = 0; d < std::min(max_disparity, width - x - patch_size / 2); ++d)
            {
                int x_left = x + d;
                cv::Mat left_patch = left_img(cv::Rect(x_left - patch_size / 2, y - patch_size / 2, patch_size, patch_size));

                double zncc = calculate_zncc(right_patch, left_patch);
                if (zncc > best_zncc)
                {
                    best_zncc = zncc;
                    best_match = d;
                }
            }
            disparity_map.at<float>(y, x) = -best_match;
        }
    }

    auto stop = high_resolution_clock::now();                  // Stop measuring time
    auto duration = duration_cast<microseconds>(stop - start); // Calculate duration
    cout << "Calculation of right dispairty map took: " << duration.count() << " microseconds." << endl;

    return disparity_map;
}

cv::Mat cross_check_disparity_maps(const cv::Mat &left_disparity, const cv::Mat &right_disparity, int threshold)
{
    auto start = high_resolution_clock::now();

    CV_Assert(left_disparity.size() == right_disparity.size() && left_disparity.type() == right_disparity.type());

    int height = left_disparity.rows;
    int width = left_disparity.cols;

    cv::Mat result = left_disparity.clone();

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            float left_disp = left_disparity.at<float>(y, x);
            float right_disp = right_disparity.at<float>(y, x);

            if (std::abs(left_disp - right_disp) > threshold)
            {
                result.at<float>(y, x) = 0;
            }
        }
    }

    auto stop = high_resolution_clock::now();                  // Stop measuring time
    auto duration = duration_cast<microseconds>(stop - start); // Calculate duration
    cout << "Cross checking disparity maps took: " << duration.count() << " microseconds." << endl;

    return result;
}

// cv::Mat fill_occlusions(const cv::Mat &disparity_map, int max_search_distance)
// {
//     cv::Mat filled_disparity = disparity_map.clone(); // Make a copy to avoid modifying the original
//     int height = disparity_map.rows;
//     int width = disparity_map.cols;

//     for (int y = 0; y < height; ++y)
//     {
//         for (int x = 0; x < width; ++x)
//         {
//             if (disparity_map.at<float>(y, x) == 0)
//             { // Check for invalid disparity (occlusion)
//                 for (int d = 1; d <= max_search_distance; ++d)
//                 {
//                     // Check neighbors in 4 directions
//                     bool found_valid_neighbor = false;
//                     for (const auto &[dx, dy] : std::vector<std::pair<int, int>>{{0, d}, {0, -d}, {d, 0}, {-d, 0}})
//                     {
//                         int nx = x + dx;
//                         int ny = y + dy; // Neighbor coordinates
//                         if (nx >= 0 && nx < width && ny >= 0 && ny < height && disparity_map.at<float>(ny, nx) != 0)
//                         {
//                             filled_disparity.at<float>(y, x) = disparity_map.at<float>(ny, nx); // Fill with neighbor value
//                             found_valid_neighbor = true;
//                             break; // Stop searching once a valid neighbor is found
//                         }
//                     }
//                     if (found_valid_neighbor)
//                     {
//                         break; // Exit the inner loop if a valid neighbor was found
//                     }
//                 }
//             }
//         }
//     }

//     return filled_disparity;
// }
cv::Mat fill_occlusions(const cv::Mat &disparity_map, int max_search_distance)
{
    cv::Mat filled_disparity = disparity_map.clone(); // Make a copy to avoid modifying the original
    int height = disparity_map.rows;
    int width = disparity_map.cols;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            if (disparity_map.at<float>(y, x) == 0)
            { // Check for invalid disparity (occlusion)
                for (int d = 1; d <= max_search_distance; ++d)
                {
                    // Check neighbors in 4 directions
                    bool found_valid_neighbor = false;
                    // Option 1: Manual Unpacking (preferred for readability)
                    const std::pair<int, int> offsets[] = {{0, d}, {0, -d}, {d, 0}, {-d, 0}};
                    for (const std::pair<int, int> &neighbor : offsets)
                    {
                        int dx = neighbor.first;
                        int dy = neighbor.second;
                        int nx = x + dx;
                        int ny = y + dy; // Neighbor coordinates
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height && disparity_map.at<float>(ny, nx) != 0)
                        {
                            filled_disparity.at<float>(y, x) = disparity_map.at<float>(ny, nx); // Fill with neighbor value
                            found_valid_neighbor = true;
                            break; // Stop searching once a valid neighbor is found
                        }
                    }

                    // Option 2: Loop with Index (alternative)
                    /*
                    for (int i = 0; i < 4; ++i) {
                      int dx = offsets[i].first;
                      int dy = offsets[i].second;
                      // ... rest of the loop body using dx and dy
                    }
                    */

                    if (found_valid_neighbor)
                    {
                        break; // Exit the inner loop if a valid neighbor was found
                    }
                }
            }
        }
    }

    return filled_disparity;
}

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

    auto start = high_resolution_clock::now();

    cv::Mat left_img = cv::imread("./images/im0.png", cv::IMREAD_GRAYSCALE);
    cv::Mat right_img = cv::imread("./images/im1.png", cv::IMREAD_GRAYSCALE);

    if (left_img.empty() || right_img.empty())
    {
        std::cerr << "Error loading images" << std::endl;
        return -1;
    }

    int patch_size = 21;
    int max_disparity = 260;
    int cross_checking_threshold = 8;
    int image_downsample_factor = 4;
    int max_search_distance = 5;

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

    max_disparity = int(max_disparity / image_downsample_factor);

    // cv::imshow("Orignal left_img", left_img);
    // cv::imshow("Orignal right_img", right_img);

    left_img = resize_image(left_img, image_downsample_factor);
    right_img = resize_image(right_img, image_downsample_factor);

    // cv::imshow("Resized left_img", left_img);
    // cv::imshow("Resized right_img", right_img);

    // cv::waitKey(0);
    // cv::destroyAllWindows();

    std::cout << "Width: " << left_img.size().width << ", Height: " << left_img.size().height << std::endl;
    // return 0;

    cv::Mat disparity_map_left = calculate_disparity_map_left(left_img, right_img, patch_size, max_disparity);
    // cv::Mat depth_map = estimate_depth(disparity_map_left, 1.0, 1.0);
    save_and_show(output_dir, "Disparity Map Left Original", disparity_map_left);
    // Save and show with displaying
    save_and_show(output_dir, "Disparity Map Left / max_disparity", disparity_map_left / max_disparity, true); // Save and show with displaying

    cv::Mat disparity_map_left_normalized;
    cv::normalize(disparity_map_left, disparity_map_left_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
    save_and_show(output_dir, "Disparity Map Left 0 - 255", disparity_map_left_normalized, true);

    cv::Mat disparity_map_right = calculate_disparity_map_right(right_img, left_img, patch_size, max_disparity);
    save_and_show(output_dir, "Disparity Map Right Original", disparity_map_right, true);
    save_and_show(output_dir, "Disparity Map Right / max_disparity", disparity_map_right / max_disparity, true); // Save and show with displaying

    cv::Mat disparity_map_right_normalized;
    cv::normalize(disparity_map_right, disparity_map_right_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
    save_and_show(output_dir, "Disparity Map Right 0 - 255", disparity_map_right_normalized, true); // Save and show with displaying

    // cv::imshow("Disparity Map Left Original", disparity_map_left);
    // cv::imshow("Disparity Map Left 0 - 255", disparity_map_left_normalized);
    // cv::imshow("Disparity Map Left / max_disparity", disparity_map_left / max_disparity);

    // cv::imshow("Disparity Map Right Original", disparity_map_right);
    // cv::imshow("Disparity Map Right 0 - 255", disparity_map_right_normalized);
    // cv::imshow("Disparity Map Right / max_disparity", disparity_map_right / max_disparity);

    cv::Mat disparity_map_left_cross_checked = cross_check_disparity_maps(disparity_map_left, disparity_map_right, cross_checking_threshold);
    // cv::imshow("Disparity Map Left cross checked", disparity_map_left_cross_checked);

    cv::Mat disparity_map_left_cross_checked_normalized;
    cv::normalize(disparity_map_left_cross_checked, disparity_map_left_cross_checked_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
    // cv::imshow("Disparity Map Left cross checked normalized", disparity_map_left_cross_checked_normalized);

    cv::Mat disparity_map_left_filled = fill_occlusions(disparity_map_left_cross_checked, max_search_distance);
    // cv::imshow("Disparity Map With Occlusion", disparity_map_left_filled);

    cv::Mat disparity_map_crossCheck;
    cv::Mat disparity_map = cross_check_disparity_maps(disparity_map_left_normalized, disparity_map_right_normalized, cross_checking_threshold);
    cv::normalize(disparity_map, disparity_map_crossCheck, 0, 255, cv::NORM_MINMAX, CV_8U);

    save_and_show(output_dir, "Disparity Map Left - Cross Checked", disparity_map_crossCheck, true); // Save and show with displaying

    save_and_show(output_dir, "Disparity Map Left cross checked", disparity_map_left_cross_checked, true);                       // Save and show with displaying
    save_and_show(output_dir, "Disparity Map Left cross checked normalized", disparity_map_left_cross_checked_normalized, true); // Save and show with displaying
    save_and_show(output_dir, "Disparity Map With Occlusion", disparity_map_left_filled, true);                                  // Save and show with displaying
    // cv::imwrite("/Users/abdurrehman/C++Prac/Zero-mean-Normalized-Cross-Correlation-Algorithm/zncc_final_outputs/disparity_map_left_original.png", disparity_map_left);                // Save the original disparity map for left image
    // cv::imwrite("/Users/abdurrehman/C++Prac/Zero-mean-Normalized-Cross-Correlation-Algorithm/zncc_final_outputs/disparity_map_left_normalized.png", disparity_map_left_normalized);   // Save the normalized disparity map for left image
    // cv::imwrite("/Users/abdurrehman/C++Prac/Zero-mean-Normalized-Cross-Correlation-Algorithm/zncc_final_outputs/disparity_map_left_divided.png", disparity_map_left / max_disparity); // Save the disparity map divided by max_disparity for left image

    // cv::imwrite("/Users/abdurrehman/C++Prac/Zero-mean-Normalized-Cross-Correlation-Algorithm/zncc_final_outputs/disparity_map_right_original.png", disparity_map_right);                // Save the original disparity map for right image
    // cv::imwrite("/Users/abdurrehman/C++Prac/Zero-mean-Normalized-Cross-Correlation-Algorithm/zncc_final_outputs/disparity_map_right_normalized.png", disparity_map_right_normalized);   // Save the normalized disparity map for right image
    // cv::imwrite("/Users/abdurrehman/C++Prac/Zero-mean-Normalized-Cross-Correlation-Algorithm/zncc_final_outputs/disparity_map_right_divided.png", disparity_map_right / max_disparity); // Save the disparity map divided by max_disparity for right image

    // cv::waitKey(0);
    // cv::destroyAllWindows();

    // cv::imshow("Depth Map Original", depth_map);
    // cv::imshow("Depth Map / max depth", depth_map / cv::norm(depth_map, cv::NORM_INF));
    // cv::Mat depth_map_normalized;
    // cv::normalize(depth_map, depth_map_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
    // cv::imshow("Depth Map 0 - 255", depth_map_normalized);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    // cv::imwrite("/Users/abdurrehman/C++Prac/Zero-mean-Normalized-Cross-Correlation-Algorithm/zncc_final_outputs/disparity_map_crossCheck.png", disparity_map_crossCheck);
    // cv::imshow("Disparity Map Left - Cross Checked", disparity_map_crossCheck);
    // cv::waitKey(0);
    // cv::destroyAllWindows();
    auto stop = high_resolution_clock::now();                  // Stop measuring time
    auto duration = duration_cast<microseconds>(stop - start); // Calculate duration
    cout << "Entire code took: " << duration.count() << " microseconds." << endl;
    return 0;
}
