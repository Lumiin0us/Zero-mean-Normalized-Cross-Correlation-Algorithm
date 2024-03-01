#include <iostream>
#include <cmath>
#include <chrono>

using namespace std;
using namespace std::chrono; 

// Function to compute the template image
double** template_image(double image[][4], int image_size, int window_size) {
    auto start = high_resolution_clock::now();
    bool processed_matrix[image_size][4];
    double** averages = new double*[image_size];
    for (int i = 0; i < image_size; ++i) {
        averages[i] = new double[4];
        for (int j = 0; j < 4; ++j) {
            processed_matrix[i][j] = false;
            averages[i][j] = image[i][j]; // Initialize averages with original image values
        }
    }
    for (int j = 0; j <= image_size - window_size; ++j) {
        for (int i = 0; i <= 4 - window_size; ++i) {
            double window_average = 0;
            for (int win_y = 0; win_y < window_size; ++win_y) {
                for (int win_x = 0; win_x < window_size; ++win_x) {
                    window_average += image[win_x + i][win_y + j];
                }
            }
            window_average /= window_size * 2;

            for (int win_y = (-window_size / 2) + 1; win_y < (window_size / 2) - 1; ++win_y) {
                for (int win_x = (-window_size / 2) + 1; win_x < (window_size / 2) - 1; ++win_x) {
                    if (!processed_matrix[win_x + i][win_y + j]) {
                        averages[win_x + i][win_y + j] -= window_average;
                        processed_matrix[win_x + i][win_y + j] = true;
                    }
                }
            }
        }
    }
    auto stop = high_resolution_clock::now(); // Stop measuring time
    auto duration = duration_cast<microseconds>(stop - start); // Calculate duration
    cout << "Template image function took " << duration.count() << " microseconds." << endl;
    return averages;
}


// Function to compute the target image
double** target_image(double image[][4], int image_size, int disparity_value, int window_size) {
    auto start = high_resolution_clock::now();
    bool processed_matrix[image_size][4];
    double** averages = new double*[image_size];
    for (int i = 0; i < image_size; ++i) {
        averages[i] = new double[4];
        for (int j = 0; j < 4; ++j) {
            processed_matrix[i][j] = false;
            averages[i][j] = image[i][j]; // Initialize averages with original image values
        }
    }
    for (int j = 0; j <= image_size - window_size; ++j) {
        for (int i = 0; i <= 4 - window_size; ++i) {
            for (int d = 0; d < disparity_value; ++d) {
                double window_average = 0;
                for (int win_y = (-window_size / 2) + 1; win_y < (window_size / 2) - 1; ++win_y) {
                    for (int win_x = (-window_size / 2) + 1; win_x < (window_size / 2) - 1; ++win_x) {
                        if (win_y - disparity_value > 0) {
                            window_average += image[win_x + i - disparity_value][win_y + j];
                        } else {
                            window_average += image[win_x + i][win_y + j];
                        }
                    }
                }
                window_average /= window_size * 2;

                for (int win_y = (-window_size / 2) + 1; win_y < (window_size / 2) - 1; ++win_y) {
                    for (int win_x = (-window_size / 2) + 1; win_x < (window_size / 2) - 1; ++win_x) {
                        if (!processed_matrix[win_x + i][win_y + j]) {
                            averages[win_x + i][win_y + j] -= window_average;
                            processed_matrix[win_x + i][win_y + j] = true;
                        }
                    }
                }
            }
        }
    }
    auto stop = high_resolution_clock::now(); // Stop measuring time
    auto duration = duration_cast<microseconds>(stop - start); // Calculate duration
    cout << "Target image function took " << duration.count() << " microseconds." << endl;
    return averages;
}

double** numerator(double** template_img, double** target_img, int image_size) {
    auto start = high_resolution_clock::now();
    double** result = new double*[image_size];
    for (int i = 0; i < image_size; ++i) {
        result[i] = new double[4];
        for (int j = 0; j < 4; ++j) {
            result[i][j] = template_img[i][j] * target_img[i][j];
        }
    }
    auto stop = high_resolution_clock::now(); // Stop measuring time
    auto duration = duration_cast<microseconds>(stop - start); // Calculate duration
    cout << "Numerator function took " << duration.count() << " microseconds." << endl;
    return result;
}

double** denominator(double** template_img, double** target_img, int image_size) {
    auto start = high_resolution_clock::now();
    double** result = new double*[image_size];
    for (int i = 0; i < image_size; ++i) {
        result[i] = new double[4];
        for (int j = 0; j < 4; ++j) {
            // Square each element directly
            result[i][j] = sqrt(template_img[i][j] * template_img[i][j]) * sqrt(target_img[i][j] * target_img[i][j]);
        }
    }
    auto stop = high_resolution_clock::now(); // Stop measuring time
    auto duration = duration_cast<microseconds>(stop - start); // Calculate duration
    cout << "Denominator function took " << duration.count() << " microseconds." << endl;
    return result;
}

// Function to compute ZNCC
double** zncc(double** num, double** den, int image_size) {
    auto start = high_resolution_clock::now();
    double** result = new double*[image_size];
    for (int i = 0; i < image_size; ++i) {
        result[i] = new double[4];
        for (int j = 0; j < 4; ++j) {
            if (den[i][j] != 0) {
                result[i][j] = num[i][j] / den[i][j];
            } else {
                result[i][j] = 0; // Handle division by zero
            }
        }
    }
    auto stop = high_resolution_clock::now(); // Stop measuring time
    auto duration = duration_cast<microseconds>(stop - start); // Calculate duration
    cout << "zncc function took " << duration.count() << " microseconds." << endl;
    return result;
}

int main() {
    auto start = high_resolution_clock::now();
    const int image_size = 4;
    double image[image_size][4] = {
        {1, 2, 33, 41},
        {10, 2, 25, 1},
        {9, 15, 5, 20},
        {22, 1, 1, 24}
    };

    double template_img[image_size][4];
    double target_img[image_size][4];

    double** template_img_average = template_image(image, image_size, 2);
    double** target_img_average = target_image(image, image_size, 2, 2);

    double** num = numerator(template_img_average, target_img_average, image_size);
    double** den = denominator(template_img_average, target_img_average, image_size);


    // Print numerator
    cout << "Numerator:" << endl;
    for (int i = 0; i < image_size; ++i) {
        for (int j = 0; j < 4; ++j) {
            cout << num[i][j] << " ";
        }
        cout << endl;
    }

    // Print denominator
    cout << "Denominator:" << endl;
    for (int i = 0; i < image_size; ++i) {
        for (int j = 0; j < 4; ++j) {
            cout << den[i][j] << " ";
        }
        cout << endl;
    }

    double** zncc_matrix = zncc(num, den, image_size);
    for (int i = 0; i < image_size; ++i) {
        for (int j = 0; j < 4; ++j) {
            cout << zncc_matrix[i][j] << " ";
        }
        cout << endl;
    }
    auto stop = high_resolution_clock::now(); // Stop measuring time
    auto duration = duration_cast<microseconds>(stop - start); // Calculate duration
    cout << "Entire code took " << duration.count() << " microseconds." << endl;
    return 0;
}
