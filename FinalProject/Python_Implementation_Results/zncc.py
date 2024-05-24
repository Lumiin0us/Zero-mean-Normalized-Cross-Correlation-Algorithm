import cv2
import numpy as np
import os
from datetime import datetime
import time


patch_size = 21               # Size of the image patches to compare
image_downsample_factor = 4
cross_checking_threshold = 8

"""
patch_size = 3 
image_downsample_factor = 4
cross_checking_threshold = 20


patch_size = 5
image_downsample_factor = 4
cross_checking_threshold = 20


patch_size = 5
image_downsample_factor = 4
cross_checking_threshold = 8


patch_size = 9
image_downsample_factor = 4
cross_checking_threshold = 8


patch_size = 9
image_downsample_factor = 4
cross_checking_threshold = 20


patch_size = 19
image_downsample_factor = 4
cross_checking_threshold = 80


patch_size = 19
image_downsample_factor = 4
cross_checking_threshold = 8


patch_size = 19
image_downsample_factor = 4
cross_checking_threshold = 20


patch_size = 19
image_downsample_factor = 4
cross_checking_threshold = 50


patch_size = 19
image_downsample_factor = 4
cross_checking_threshold = 200


patch_size = 19
image_downsample_factor = 4
cross_checking_threshold = 150

patch_size = 19
image_downsample_factor = 4
cross_checking_threshold = 4

"""


max_disparity = 260   # Maximum possible disparity to search for 
max_disparity = int(max_disparity / image_downsample_factor)




timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"disparity_results_{timestamp}"

SHOW_IMAGES = False
PROFILE = False

if PROFILE:
    import cProfile
    import pstats
    from pstats import SortKey
    #from memory_profiler import profile

#@profile
def resize_image(image):
    """Resizes an image to some factor N of its original size by taking every Nth pixel."""

    # Ensure the image is a numpy array
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Slice every 4th row and column
    resized_image = image[::image_downsample_factor, ::image_downsample_factor]

    return resized_image

# ZNCC Calculation Function
#@profile
def calculate_zncc(patch1, patch2):
    """Calculates ZNCC between two image patches."""

    # Zero-Mean: Subtract the mean intensity from each patch
    mean1 = np.mean(patch1)
    mean2 = np.mean(patch2)
    patch1_zero_mean = patch1 - mean1
    patch2_zero_mean = patch2 - mean2

    # Normalization: Divide each pixel by its patch's standard deviation
    std1 = np.std(patch1_zero_mean)
    std2 = np.std(patch2_zero_mean)
    
    """
    print("patch1", patch1)
    print("patch2", patch2)
    print("mean1", mean1)
    print("mean2", mean2)
    print("patch1_zero_mean", patch1_zero_mean)
    print("patch2_zero_mean", patch2_zero_mean)
    print("std1", std1)
    print("std2", std2)
    """

    # Handle potential division by zero if either patch is entirely uniform
    if std1 == 0 or std2 == 0:
        return 0  # No correlation in this case

    # Cross-Correlation: Calculate the normalized cross-correlation
    numerator = np.sum(patch1_zero_mean * patch2_zero_mean)
    denominator = std1 * std2
    zncc = numerator / denominator
    #print("zncc:", zncc)
    return zncc

# Disparity Map Calculation (Right Image as Reference)
#@profile
def calculate_disparity_map_right(right_img, left_img, patch_size, max_disparity):
    """Calculates the disparity map using ZNCC, with the right image as reference."""

    height, width = right_img.shape
    disparity_map = np.zeros((height, width))  

    for y in range(patch_size // 2, height - patch_size // 2): 
        for x in range(patch_size // 2, width - patch_size // 2):  

            # Extract the right patch centered at (x, y)
            right_patch = right_img[y - patch_size // 2 : y + patch_size // 2 + 1,
                                    x - patch_size // 2 : x + patch_size // 2 + 1]

            best_match = 0 
            best_zncc = float('-inf')

            # Search for the best matching patch in the *left* image
            # Loop through possible disparities
            for d in range(min(max_disparity, width - x - patch_size // 2)):  
                x_left = x + d  # Calculate corresponding x in the left image
                left_patch = left_img[y - patch_size // 2 : y + patch_size // 2 + 1,
                                      x_left - patch_size // 2 : x_left + patch_size // 2 + 1]

                zncc = calculate_zncc(right_patch, left_patch)  # Order of arguments is reverse from when we use left image as reference
                if zncc > best_zncc:
                    best_zncc = zncc
                    best_match = d

            disparity_map[y, x] = -best_match  # Store negative disparity for right reference

    return disparity_map

# Disparity Map Calculation (Left Image as Reference)
#@profile
def calculate_disparity_map(left_img, right_img, patch_size, max_disparity):
    """Calculates the disparity map using ZNCC."""

    height, width = left_img.shape
    disparity_map = np.zeros((height, width))  # Initialize with zeros

    for y in range(patch_size // 2, height - patch_size // 2):  # Iterate over rows (excluding borders)
        for x in range(patch_size // 2, width - patch_size // 2):  # Iterate over columns (excluding borders)

            # Extract the left patch centered at (x, y)
            left_patch = left_img[y - patch_size // 2 : y + patch_size // 2 + 1,
                                  x - patch_size // 2 : x + patch_size // 2 + 1]

            best_match = 0  # Initialize best disparity
            best_zncc = float('-inf')   # Initialize lowest possible correlation

            # Search for the best matching patch in the right image
            # Loop through possible disparities
            for d in range(min(max_disparity, x - patch_size//2 + 1)):  # Prevent negative indices (going off the image)
                x_right = x - d  # No need for max(0, x - d) since we ensure d is not too large
                right_patch = right_img[y - patch_size // 2 : y + patch_size // 2 + 1,
                                        x_right - patch_size // 2 : x_right + patch_size // 2 + 1]

                """
                print("y:", y)
                print("x:", x)
                print("x_right:", x_right)
                print("d:", d)
                """
                zncc = calculate_zncc(left_patch, right_patch)
                if zncc > best_zncc:  # Update if better match found
                    best_zncc = zncc
                    best_match = d

            disparity_map[y, x] = best_match  # Store the best disparity for the current pixel

    return disparity_map

# Depth Estimation (Triangulation) - Assuming focal length is in pixels and baseline in millimeters
#@profile
def estimate_depth(disparity_map, baseline=174.945, focal_length=7190.247):
    """Estimates depth from disparity map using triangulation."""
    depth_map = np.zeros_like(disparity_map, dtype=np.float32)

    # Iterate over each pixel and calculate depth
    for y in range(disparity_map.shape[0]):
        for x in range(disparity_map.shape[1]):
            disparity = disparity_map[y, x]
            if disparity > 0:
                depth_map[y, x] = baseline * focal_length / disparity  # Depth in mm
    
    # Normalize depth map to 0-255 range
    #depth_map_normalized = cv2.normalize(
    #    depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    #)
    #return depth_map_normalized
    return depth_map


#@profile
def cross_check_disparity_maps(left_disparity, right_disparity, threshold=1):
    """Performs cross-checking between left and right disparity maps.

    Args:
        left_disparity: The disparity map calculated with the left image as reference.
        right_disparity: The disparity map calculated with the right image as reference.
        threshold: The maximum allowed difference between corresponding disparities.

    Returns:
        The left disparity map with invalidated disparities.
    """
    
    # Ensure disparity maps have the same shape
    assert left_disparity.shape == right_disparity.shape, "Disparity maps must have the same shape"

    height, width = left_disparity.shape

    # Iterate over the disparity maps
    for y in range(height):
        for x in range(width):
            # Get disparities for the current pixel from both maps
            left_disp = left_disparity[y, x]
            right_disp = right_disparity[y, x]

            # Check if the disparities are consistent within the threshold
            #if abs(int(left_disp) - int(right_disp)) > threshold:  # Symmetrical check - Checks if magnitude difference is higher than threshold
            #print(left_disp, right_disp, left_disp - right_disp, abs(left_disp - right_disp))
            if abs(left_disp - right_disp) > threshold:  # Symmetrical check - Checks if magnitude difference is higher than threshold
                left_disparity[y, x] = 0  # Invalidate if inconsistent
    
    return left_disparity


def fill_occlusions(disparity_map, max_search_distance):
    """Fills occluded regions (invalid disparities) using neighboring valid values."""
    filled_disparity = disparity_map.copy()  # Make a copy to avoid modifying original
    height, width = disparity_map.shape

    for y in range(height):
        for x in range(width):
            if disparity_map[y, x] == 0:  # Check for invalid disparity (occlusion)
                for d in range(1, max_search_distance + 1):
                    # Check neighbors in 4 directions
                    for dx, dy in [(0, d), (0, -d), (d, 0), (-d, 0)]:
                        nx, ny = x + dx, y + dy  # Neighbor coordinates
                        if 0 <= nx < width and 0 <= ny < height and disparity_map[ny, nx] != 0:
                            filled_disparity[y, x] = disparity_map[ny, nx]  # Fill with neighbor value
                            break  # Stop searching once a valid neighbor is found
                    else:
                        continue  # Continue to the next neighbor if no valid value was found
                    break  # Exit the inner loop if a valid neighbor was found

    return filled_disparity


def save_and_show(name, img):
    cv2.imwrite(f"{output_dir}/{name}.png", img)  
    if SHOW_IMAGES:
        cv2.imshow(name, img)

def show_img_wait():
    if SHOW_IMAGES:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():

    os.makedirs(output_dir, exist_ok=True)  # Create directory, handle if it exists

    with open(os.path.join(output_dir, "params.txt"), "w") as f:
        f.write("\nParameters\n")
        f.write("-------------\n")
        f.write(f"patch_size: {patch_size}\n")
        f.write(f"image_downsample_factor: {image_downsample_factor}\n")
        f.write(f"max_disparity: {max_disparity}\n")
        f.write(f"cross_checking_threshold: {cross_checking_threshold}\n")


    start_time = time.time()  # Record overall script start time

    # Load rectified stereo images in grayscale
    #left_img = cv2.imread('file0.png', cv2.IMREAD_GRAYSCALE)
    #right_img = cv2.imread('file1.png', cv2.IMREAD_GRAYSCALE)

    if PROFILE:
        profiler = cProfile.Profile()  # Create a profiler object
        profiler.enable()  # Start profiling


    left_img_read_start_time = time.time()
    left_img = cv2.imread('img0.png', cv2.IMREAD_GRAYSCALE)
    left_img_read_time = time.time() - left_img_read_start_time

    right_img_read_start_time = time.time()
    right_img = cv2.imread('img1.png', cv2.IMREAD_GRAYSCALE)
    right_img_read_time = time.time() - right_img_read_start_time

    resize_left_start_time = time.time()
    left_img = resize_image(left_img)
    resize_left_time = time.time() - resize_left_start_time

    resize_right_start_time = time.time()
    right_img = resize_image(right_img)
    resize_right_time = time.time() - resize_right_start_time


    save_and_show('Resized left_img', left_img)
    save_and_show('Resized right_img', right_img)
    show_img_wait()


    disparity_start_time = time.time()
    disparity_map_left = calculate_disparity_map(left_img, right_img, patch_size, max_disparity)
    disparity_left_time = time.time() - disparity_start_time

    save_and_show("Disparity Map Left Original", disparity_map_left)
    save_and_show("Disparity Map Left 0 - 255", cv2.normalize(disparity_map_left, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))
    save_and_show("Disparity Map Left / max_disparity", disparity_map_left/max_disparity) # Dividing the disparity values by max_disparity scales them down to a range between 0 and 1. This makes the disparity map visually interpretable such that darker pixels represent smaller disparities (farther objects) and brighter pixels represent larger disparities (closer objects).


    depth_start_time = time.time()
    depth_map = estimate_depth(disparity_map_left) 
    #depth_map = estimate_depth(disparity_map_left/max_disparity) 
    depth_time = time.time() - depth_start_time


    #cv2.imshow("Depth Map", depth_map/np.max(depth_map)) # Normalizing here helps with enhancing contrast - The original normalized depth map might have a wide range of values, but not necessarily spanning the full 0-255 range. Dividing by the maximum value ensures that the brightest point in the depth map will be mapped to 255 (pure white), and the contrast between different depth levels will be maximized. 
    save_and_show("Depth Map Original", depth_map) 
    save_and_show("Depth Map / np.max(depth_map)", depth_map/np.max(depth_map)) 
    save_and_show("Depth Map 0 - 255", cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)) 

    show_img_wait()



    disparity_right_start_time = time.time()
    disparity_map_right = calculate_disparity_map_right(right_img, left_img, patch_size, max_disparity)
    disparity_right_time = time.time() - disparity_right_start_time

    save_and_show("Disparity Map Right Original", disparity_map_right)
    save_and_show("Disparity Map Right 0 - 255", cv2.normalize(disparity_map_right, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))
    save_and_show("Disparity Map Right / max_disparity", disparity_map_right/max_disparity) # Dividing the disparity values by max_disparity scales them down to a range between 0 and 1. This makes the disparity map visually interpretable such that darker pixels represent smaller disparities (farther objects) and brighter pixels represent larger disparities (closer objects).
    
    show_img_wait()



    cross_check_start_time = time.time()
    disparity_map_left_cross_checked = cross_check_disparity_maps(disparity_map_left, disparity_map_right, threshold=cross_checking_threshold)
    cross_check_time = time.time() - cross_check_start_time


    if PROFILE:
        profiler.disable() # Stop profiling
        # Process the profile statistics
        stats = pstats.Stats(profiler)
        stats.sort_stats(SortKey.TIME)  # Sort by time (or other metrics)
        stats.print_stats()  # Print the results to the console



    save_and_show("Disparity Map Left - Cross Checked", disparity_map_left_cross_checked) 
    save_and_show("Disparity Map Left - Cross Checked 0 - 255", cv2.normalize(disparity_map_left_cross_checked, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))
    show_img_wait()


    # Fill occlusions in the left disparity map
    disparity_map_left_filled = fill_occlusions(disparity_map_left_cross_checked, max_search_distance=5)
    save_and_show("Disparity Map Left - Cross Checked and Filled", cv2.normalize(disparity_map_left_filled, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)) 


    with open(os.path.join(output_dir, "timings.txt"), "w") as timings_file:
        total_time = time.time() - start_time
        timings_file.write(f"Total Time: {total_time:.8f} seconds\n")

        timings_file.write(f"Read Left Image Time: {left_img_read_time:.8f} seconds\n")
        timings_file.write(f"Read Right Image Time: {right_img_read_time:.8f} seconds\n")

        timings_file.write(f"Resize Left Image Time: {resize_left_time:.8f} seconds\n")
        timings_file.write(f"Resize Right Image Time: {resize_right_time:.8f} seconds\n")
        
        timings_file.write(f"Disparity Map Left Calculation Time: {disparity_left_time:.8f} seconds\n")
        timings_file.write(f"Disparity Map Right Calculation Time: {disparity_right_time:.8f} seconds\n")
        
        timings_file.write(f"Depth Map Calculation Time: {depth_time:.8f} seconds\n")
        
        timings_file.write(f"Cross-Checking Time: {cross_check_time:.8f} seconds\n")

        timings_file.write("\nParameters\n")
        timings_file.write("-------------\n")
        timings_file.write(f"patch_size: {patch_size}\n")
        timings_file.write(f"image_downsample_factor: {image_downsample_factor}\n")
        timings_file.write(f"max_disparity: {max_disparity}\n")
        timings_file.write(f"cross_checking_threshold: {cross_checking_threshold}\n")



main()
"""
os.makedirs(output_dir, exist_ok=True)  # Create directory, handle if it exists

"""
"""
disparity_map_left = cv2.imread('Disparity Map Left Original.png', cv2.IMREAD_GRAYSCALE)
disparity_map_right = cv2.imread('Disparity Map Right Original.png', cv2.IMREAD_GRAYSCALE)
disparity_map_left_cross_checked = cross_check_disparity_maps(disparity_map_left, disparity_map_right, threshold=cross_checking_threshold)
save_and_show("Disparity Map Left - Cross Checked 0 - 255", cv2.normalize(disparity_map_left_cross_checked, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))

disparity_map_left_filled = fill_occlusions(disparity_map_left_cross_checked, max_search_distance=5)
save_and_show("Disparity Map Left - Cross Checked and Filled", cv2.normalize(disparity_map_left_filled, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)) 
"""
"""

disparity_map_left = cv2.imread('Disparity Map Left Original.png', cv2.IMREAD_GRAYSCALE)
import cv2
filtered_disparity = cv2.medianBlur(disparity_map_left, 5)  # 5 is the kernel size
save_and_show("Disparity Map Left - filtered_disparity 5 original", filtered_disparity) 
save_and_show("Disparity Map Left - filtered_disparity 5 normalized", cv2.normalize(filtered_disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)) 

# Median Filter (already in your code)
disparity_map_left_median = cv2.medianBlur(disparity_map_left.astype(np.uint8), 5)
save_and_show("Disparity Map Left - Median Filtered", cv2.normalize(disparity_map_left_median, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))

# Gaussian Filter
disparity_map_left_gaussian = cv2.GaussianBlur(disparity_map_left.astype(np.uint8), (5, 5), 0)  # Kernel size 5x5
save_and_show("Disparity Map Left - Gaussian Filtered", cv2.normalize(disparity_map_left_gaussian, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))

# Bilateral Filter
disparity_map_left_bilateral = cv2.bilateralFilter(disparity_map_left.astype(np.uint8), 3, 100, 100)
save_and_show("Disparity Map Left - Bilateral Filtered", cv2.normalize(disparity_map_left_bilateral, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))

"""




