import numpy as np 

image = [
    [1, 2, 33, 41],
    [10, 2, 25, 1],
    [9, 15, 5, 20],
    [22, 1, 1, 24],
]

def template_image(image, window_size):
    processed_matrix = [[False] * len(image[0]) for _ in range(len(image))]
    averages = image
    print(averages)
    width = image[0]
    height = image
    for j in range(len(height) - window_size + 1):
        for i in range(len(width) - window_size + 1):
            window_average = 0
            for win_y in range(window_size):
                for win_x in range(window_size):
                        window_average += image[win_x + i][win_y + j]
            window_average /= window_size * 2

            for win_y in range((-window_size // 2) + 1, (window_size // 2) - 1):
                for win_x in range((-window_size // 2) + 1, (window_size // 2) - 1):
                    if processed_matrix[win_x + i][win_y + j] == False:
                        averages[win_x + i][win_y + j] -= window_average
                        processed_matrix[win_x + i][win_y + j] = True
    return averages

def target_image(image, disparity_value, window_size):
    processed_matrix = [[False] * len(image[0]) for _ in range(len(image))]
    averages = image
    print(averages)
    width = image[0]
    height = image
    for j in range(len(height) - window_size + 1):
        for i in range(len(width) - window_size + 1):
            for d in range(disparity_value):
                window_average = 0
                for win_y in range((-window_size // 2) + 1, (window_size // 2) - 1):
                    for win_x in range((-window_size // 2) + 1, (window_size // 2) - 1):
                        if win_y - disparity_value > 0:
                            window_average += image[win_x + i - disparity_value][win_y + j]
                        else:
                            window_average += image[win_x + i][win_y + j]
                window_average /= window_size * 2

                for win_y in range((-window_size // 2) + 1, (window_size // 2) - 1):
                    for win_x in range((-window_size // 2) + 1, (window_size // 2) - 1):
                        if processed_matrix[win_x + i][win_y + j] == False:
                            averages[win_x + i][win_y + j] -= window_average
                            processed_matrix[win_x + i][win_y + j] = True
    return averages

def numerator(template, target):
    return np.matmul(template, target)
def denominator(template, target):
    template **= 2
    target **= 2

    template = np.sqrt(template)
    target = np.sqrt(target)

    return np.matmul(template, target)

def zncc():
    template = np.asarray(template_image(image, 2))
    target = np.asarray(target_image(image, 2, 2))
    
    num = numerator(template, target)
    den = denominator(template, target)

    print(num)
    print(den)
    
    zncc = np.divide(num, den)
    print(zncc)

zncc()