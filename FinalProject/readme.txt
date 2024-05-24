--------------------FINAL PROJECT SUBMISSION---------------------------

---- Folder descriptions and commands that I used to run for Mac -----

- Python_implementation_results and cpp_results: These folders contain python implementation and               c++ / python experimentations and output images along with parameters.

- hello_world || image_processing || matrix_addition: These are all the sub-tasks from the guide

openCL_files -> These contain the host file for running kernels along with kernel files

zncc.cpp -> The zncc algorithm implementation
zncc_with_threads -> threading using thread header file
zncc_openMP -> parallelizing zncc

Commands that I used to run: 
----------------------------
Normal Files with OpenCV -> clang++ --std=c++11 zncc.cpp  -o zncc`pkg-config --cflags --libs opencv4`

OpenMP -> clang++ --std=c++11 -Xpreprocessor -fopenmp zncc3_openMP.cpp -o zncc3_openMP -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp `pkg-config --cflags --libs opencv4`

OpenCL -> clang++ --std=c++11 host.cpp -o host_test -framework OpenCL `pkg-config --cflags --libs opencv4`
