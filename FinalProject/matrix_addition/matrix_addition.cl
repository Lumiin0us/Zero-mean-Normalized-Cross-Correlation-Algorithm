__kernel void matrix_addition(){
    // Get the global ID of the work item
    int global_id = get_global_id(0);
    
    int matrix_1[3][3] = {{5, 1, 2},{2, 2, 2},{1, 1, 1}};
    int matrix_2[3][3] = {{1, -1, 22},{12, 1, -3},{9, 12, 10}};
    int matrix_sum[3][3];
    
    // summing 1 and 2 in matrix_sum
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            matrix_sum[i][j] = matrix_1[i][j] + matrix_2[i][j];
        }
    }
    
    // printing matrix_sum
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            printf("%d ", matrix_sum[i][j]);
        }
        printf("\n");
    }
}