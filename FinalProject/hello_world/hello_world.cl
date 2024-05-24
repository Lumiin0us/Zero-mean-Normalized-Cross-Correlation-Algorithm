__kernel void hello_world() {
    // Get the global ID of the work item
    int global_id = get_global_id(0);

    printf("Hello, World! from work item %d\n", global_id);
}