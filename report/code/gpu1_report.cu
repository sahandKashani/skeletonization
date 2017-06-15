// Performs an image skeletonization algorithm on the input Bitmap, and stores
// the result in the output Bitmap.
int skeletonize(Bitmap** src_bitmap, Bitmap** dst_bitmap, dim3 grid_dim,
                dim3 block_dim)
{
  // allocate memory on device
  uint8_t* g_src_data = NULL;
  uint8_t* g_dst_data = NULL;
  int g_data_size = (*src_bitmap)->width * (*src_bitmap)->height * sizeof(uint8_t);
  cudaMalloc((void**) &g_src_data, g_data_size);
  cudaMalloc((void**) &g_dst_data, g_data_size);

  // send g_src_data to device
  cudaMemcpy(g_src_data, (*src_bitmap)->data, g_data_size, cudaMemcpyHostToDevice);

  int iterations = 0;
  do
  {
    // perform 1 iteration of the skeletonization algorithm
    skeletonize_pass<<<grid_dim, block_dim>>>(g_src_data, g_dst_data,
                                              (*src_bitmap)->width,
                                              (*src_bitmap)->height);

    // bring g_src_data and g_dst_data back from device
    cudaMemcpy((*src_bitmap)->data,g_src_data,g_data_size,cudaMemcpyDeviceToHost);
    cudaMemcpy((*dst_bitmap)->data,g_dst_data,g_data_size,cudaMemcpyDeviceToHost);

    // swap g_src_data and g_dst_data pointers
    swap_bitmaps((void**) &g_src_data, (void**) &g_dst_data);

    iterations++;
  }
  // check if the images are the same on the host
  while (!are_identical_bitmaps(*src_bitmap, *dst_bitmap));

  // free memory on device
  cudaFree(g_src_data);
  cudaFree(g_dst_data);

  return iterations;
}

// Performs 1 iteration of the thinning algorithm.
__global__ void skeletonize_pass(uint8_t* g_src, uint8_t* g_dst, int g_width,
                                 int g_height)
{
  int g_row = blockIdx.y * blockDim.y + threadIdx.y;
  int g_col = blockIdx.x * blockDim.x + threadIdx.x;

  uint8_t NZ = black_neighbors_around(g_src, g_row, g_col, g_width, g_height);
  uint8_t TR_P1 = wb_transitions_around(g_src, g_row, g_col, g_width, g_height);
  uint8_t TR_P2 = wb_transitions_around(g_src, g_row-1, g_col, g_width, g_height);
  uint8_t TR_P4 = wb_transitions_around(g_src, g_row, g_col-1, g_width, g_height);
  uint8_t P2 = P2_f(g_src, g_row, g_col, g_width, g_height);
  uint8_t P4 = P4_f(g_src, g_row, g_col, g_width, g_height);
  uint8_t P6 = P6_f(g_src, g_row, g_col, g_width, g_height);
  uint8_t P8 = P8_f(g_src, g_row, g_col, g_width, g_height);

  uint8_t thinning_cond_1 = ((2 <= NZ) & (NZ <= 6));
  uint8_t thinning_cond_2 = (TR_P1 == 1);
  uint8_t thinning_cond_3 = (((P2 & P4 & P8) == 0) | (TR_P2 != 1));
  uint8_t thinning_cond_4 = (((P2 & P4 & P6) == 0) | (TR_P4 != 1));
  uint8_t thinning_cond_ok = thinning_cond_1 & thinning_cond_2 &
                             thinning_cond_3 & thinning_cond_4;

  uint8_t write_data = BINARY_WHITE + ((1 - thinning_cond_ok) *
                       global_mem_read(g_src, g_row, g_col, g_width, g_height));

  global_mem_write(g_dst, g_row, g_col, g_width, g_height, write_data);
}

// function to safely reaad from global memory (avoiding borders)
__device__ uint8_t global_mem_read(uint8_t* g_data, int g_row, int g_col,
                                   int g_width, int g_height)
{
  return is_outside_image(g_row, g_col, g_width, g_height)
                                              ? BINARY_WHITE
                                              : g_data[g_row * g_width + g_col];
}

// function to safely write to global memory (avoiding borders)
__device__ void global_mem_write(uint8_t* g_data, int g_row, int g_col,
                                 int g_width, int g_height, uint8_t write_data)
{
  if (!is_outside_image(g_row, g_col, g_width, g_height)) {
      g_data[g_row * g_width + g_col] = write_data;
  }
}
