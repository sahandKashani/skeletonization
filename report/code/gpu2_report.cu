void and_reduction(uint8_t* g_data, int g_width, int g_height, dim3 grid_dim,
                   dim3 block_dim)
{
  int shared_mem_size = block_dim.x * block_dim.y * sizeof(uint8_t);

  // iterative reductions of g_data
  do
  {
    and_reduction<<<grid_dim,block_dim,shared_mem_size>>>(g_data,g_width,g_height);

    g_width = grid_dim.x;
    g_height = grid_dim.y;
    grid_dim.x = ceil(grid_dim.x / ((double) block_dim.x));
    grid_dim.y = ceil(grid_dim.y / ((double) block_dim.y));
  }
  while ((g_width * g_height) != 1);
}

// Adapted for 2D arrays from Nvidia cuda SDK samples
__global__ void and_reduction(uint8_t* g_data, int g_width, int g_height)
{
  // shared memory for tile
  extern __shared__ uint8_t s_data[];

  int g_row = blockIdx.y * blockDim.y + threadIdx.y;
  int g_col = blockIdx.x * blockDim.x + threadIdx.x;

  int tid = threadIdx.y * blockDim.x + threadIdx.x;

  // Load equality values into shared memory tile. We use 1 as the default
  // value, as it is an AND reduction
  s_data[tid] = is_outside_image(g_row, g_col, g_width, g_height)
                ? 1
                : global_mem_read(g_data, g_row, g_col, g_width, g_height);

  __syncthreads();

  // do reduction in shared memory
  uint8_t write_data = block_and_reduce(s_data);

  // write result for this block to global memory
  if (tid == 0)
  {
    global_mem_write(g_data, blockIdx.y, blockIdx.x, gridDim.x, gridDim.y,
                     write_data);
  }
}

__device__ uint8_t block_and_reduce(uint8_t* s_data)
{
  int tid = threadIdx.y * blockDim.x + threadIdx.x;

  for (int s = ((blockDim.x * blockDim.y) / 2); s > 0; s >>= 1) {
    if (tid < s) {
      s_data[tid] &= s_data[tid + s];
    }
    __syncthreads();
  }

  return s_data[0];
}

__global__ void pixel_equality(uint8_t* g_in_1, uint8_t* g_in_2, uint8_t* g_out,
                               int g_width, int g_height)
{
  int g_row = blockIdx.y * blockDim.y + threadIdx.y;
  int g_col = blockIdx.x * blockDim.x + threadIdx.x;

  uint8_t write_data = (global_mem_read(g_in_1, g_row, g_col, g_width, g_height) ==
                        global_mem_read(g_in_2, g_row, g_col, g_width, g_height));
  global_mem_write(g_out, g_row, g_col, g_width, g_height, write_data);
}

// Performs an image skeletonization algorithm on the input Bitmap, and stores
// the result in the output Bitmap.
int skeletonize(Bitmap** src_bitmap, Bitmap** dst_bitmap, dim3 grid_dim, dim3 block_dim)
{
  // allocate memory on device
  uint8_t* g_src_data = NULL;
  uint8_t* g_dst_data = NULL;
  uint8_t* g_equ_data = NULL; // used for reduction
  int g_data_size = (*src_bitmap)->width * (*src_bitmap)->height * sizeof(uint8_t);
  cudaMalloc((void**) &g_src_data, g_data_size);
  cudaMalloc((void**) &g_dst_data, g_data_size);
  cudaMalloc((void**) &g_equ_data, g_data_size); // reduction data has the same
                                                 // size as the image itself

  // send g_src_data to device
  cudaMemcpy(g_src_data, (*src_bitmap)->data, g_data_size, cudaMemcpyHostToDevice);

  // flag used for deciding whether to relaunch an iteration
  uint8_t are_identical_bitmaps = 0;

  int iterations = 0;
  do
  {
    // perform 1 iteration of the skeletonization algorithm
    skeletonize_pass<<<grid_dim, block_dim>>>(g_src_data, g_dst_data,
                                              (*src_bitmap)->width,
                                              (*src_bitmap)->height);

    // compute an equality operator between each pixel to obtain data for the
    // reduction
    pixel_equality<<<grid_dim, block_dim>>>(g_src_data, g_dst_data, g_equ_data,
                                            (*src_bitmap)->width,
                                            (*src_bitmap)->height);

    // perform iterative binary AND reduction on data
    and_reduction(g_equ_data, (*src_bitmap)->width, (*src_bitmap)->height,
                  grid_dim, block_dim);

    // bring reduced bitmap equality flag back from device
    cudaMemcpy(&are_identical_bitmaps, g_equ_data, 1 * sizeof(uint8_t),
               cudaMemcpyDeviceToHost);

    swap_bitmaps((void**) &g_src_data, (void**) &g_dst_data);

    iterations++;
  }
  while (!are_identical_bitmaps);

  // bring dst_bitmap back from device
  cudaMemcpy((*dst_bitmap)->data, g_dst_data, g_data_size, cudaMemcpyDeviceToHost);

  // free memory on device
  cudaFree(g_src_data);
  cudaFree(g_dst_data);
  cudaFree(g_equ_data);

  return iterations;
}
