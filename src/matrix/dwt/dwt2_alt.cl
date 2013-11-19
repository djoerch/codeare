// created on Aug 28, 2013


#pragma OPENCL EXTENSION cl_amd_printf: enable

//int printf(constant char * restrict format, ...);

//typedef float A_type;

# ifndef GROUP_SIZE_0
  # define GROUP_SIZE_0 128
# endif

# ifndef GROUP_SIZE_1
  # define GROUP_SIZE_1 128
# endif

# ifndef NUM_GROUPS_0
  # define NUM_GROUPS_0 4
# endif

# ifndef NUM_GROUPS_1
  # define NUM_GROUPS_1 4
# endif

# ifndef LDA
  # define LDA 512
# endif

# ifndef FL
  # define FL 4
# endif


void
global2local_alt        (__global A_type * arg1, __local A_type * tmp,
                     const int upper_left, const int local_c1, const int local_c2,
                     const int border_block_size_0,
                     const int border_block_size_1, __constant int * line_length)
{
  
  const int c1_base = get_group_id (0) * (border_block_size_0-offset) + local_c1 - offset;
  const int c2_base = get_group_id (1) * (border_block_size_1-offset) + local_c2 - offset;

  int j;
  for (j = 0; j < border_block_size_1-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < border_block_size_0-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left + (local_c2 + j) * LDA + local_c1 + i;
      index = index + (c1_base + i < 0 ? *line_length : 0)
                    + (c2_base + j < 0 ? *line_length * LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
            = upper_left;//arg1 [index];
    }
    if (i + local_c1 < border_block_size_0)
    {
      int index = upper_left + (local_c2 + j) * LDA + local_c1 + i;
      index = index + (c1_base + i < 0 ? *line_length : 0)
                    + (c2_base + j < 0 ? *line_length * LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
            = upper_left;//arg1 [index];
    }
  }
  if (j + local_c2 < border_block_size_1)
  {
    int i;
    for (i = 0; i < border_block_size_0-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left + (local_c2 + j) * LDA + local_c1 + i;
      index = index + (c1_base + i < 0 ? *line_length : 0)
                    + (c2_base + j < 0 ? *line_length * LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
            = 55.5;//arg1 [index];
    }
    if (i + local_c1 < border_block_size_0)
    {
      int index = upper_left + (local_c2 + j) * LDA + local_c1 + i;
      index = index + (c1_base + i < 0 ? *line_length : 0)
                    + (c2_base + j < 0 ? *line_length * LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
            = 66.6;//arg1 [index];
    }
  }
}


void
local2global_alt        (__local A_type * tmp2, __global A_type * arg2,
                     const int upper_left2, const int local_c1, const int local_c2,
                     const int block_size_0, const int block_size_1,
                     __constant int * line_length)
{
 
  const int shift = - offset/2;

  const int c1_base = get_group_id (0) * block_size_0/2 + local_c1 + shift;
  const int c2_base = get_group_id (1) * block_size_1/2 + local_c2 + shift;

  ///////////
  // part: LL
  ///////////
  int j;
  for (j = 0; j < block_size_1/2-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left2 + (local_c2 + j + shift) * LDA + local_c1 + i + shift;
      index = index + (c1_base + i < 0 ? *line_length/2 : 0)
                    + (c2_base + j < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
    }
    if (i + local_c1 < block_size_0/2)
    {
      int index = upper_left2 + (local_c2 + j + shift) * LDA + local_c1 + i + shift;
      index = index + (c1_base + i < 0 ? *line_length/2 : 0)
                    + (c2_base + j < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
    }
  }
  if (j + local_c2 < block_size_1/2)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left2 + (local_c2 + j + shift) * LDA + local_c1 + i + shift;
      index = index + (c1_base + i < 0 ? *line_length/2 : 0)
                    + (c2_base + j < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
    }
    if (i + local_c1 < block_size_0/2)
    {
      int index = upper_left2 + (local_c2 + j + shift) * LDA + local_c1 + i + shift;
      index = index + (c1_base + i < 0 ? *line_length/2 : 0)
                    + (c2_base + j < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
    }
  }

  ///////////
  // part: LH
  ///////////
  for (j = block_size_1/2; j < block_size_1-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left2 + (*line_length/2 + local_c2 + j - block_size_1/2 ) * LDA + local_c1 + i + shift;
      index = index + (c1_base + i < 0 ? *line_length/2 : 0)
                    + (c2_base + j -shift- block_size_1/2 < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
    }
    if (i + local_c1 < block_size_0/2)
    {
      int index = upper_left2 + (*line_length/2 + local_c2 + j - block_size_1/2) * LDA + local_c1 + i + shift;
      index = index + (c1_base + i < 0 ? *line_length/2 : 0)
                    + (c2_base + j -shift- block_size_1/2 < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
    }
  }
  if (j + local_c2 < block_size_1)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left2 + (*line_length/2 + local_c2 + j - block_size_1/2) * LDA + local_c1 + i + shift;
      index = index + (c1_base + i < 0 ? *line_length/2 : 0)
                    + (c2_base + j -shift- block_size_1/2 < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
    }
    if (i + local_c1 < block_size_0/2)
    {
      int index = upper_left2 + (*line_length/2 + local_c2 + j - block_size_1/2) * LDA + local_c1 + i + shift;
      index = index + (c1_base + i < 0 ? *line_length/2 : 0)
                    + (c2_base + j -shift - block_size_1/2 < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
    }
  }

  ///////////
  // part: HL
  ///////////
  for (j = 0; j < block_size_1/2-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left2 + (local_c2 + j + shift) * LDA + local_c1 + i + *line_length/2;
      index = index + (c1_base + i -shift < 0 ? *line_length/2 : 0)
                    + (c2_base + j < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i + block_size_0/2];
    }
    if (i + local_c1 < block_size_0/2)
    {
      int index = upper_left2 + (local_c2 + j + shift) * LDA + local_c1 + i + *line_length/2;
      index = index + (c1_base + i -shift < 0 ? *line_length/2 : 0)
                    + (c2_base + j < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i + block_size_0/2];
    }
  }
  if (j + local_c2 < block_size_1/2)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left2 + (local_c2 + j + shift) * LDA + local_c1 + i + *line_length/2;
      index = index + (c1_base + i -shift < 0 ? *line_length/2 : 0)
                    + (c2_base + j < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i + block_size_0/2];
    }
    if (i + local_c1 < block_size_0/2)
    {
      int index = upper_left2 + (local_c2 + j + shift) * LDA + local_c1 + i + *line_length/2;
      index = index + (c1_base + i -shift < 0 ? *line_length/2 : 0)
                    + (c2_base + j < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i + block_size_0/2];
    }
  }

  ///////////
  // part: HH
  ///////////
  for (j = block_size_1/2; j < block_size_1-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left2 + (*line_length/2 + local_c2 + j - block_size_1/2) * LDA + local_c1 + i + *line_length/2;
      index = index + (c1_base + i -shift < 0 ? *line_length/2 : 0)
                    + (c2_base + j -shift - block_size_1/2 < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = upper_left2;//tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i + block_size_0/2];
    }
    if (i + local_c1 < block_size_0/2)
    {
      int index = upper_left2 + (*line_length/2 + local_c2 + j - block_size_1/2) * LDA + local_c1 + i + *line_length/2;
      index = index + (c1_base + i -shift < 0 ? *line_length/2 : 0)
                    + (c2_base + j -shift - block_size_1/2 < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = upper_left2;//tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i + block_size_0/2];
    }
  }
  if (j + local_c2 < block_size_1)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left2 + (*line_length/2 + local_c2 + j - block_size_1/2) * LDA + local_c1 + i + *line_length/2;
      index = index + (c1_base + i -shift < 0 ? *line_length/2 : 0)
                    + (c2_base + j -shift - block_size_1/2 < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = upper_left2;//tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i + block_size_0/2];
    }
    if (i + local_c1 < block_size_0/2)
    {
      int index = upper_left2 + (*line_length/2 + local_c2 + j - block_size_1/2) * LDA + local_c1 + i + *line_length/2;
      index = index + (c1_base -shift + i < 0 ? *line_length/2 : 0)
                    + (c2_base -shift + j - block_size_1/2 < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = upper_left2;//tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i + block_size_0/2];
    }
  }

}



/**
 * @author djoergens
 */
kernel void dwt2_alt (__global A_type * input,
          __constant A_type * _lpf,
          __constant A_type * _hpf,
          __global A_type * output,
          __local A_type * loc_mem,
          __constant int * n,
          __global int * m,
          __global int * k,
          __constant int * line_length,
          __constant int * num_slices,
          __global int * loc_mem_size)
{

  // choose active threads
  if (get_global_id (0) < *line_length
    && get_global_id (1) < *line_length
    && get_global_id (2) < *line_length)
  {

    const int active_threads_0 = min (*line_length, (int) get_global_size (0));
    const int active_threads_1 = min (*line_length, (int) get_global_size (1));
    
    const int num_tiles_0 = *line_length / active_threads_0;
    const int num_tiles_1 = *line_length / active_threads_1;

    const int block_size_0 = 2*GROUP_SIZE_0;
    const int block_size_1 = 2*GROUP_SIZE_1;
    const int border_block_size_0 = block_size_0 + offset;
    const int border_block_size_1 = block_size_1 + offset;

    const int local_c1 = get_local_id (0);
    const int local_c2 = get_local_id (1);

    __local A_type * tmp  = & loc_mem [get_local_id (2) * (border_block_size_0 * border_block_size_1 + block_size_0 * border_block_size_1)];
    __local A_type * tmp2 = & loc_mem [get_local_id (2) * (border_block_size_0 * border_block_size_1 + block_size_0 * border_block_size_1)
                                       + border_block_size_0 * border_block_size_1];

    for (int tile_0 = 0; tile_0 < num_tiles_0; tile_0 ++)
      for (int tile_1 = 0; tile_1 < num_tiles_1; tile_1 ++)
    {

    const int upper_left =
                           tile_0 * active_threads_0
                         + tile_1 * active_threads_1 * LDA
                         + get_group_id (1) * block_size_1 * LDA
                         + get_group_id (0) * block_size_0
                         - offset * LDA
                         - offset;
    const int upper_left2 =
                            tile_0 * active_threads_0
                          + tile_1 * active_threads_1 * LDA
                          + get_group_id (1) * block_size_1 / 2 * LDA
                          + get_group_id (0) * block_size_0 / 2;

    for (int slice = get_group_id (2) * get_local_size (2); slice < *num_slices; slice += get_global_size (2))
    {

      // update start address of current slice
      __global A_type * arg1 = input + slice * LDA * LDB;
      __global A_type * arg2 = output + slice * LDA * LDB;

      barrier (CLK_LOCAL_MEM_FENCE);

      /////////////////////////////
      // load block to local memory
      /////////////////////////////
      global2local_alt (arg1, tmp, upper_left, local_c1, local_c2, border_block_size_0, border_block_size_1, line_length);

      barrier (CLK_LOCAL_MEM_FENCE); // local mem fence since work is performed on local memory !!!

      // filter operations
//      filter_columns (local_c1, local_c2, tmp, tmp2, _lpf, _hpf, block_size_0, block_size_1, border_block_size_0, border_block_size_1);

      barrier (CLK_LOCAL_MEM_FENCE);

//      filter_rows (local_c1, local_c2, tmp2, tmp, _lpf, _hpf, block_size_0, block_size_1);

      barrier (CLK_LOCAL_MEM_FENCE);

      //////////////////////////////
      // write back to global memory
      //////////////////////////////
      local2global_alt (tmp, arg2, upper_left2, local_c1, local_c2, block_size_0, block_size_1, line_length);

    } // loop over slices

  }

  } // if: choose active threads

}