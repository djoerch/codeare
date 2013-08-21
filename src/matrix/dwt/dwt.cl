
#pragma OPENCL EXTENSION cl_amd_printf: enable

//int printf(constant char * restrict format, ...);

//typedef float A_type;

# define LENGTH 512

/**
 * @brief                 One DWT step for columns.
 *
 * @param  arg1           Start address of signal vector.
 * @param   lpf           Start address of lowpass convolution kernel.
 * @param   hpf           Start address of highpass convolution kernel.
 * @param  arg2           Start address of result vector.
 * @param  n              Length along first dimension.
 * @param  m              Length along second dimension.
 * @param  k              Length along third dimension.
 * @param  fl             Filter length.
 */
__kernel
void
dwt_cols (__global A_type * arg1,
          __constant A_type * _lpf,
          __constant A_type * _hpf,
          __global A_type * arg2,
          __local A_type * loc_mem,
          __constant int * n,
          __global int * m,
          __global int * k,
          __global int * fl,
          __global int * loc_mem_size)
{

//  const int num_groups = get_num_groups (0);
//  const int group_id = get_group_id (0);

  const int local_c1     = get_local_id (0);     // current row
  const int local_size1  = get_local_size (0);   // number of rows

  const int global_c1    = get_global_id (0);   // current row
//  const int global_size1 = get_global_size (0); // number of rows
  const int global_c2    = get_global_id (1);   // current column
//  const int global_size2 = get_global_size (1); // number of columns

  __local A_type * tmp  = & loc_mem [0];
  __local A_type * tmp2 = & loc_mem [*n];

  const int column_offset = global_c2 * *m;


  /* calculation */
  /* COLUMN MAJOR */

  // COLUMNS

  // copy column to local memory
  tmp [local_c1] = arg1 [column_offset + global_c1];
  
  A_type sum = 0;
  barrier (CLK_LOCAL_MEM_FENCE);

  // partitioning for highpass and lowpass filtering
  if (local_c1 < *n/2)
  {
    // work on local memory (work_group)
    // lowpass & highpass
    # pragma unroll
    for (int j = *fl-1; j >= 0; j--)
    {
      sum += tmp [(2*local_c1-j)&(*n-1)] * _lpf [j];
    }
    tmp2 [local_c1] = sum;
  }
  else
  {
    // work on local memory (work_group)
    // lowpass & highpass
    # pragma unroll
    for (int j = *fl-1; j >= 0; j--)
    {
      sum += tmp [(2*(local_c1-*n/2)+1-j)&(*n-1)] * _hpf [j];
    }
    tmp2 [local_c1] = sum;
  }



//  barrier (CLK_LOCAL_MEM_FENCE);


  // copy column back to global memory
  arg2 [column_offset + global_c1] = tmp2 [local_c1];

}


/**
 * @brief                 One DWT step for rows.
 *
 * @param  arg1           Start address of signal vector.
 * @param   lpf           Start address of lowpass convolution kernel.
 * @param   hpf           Start address of highpass convolution kernel.
 * @param  arg2           Start address of result vector.
 * @param  n              Length along first dimension.
 * @param  m              Length along second dimension.
 * @param  k              Length along third dimension.
 * @param  fl             Filter length.
 */
__kernel
void
dwt_rows (__global A_type * arg1,
          __constant A_type * _lpf,
          __constant A_type * _hpf,
          __global A_type * arg2,
          __local A_type * loc_mem,
          __constant int * n,
          __global int * m,
          __global int * k,
          __global int * fl,
          __global int * loc_mem_size)
{

//  const int num_groups = get_num_groups (0);
//  const int group_id = get_group_id (0);

  const int local_c1     = get_local_id (0);     // current row
  const int local_size1  = get_local_size (0);   // number of rows

  const int global_c1    = get_global_id (0);   // current row
//  const int global_size1 = get_global_size (0); // number of rows
  const int global_c2    = get_global_id (1);   // current column
//  const int global_size2 = get_global_size (1); // number of columns

  __local A_type * tmp  = & loc_mem [0];
  __local A_type * tmp2 = & loc_mem [*n];


  /* calculation */
  /* COLUMN MAJOR */


  // ROWS

  // copy row to local memory
  tmp [local_c1] = arg2 [global_c2 + *m * local_c1];
  
  
  A_type sum = 0;
  barrier (CLK_LOCAL_MEM_FENCE);

  // partitioning for highpass and lowpass filtering
  if (local_c1 < *n/2)
  {
    // work on local memory (work_group)
    // lowpass & highpass
    # pragma unroll
    for (int j = *fl-1; j >= 0; j--)
    {
      sum += tmp [(2*local_c1-j)&(*n-1)] * _lpf [j];
    }
    tmp2 [local_c1] = sum;
  }
  else
  {
    // work on local memory (work_group)
    // lowpass & highpass
    # pragma unroll
    for (int j = *fl-1; j >= 0; j--)
    {
      sum += tmp [(2*(local_c1-*n/2)+1-j)&(*n-1)] * _hpf [j];
    }
    tmp2 [local_c1] = sum;
  }

//  barrier (CLK_LOCAL_MEM_FENCE);


  // copy column back to global memory
  arg2 [global_c2 + local_c1 * *m] = tmp2 [local_c1];

}