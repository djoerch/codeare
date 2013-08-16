
#pragma OPENCL EXTENSION cl_amd_printf: enable

int printf(constant char * restrict format, ...);

//typedef float A_type;

# define LENGTH 512

/**
 * @brief                 3D DWT.
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
dwt(__global A_type * arg1,
        __global A_type * lpf,
        __global A_type * hpf,
        __global A_type * arg2,
        __global int * n,
        __global int * m,
        __global int * k,
        __global int * fl)
{

  const int num_groups = get_num_groups (0);
  const int group_id = get_group_id (0);

  const int local_index = get_local_id (0);
  const int local_size  = get_local_size (0);

  const int amount_each = *n / local_size;

  __local float tmp [LENGTH];
  __local float tmp2 [LENGTH];

  /* calculation */
  /* COLUMN MAJOR */
  for (int j = group_id; j < *m; j += num_groups)
  {
    
    // copy to local memory (work_group)
    for (int i = local_index * amount_each; i < (local_index + 1) * amount_each; i++)
    {
      tmp [i] = arg1 [j * *n + i];
    }
    
    barrier (CLK_LOCAL_MEM_FENCE);
    
    // work on local memory (work_group)
    float sum;
    for (int i = max (local_index * amount_each, *fl); i < (local_index + 1) * amount_each; i++)
    {
        sum = 0;
        for (int j = *fl-1; j >= 0; j--)
          sum += tmp [i-j] * hpf [j];
        tmp2 [i] = sum;        
    }
    
    barrier (CLK_LOCAL_MEM_FENCE);
    
    // copy from local memory (work_group)
    for (int i = local_index * amount_each; i < (local_index + 1) * amount_each; i++)
    {
      arg2 [j * *n + i] = tmp2 [i];
    }    
    
  }

}