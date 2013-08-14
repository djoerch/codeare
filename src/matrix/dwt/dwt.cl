

bool
init_params_dwt           ( int * index,
                            int * global_size,
                            int * global_inc,
                            int * local_position )
{

  if (get_work_dim () == 1)
    *index = get_local_id (0);
  else
    return false;

  *global_size    = get_global_size (0);
  *global_inc     = *global_size;
  *local_position = *index;

  return true;

}


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
dwt                       ( __global A_type *   arg1,
                            __global A_type   *    lpf,
                            __global A_type   *    hpf,
                            __global A_type *   arg2,
                            __global      int *      n,
                            __global      int *      m,
                            __global      int *      k,
                            __global      int *     fl )
{

  int index, global_size, global_inc, local_position;
  
  /* initialize parameters */
  init_params_dwt (&index, &global_size, &global_inc, &local_position);

  const int num_elems = *n * *m * *k;
  
  const int chunk_size = num_elems / global_size; 
    
  /* calculation */
  /* COLUMN MAJOR */
  
  for (int i = local_position; i < num_elems; i += global_inc)
  {
    arg2 [i] = arg1 [i];
  }
   
}