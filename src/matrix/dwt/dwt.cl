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
dwt                       ( __global A_type_n *   arg1,
                            __global A_type   *    lpf,
                            __global A_type   *    hpf,
                            __global A_type_n *   arg2,
                            __global      int *      n,
                            __global      int *      m,
                            __global      int *      k,
                            __global      int *     fl )
{

  int index, global_size, global_inc, local_position;
  
  /* initialize parameters */
  init_params (&index, &global_size, &global_inc, &local_position);

  int num_elems = *n * *m * *k;
  
  /* calculation */
  for (int i = local_position; i < num_elems / vec_len; i += global_inc)
  {
    arg2 [i] = arg1 [i];
  }
   
}