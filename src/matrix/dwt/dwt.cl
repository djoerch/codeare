
/**
 * added by oclConnection:
 *    A_type, A_type_n
 *    B_type, B_type_n
 *    vec_len
 */

/**
 * @brief                 Helper function for initializing kernel parameters.
 *
 * @return                State of success.
 */
bool
init_params               ( int * index,
                              int * global_size,
                              int * global_inc,
                              int * local_position )
  {
  
    if (get_work_dim () == 1)
    {
      *index = get_local_id (0);
    }
    else
    {
      return false;
    }
   
    *global_size = get_global_size (0);
    
    *global_inc = *global_size;
    
    *local_position = *index;
  
    return true;
  
  }


# ifndef __OCL_DWT_KERNEL__

# define __OCL_DWT_KERNEL__



/**
 * @brief                 Elementwise increment vector.
 *
 * @param  arg1           Start address of first vector.
 * @param  arg2           Start address of second vector.
 * @param  filter		  Start address of convolution kernel.
 * @param  num_elems      Number of elements of first and second vector.
 * @param  fl			  Filter length.
 */
__kernel
void
dwt                       ( __global A_type_n *      arg1,
                            __global A_type   *    scalar,
                            __global      int * num_elems,
                            __global      int *        fl )
{

  int index, global_size, global_inc, local_position;
  
  /* initialize parameters */
  init_params (&index, &global_size, &global_inc, &local_position);

  /* calculation */
  for (int i = local_position; i < *num_elems / vec_len; i += global_inc)
  {
    arg2 [i] = arg1 [i];
  }
 
}


# endif __OCL_DWT_KERNEL__
