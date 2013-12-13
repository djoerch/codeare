// created on Dec 10, 2013

# ifndef I_OFFSET
  __constant const int i_offset = FL-1;
  # define I_OFFSET
# endif

void
ifilter                  (const int lid, const int lsize,
                          __local A_type * tmp, __local A_type * tmp2,
                          __constant B_type * _lpf, __constant B_type * _hpf,
                          const int block_size, const int border_block_size);

/**
 * @author djoergens
 */
kernel void idwt_1 (__global A_type * arg1,
          __constant B_type * _lpf,
          __constant B_type * _hpf,
          __global A_type * arg2,
          __local A_type * loc_mem,
          __constant int * n,
          __global int * m,
          __global int * k,
          __constant int * line_length,
          __constant int * chunk_size,
          __global int * loc_mem_size)
{

  // choose active threads
  if (get_global_id (0) < *line_length
    && get_global_id (1) < *chunk_size
    && get_global_id (2) < *chunk_size)
  {

    const int active_threads_0 = min (*line_length, (int) get_global_size (0));
    const int active_threads_1 = min (*line_length, (int) get_global_size (1));
    const int active_threads_2 = min (*chunk_size, (int) get_global_size (2));

    const int block_size = *line_length / (active_threads_0 / get_local_size (0));
    const int border_block_size = block_size + 2 * i_offset;

    const int lid_0 = get_local_id (0);
    const int lid_1 = get_local_id (1);
    const int lid_2 = get_local_id (2);
    
    const int lsize_0 = get_local_size (0);
    const int lsize_1 = get_local_size (1);
    const int lsize_2 = get_local_size (2);
    
    const int slice = *line_length * *line_length;
    const int half_line_length = *line_length/2;
    const int tid = lid_1 + lsize_1 * lid_2;

    __local A_type * tmp  = & loc_mem [tid * (border_block_size + block_size)];
    __local A_type * tmp2 = & loc_mem [tid * (border_block_size + block_size) + border_block_size];

    const int num_blocks_1 = *line_length / active_threads_1;
    const int num_blocks_2 = *chunk_size / active_threads_2;


    const int index_base = (get_group_id (1) * lsize_1 + lid_1) * *line_length
                         + (get_group_id (2) * lsize_2 + lid_2) * slice;
                        

    for (int d1 = 0; d1 < num_blocks_1; d1 ++) // loop over global blocks in second dimension
    {
        
      for (int d2 = 0; d2 < num_blocks_2; d2 ++) // loop over global blocks in third dimension
      {
        
        const int base_index = index_base
                             + (d1 * active_threads_1) * *line_length
                             + (d2 * active_threads_2) * slice;
     
        barrier (CLK_LOCAL_MEM_FENCE);

        /////
        // memory transfer: global -> local
        /////
        const int i_max = block_size / 2 + i_offset;

        if (lid_2 < i_max)
        {
        
          // lowpass
          const int local_base_lo = lid_0 - i_offset;
          int i;
          for (i = 0; i < i_max - lsize_0; i += lsize_0)
          {
            const int index = base_index
                            + ((local_base_lo + i)
                            +  (local_base_lo + i < 0) * half_line_length);
            tmp [lid_0 + i] = arg1 [index];
          }
          if (i + lid_0 < i_max)
          {
            const int index = base_index
                            + ((local_base_lo + i)
                            +  (local_base_lo + i < 0) * half_line_length);
            tmp [lid_0 + i] = arg1 [index];
          }

          // highpass
          for (i = 0; i < i_max - lsize_0; i += lsize_0)
          {
            const int index = base_index
                            + ((lid_0 + half_line_length + i)
                            -  (lid_0 + i + 1 > half_line_length) * half_line_length);
            tmp [lid_0 + i + i_max] = arg1 [index];
          }
          if (i + lid_0 < i_max)
          {
            const int index = base_index
                            + ((lid_0 + half_line_length + i)
                            -  (lid_0 + i + 1 > half_line_length) * half_line_length);
            tmp [lid_0 + i + i_max] = arg1 [index];
          }
        
        }


        barrier (CLK_LOCAL_MEM_FENCE);

        /////
        // filter operations
        /////
        if (lid_0 < block_size / 2)
          ifilter (lid_0, lsize_0, tmp, tmp2, _lpf, _hpf, block_size, border_block_size);

        barrier (CLK_LOCAL_MEM_FENCE);

        /////
        // memory transfer: local -> global
        /////
        if (lid_0 < block_size)
        {
          int i;
          for (i = 0; i < block_size - lsize_0; i += lsize_0)
          {
            const int index = base_index + (lid_0 + i);
            arg2 [index] = tmp2 [lid_0 + i];
          }
          if (i + lid_0 < block_size)
          {
            const int index = base_index + (lid_0 + i);
            arg2 [index] = tmp2 [lid_0 + i];
          }
        }

      } // loop over global blocks in third dimension
    
    } // loop over global blocks in second dimension

  } // if: choose active threads

}



/**
 * @author djoergens
 */
kernel void idwt_2 (__global A_type * arg1,
          __constant B_type * _lpf,
          __constant B_type * _hpf,
          __global A_type * arg2,
          __local A_type * loc_mem,
          __constant int * n,
          __global int * m,
          __global int * k,
          __constant int * line_length,
          __constant int * chunk_size,
          __global int * loc_mem_size)
{

  // choose active threads
  if (get_global_id (0) < *line_length
    && get_global_id (1) < *line_length
    && get_global_id (2) < *chunk_size)
  {

    const int active_threads_0 = min (*line_length, (int) get_global_size (0));
    const int active_threads_1 = min (*line_length, (int) get_global_size (1));
    const int active_threads_2 = min (*chunk_size, (int) get_global_size (2));

    const int block_size = *line_length / (active_threads_1 / get_local_size (1));
    const int border_block_size = block_size + 2 * i_offset;

    const int lid_0 = get_local_id (0);
    const int lid_1 = get_local_id (1);
    const int lid_2 = get_local_id (2);
    
    const int lsize_0 = get_local_size (0);
    const int lsize_1 = get_local_size (1);
    const int lsize_2 = get_local_size (2);
    
    const int slice = *line_length * *line_length;
    const int half_line_length = *line_length/2;
    const int tid = lid_0 + lsize_0 * lid_2;

    __local A_type * tmp  = & loc_mem [tid * (border_block_size + block_size)];
    __local A_type * tmp2 = & loc_mem [tid * (border_block_size + block_size) + border_block_size];

    const int num_blocks_0 = *line_length / active_threads_0;
    const int num_blocks_2 = *chunk_size / active_threads_2;
    
    
    /////////////////////// GO ON FROM HERE ////////////////////////


    const int index_base = get_group_id (0) * lsize_0 + lid_0
                         + (get_group_id (2) * lsize_2 + lid_2) * slice;
                        

    for (int d0 = 0; d0 < num_blocks_0; d0 ++) // loop over global blocks in first dimension
    {
        
      for (int d2 = 0; d2 < num_blocks_2; d2 ++) // loop over global blocks in third dimension
      {
        
        const int base_index = index_base
                             + d0 * active_threads_0
                             + (d2 * active_threads_2) * slice;
     
        barrier (CLK_LOCAL_MEM_FENCE);

        /////
        // memory transfer: global -> local
        /////
        const int i_max = block_size / 2 + i_offset;

        if (lid_1 < i_max)
        {
        
          // lowpass
          const int local_base_lo = lid_1 - i_offset;
          int i;
          for (i = 0; i < i_max - lsize_1; i += lsize_1)
          {
            const int index = base_index
                            + ((local_base_lo + i)
                            +  (local_base_lo + i < 0) * half_line_length) * *line_length;
            tmp [lid_1 + i] = arg1 [index];
          }
          if (i + lid_1 < i_max)
          {
            const int index = base_index
                            + ((local_base_lo + i)
                            +  (local_base_lo + i < 0) * half_line_length) * *line_length;
            tmp [lid_1 + i] = arg1 [index];
          }

          // highpass
          for (i = 0; i < i_max - lsize_1; i += lsize_1)
          {
            const int index = base_index
                            + ((lid_1 + half_line_length + i)
                            -  (lid_1 + i + 1 > half_line_length) * half_line_length) * *line_length;
            tmp [lid_1 + i + i_max] = arg1 [index];
          }
          if (i + lid_1 < i_max)
          {
            const int index = base_index
                            + ((lid_1 + half_line_length + i)
                            -  (lid_1 + i + 1 > half_line_length) * half_line_length) * *line_length;
            tmp [lid_1 + i + i_max] = arg1 [index];
          }
        
        }


        barrier (CLK_LOCAL_MEM_FENCE);

        /////
        // filter operations
        /////
        if (lid_1 < block_size / 2)
          ifilter (lid_1, lsize_1, tmp, tmp2, _lpf, _hpf, block_size, border_block_size);

        barrier (CLK_LOCAL_MEM_FENCE);

        /////
        // memory transfer: local -> global
        /////
        if (lid_1 < block_size)
        {
          int i;
          for (i = 0; i < block_size - lsize_1; i += lsize_1)
          {
            const int index = base_index + (lid_1 + i) * *line_length;
            arg2 [index] = tmp2 [lid_1 + i];
          }
          if (i + lid_1 < block_size)
          {
            const int index = base_index + (lid_1 + i) * *line_length;
            arg2 [index] = tmp2 [lid_1 + i];
          }
        }

      } // loop over global blocks in third dimension
    
    } // loop over global blocks in first dimension

  } // if: choose active threads

}