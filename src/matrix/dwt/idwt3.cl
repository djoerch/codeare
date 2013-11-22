// created on Nov 20, 2013



void
ifilter                  (const int local_c1,
                          __local A_type * tmp, __local A_type * tmp2,
                          __constant A_type * _lpf, __constant A_type * _hpf,
                          const int block_size, const int border_block_size)
{

  const int i_max = block_size / 2;

  const int lsize_2 = get_local_size (2);

  // COLS //

  const int start_lo = FL-1;

  const int local_base_1 = local_c1 + start_lo;
  const int local_base_2 = 2 * local_c1;


    int i;
    for (i = 0; i < i_max - lsize_2; i += lsize_2)
    {
        const int index1 = local_base_1 + i;
        const int index2 = local_base_2 + 2 * i;
        iconv_step1_lo (index1, index2, _lpf, tmp, tmp2, 1);
        iconv_step2_hi (index1 + i_max, index2, _hpf, tmp, tmp2, 1);
    }
    if (i + local_c1 < i_max)
    {
        const int index1 = local_base_1 + i;
        const int index2 = local_base_2 + 2 * i;
        iconv_step1_lo (index1, index2, _lpf, tmp, tmp2, 1);
        iconv_step2_hi (index1 + i_max, index2, _hpf, tmp, tmp2, 1);
    }

//    for (i = 0; i < i_max - lsize_2; i += lsize_2)
//    {
//        const int index1 = local_base_1 + i + i_max;
//        const int index2 = local_base_2 + 2 * i;
//        iconv_step2_hi (index1, index2, _hpf, tmp, tmp2, 1);
//    }
//    if (i + local_c1 < i_max)
//    {
//        const int index1 = local_base_1 + i + i_max;
//        const int index2 = local_base_2 + 2 * i;
//        iconv_step2_hi (index1, index2, _hpf, tmp, tmp2, 1);
//    }

}



/**
 * @author djoergens
 */
kernel void idwt3 (__global A_type * arg1,
          __constant A_type * _lpf,
          __constant A_type * _hpf,
          __global A_type * arg2,
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
    const int active_threads_2 = min (*line_length, (int) get_global_size (2));

    const int block_size = *line_length / (active_threads_2 / get_local_size (2));
    const int border_block_size = block_size + 2 * i_offset;

    const int lid_0 = get_local_id (0);
    const int lid_1 = get_local_id (1);
    const int lid_2 = get_local_id (2);
    
    const int lsize_0 = get_local_size (0);
    const int lsize_1 = get_local_size (1);
    const int lsize_2 = get_local_size (2);
    
    const int slice = LDA * LDB;
    const int tid = lid_0 + lsize_0 * lid_1;

    __local A_type * tmp  = & loc_mem [tid * (border_block_size + block_size)];
    __local A_type * tmp2 = & loc_mem [tid * (border_block_size + block_size) + border_block_size];

    const int num_blocks_0 = *line_length / active_threads_0;
    const int num_blocks_1 = *line_length / active_threads_1;


    const int index_base = get_group_id (0) * lsize_0 + lid_0
                         + (get_group_id (1) * lsize_1 + lid_1) * *n;
                        

    for (int d0 = 0; d0 < num_blocks_0; d0 ++) // loop over global blocks in first dimension
    {
        
      for (int d1 = 0; d1 < num_blocks_1; d1 ++) // loop over global blocks in second dimension
      {
        
        const int base_index = index_base
                             + d0 * active_threads_0
                             + (d1 * active_threads_1) * *n;
     
        barrier (CLK_LOCAL_MEM_FENCE);

        /////
        // memory transfer: global -> local
        /////
        const int i_max = block_size / 2 + i_offset;

        if (lid_2 < i_max)
        {
        
          // lowpass
          const int shift_lo = - i_offset;
          int i;
          for (i = 0; i < i_max - lsize_2; i += lsize_2)
          {
            const int index = base_index + (lid_2 + i + shift_lo) * slice
                            + (lid_2 + i + shift_lo < 0) * *line_length/2 * slice;
            tmp [lid_2 + i] = arg1 [index];
          }
          if (i + lid_2 < i_max)
          {
            const int index = base_index + (lid_2 + i + shift_lo) * slice
                            + (lid_2 + i + shift_lo < 0) * *line_length/2 * slice;
            tmp [lid_2 + i] = arg1 [index];
          }

          // highpass
          for (i = 0; i < i_max - lsize_2; i += lsize_2)
          {
            const int index = base_index + (lid_2 + *line_length/2 + i) * slice
                            - (lid_2 + i + 1 > *line_length/2) * *line_length/2 * slice;
            tmp [lid_2 + i + i_max] = arg1 [index];
          }
          if (i + lid_2 < i_max)
          {
            const int index = base_index + (lid_2 + *line_length/2 + i) * slice
                            - (lid_2 + i + 1 > *line_length/2) * *line_length/2 * slice;
            tmp [lid_2 + i + i_max] = arg1 [index];
          }
        
        }


        barrier (CLK_LOCAL_MEM_FENCE);

        /////
        // filter operations
        /////
        if (lid_2 < block_size / 2)
          ifilter (lid_2, tmp, tmp2, _lpf, _hpf, block_size, border_block_size);

        barrier (CLK_LOCAL_MEM_FENCE);

        /////
        // memory transfer: local -> global
        /////
        int i;
        for (i = 0; i < block_size - lsize_2; i += lsize_2)
        {
          const int index = base_index + (lid_2 + i) * slice;
          arg2 [index] = tmp2 [lid_2 + i];
        }
        if (i + lid_2 < block_size)
        {
          const int index = base_index + (lid_2 + i) * slice;
          arg2 [index] = tmp2 [lid_2 + i];
        }

      } // loop over global blocks in second dimension
    
    } // loop over global blocks in first dimension

  } // if: choose active threads

}