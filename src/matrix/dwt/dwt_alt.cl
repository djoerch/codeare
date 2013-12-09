// created on Dec 2, 2013

# ifndef LDA
  # define LDA 512
# endif

# ifndef LDB
  # define LDB 512
# endif

# ifndef LDC
  # define LDC 512
# endif

# ifndef OFFSET
  # define OFFSET
  __constant const int offset = FL-2;
# endif



A_type
conv_step_hi        (const int index, __constant B_type * _filter, __local A_type * tmp,
                     const int increment);


A_type
conv_step_lo        (const int index, __constant B_type * _filter, __local A_type * tmp,
                     const int increment);



void
filter_alt               (const int lid, const int group_size,
                          __local A_type * tmp, __local A_type * tmp2,
                          __constant B_type * _lpf, __constant B_type * _hpf,
                          const int block_size,
                          const int border_block_size)
{

  // COLUMNS

  if (lid < group_size/2)  // lowpass filter
  {

    // reused index parts
    const int part_index = 2 * lid;
    const int part_index2 = lid;

    // i: loop over slices per thread
    int i;
    for (i = 0; i < block_size - group_size; i += group_size)
    {
      const int index = part_index + i;
      const int index2 = part_index2 + i/2;
      tmp2 [index2] = conv_step_lo (index, _lpf, tmp, 1);
    }
    if (i + lid < block_size)
    {
      const int index = part_index + i;
      const int index2 = part_index2 + i/2;
      tmp2 [index2] = conv_step_lo (index, _lpf, tmp, 1);
    }

  } // end of lowpass filter

  else  // highpass filter
  {

    const int start_hi = FL;

    // reused index parts
    const int part_index = 2 * (lid - group_size / 2) - 1 + start_hi;
    const int part_index2 = lid - group_size / 2 + block_size / 2;

    // i: loop over slices per thread
    int i;
    for (i = 0; i < block_size - group_size; i += group_size)
    {
      const int index = part_index + i;
      const int index2 = part_index2 + i/2;
      tmp2 [index2] = conv_step_hi (index, _hpf, tmp, 1);
    }
    if (i + lid - group_size / 2 < block_size)
    {
      const int index = part_index + i;
      const int index2 = part_index2 + i/2;
      tmp2 [index2] = conv_step_hi (index, _hpf, tmp, 1);          
    }

  } // end of highpass filter

}



/**
 * @author djoergens
 */
kernel void dwt_1 (__global A_type * arg1,
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


        if (get_global_id (0) < *line_length
            && get_global_id (1) < *chunk_size
            && get_global_id (2) < *chunk_size)
        {          

    const int lid_0 = get_local_id (0);
    const int lsize_0 = get_local_size (0);
    const int lid_1 = get_local_id (1);
    const int lsize_1 = get_local_size (1);
    const int lid_2 = get_local_id (2);
    const int lsize_2 = get_local_size (2);
    
    const int gid_0 = get_group_id (0);
    const int gid_1 = get_group_id (1);
    
    const int glob_0 = get_global_id (0);
    const int glob_1 = get_global_id (1);
    
    const int tid = (lid_1 + lsize_1 * lid_2);
    
    const int active_threads_0 = min (*line_length, (int) get_global_size (0));
    const int active_threads_1 = min (*line_length, (int) get_global_size (1));
    const int active_threads_2 = min (*chunk_size, (int) get_global_size (2));

    const int block_size = *line_length / (active_threads_0 / get_local_size (0));
    const int border_block_size = block_size + offset;

    const int num_blocks_1 = *line_length / active_threads_1;
    const int num_blocks_2 = *chunk_size / active_threads_2;

    __local A_type * tmp = & loc_mem [tid * (border_block_size + block_size)];
    __local A_type * tmp2 = & loc_mem [tid * (border_block_size + block_size) + border_block_size];

    const int slice = *line_length * *line_length;
    const int half_bs = block_size / 2;
    
    const int shift = -offset / 2;

    const int index_base = (get_group_id (1) * lsize_1 + lid_1) * *line_length
                         + (get_group_id (2) * lsize_2 + lid_2) * slice;
    const int thread_base_1 = lid_0 - offset;
    const int thread_base_2 = lid_0 + shift;


    for (int d1 = 0; d1 < num_blocks_1; d1 ++) // loop over global blocks in first dimension
    {
        
      for (int d2 = 0; d2 < num_blocks_2; d2 ++) // loop over global blocks in second dimension
      {

        const int base_index = index_base
                             + (d1 * active_threads_1) * *line_length
                             + (d2 * active_threads_2) * slice;
                            
          // read: global2local
          int i;
          for (i = 0; i < border_block_size - lsize_0; i += lsize_0)
          {
            const int index = base_index + (i + thread_base_1)
                            + (i + thread_base_1 < 0 ? *line_length : 0);
            tmp [lid_0 + i] = arg1 [index];
          }
          if (i + lid_0 < border_block_size)
          {
            const int index = base_index + (i + thread_base_1)
                            + (i + thread_base_1 < 0 ? *line_length : 0);
            tmp [lid_0 + i] = arg1 [index];
          }


        barrier (CLK_LOCAL_MEM_FENCE);

        // perform calculation

        // choose active threads
        filter_alt (lid_0, lsize_0, tmp, tmp2, _lpf, _hpf, block_size, border_block_size);

        barrier (CLK_LOCAL_MEM_FENCE);

        // write back

        // choose active threads in third dimension
        if (lid_0 < half_bs)
        {    
          
          ///////////
          // part: LOW
          ///////////
          int i;
          for (i = 0; i < half_bs - lsize_0; i += lsize_0)
          {
            const int index = base_index + (thread_base_2 + i)
                            + (thread_base_2 + i < 0 ? *line_length/2 : 0);
            arg2 [index] = tmp2 [lid_0 + i];
          }
          if (i + lid_0 < half_bs)
          {
            const int index = base_index + (thread_base_2 + i)
                            + (thread_base_2 + i < 0 ? *line_length/2 : 0);
            arg2 [index] = tmp2 [lid_0 + i];
          }

          ///////////
          // part: HIGH
          ///////////
          for (i = 0; i < half_bs - lsize_0; i += lsize_0)
          {
            const int index = base_index + (lid_0 + i)
                            + (lid_0 + i < 0 ? *line_length : *line_length/2);
            arg2 [index] = tmp2 [lid_0 + i + half_bs];
          }
          if (i + lid_0 < half_bs)
          {
            const int index = base_index + (lid_0 + i)
                            + (lid_0 + i < 0 ? *line_length : *line_length/2);
            arg2 [index] = tmp2 [lid_0 + i + half_bs];
          }
          
        } // if: choose active threads in third dimension
        
      } // loop over blocks in second dimension
    
    } // loop over blocks in first dimension
    
  } // if: choose active threads
}




/**
 * @author djoergens
 */
kernel void dwt_2 (__global A_type * arg1,
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


        if (get_global_id (1) < *line_length
            && get_global_id (0) < *line_length
            && get_global_id (2) < *chunk_size)
        {          

    const int lid_0 = get_local_id (0);
    const int lsize_0 = get_local_size (0);
    const int lid_1 = get_local_id (1);
    const int lsize_1 = get_local_size (1);
    const int lid_2 = get_local_id (2);
    const int lsize_2 = get_local_size (2);
    
    const int gid_0 = get_group_id (0);
    const int gid_1 = get_group_id (1);
    
    const int glob_0 = get_global_id (0);
    const int glob_1 = get_global_id (1);
    
    const int tid = (lid_0 + lsize_0 * lid_2);
    
    const int active_threads_0 = min (*line_length, (int) get_global_size (0));
    const int active_threads_1 = min (*line_length, (int) get_global_size (1));
    const int active_threads_2 = min (*chunk_size, (int) get_global_size (2));

    const int block_size = *line_length / (active_threads_1 / get_local_size (1));
    const int border_block_size = block_size + offset;

    const int num_blocks_0 = *line_length / active_threads_0;
    const int num_blocks_2 = *chunk_size / active_threads_2;

    __local A_type * tmp = & loc_mem [tid * (border_block_size + block_size)];
    __local A_type * tmp2 = & loc_mem [tid * (border_block_size + block_size) + border_block_size];

    const int slice = *line_length * *line_length;
    const int half_bs = block_size / 2;
    
    const int shift = -offset / 2;

    const int index_base = (get_group_id (0) * lsize_0 + lid_0)
                         + (get_group_id (2) * lsize_2 + lid_2) * slice;
    const int thread_base_1 = lid_1 - offset;
    const int thread_base_2 = lid_1 + shift;


    for (int d0 = 0; d0 < num_blocks_0; d0 ++) // loop over global blocks in first dimension
    {
        
      for (int d2 = 0; d2 < num_blocks_2; d2 ++) // loop over global blocks in second dimension
      {

        const int base_index = index_base
                             + (d0 * active_threads_0)
                             + (d2 * active_threads_2) * slice;
                            
          // read: global2local
          int i;
          for (i = 0; i < border_block_size - lsize_1; i += lsize_1)
          {
            const int index = base_index + (i + thread_base_1) * *line_length
                            + (i + thread_base_1 < 0 ? *line_length : 0) * *line_length;
            tmp [lid_1 + i] = arg1 [index];
          }
          if (i + lid_1 < border_block_size)
          {
            const int index = base_index + (i + thread_base_1) * *line_length
                            + (i + thread_base_1 < 0 ? *line_length : 0) * *line_length;
            tmp [lid_1 + i] = arg1 [index];
          }


        barrier (CLK_LOCAL_MEM_FENCE);

        // perform calculation

        // choose active threads
        filter_alt (lid_1, lsize_1, tmp, tmp2, _lpf, _hpf, block_size, border_block_size);

        barrier (CLK_LOCAL_MEM_FENCE);

        // write back

        // choose active threads in third dimension
        if (lid_1 < half_bs)
        {    
          
          ///////////
          // part: LOW
          ///////////
          int i;
          for (i = 0; i < half_bs - lsize_1; i += lsize_1)
          {
            const int index = base_index + (thread_base_2 + i) * *line_length
                            + (thread_base_2 + i < 0 ? *line_length/2 : 0) * *line_length;
            arg2 [index] = tmp2 [lid_1 + i];
          }
          if (i + lid_1 < half_bs)
          {
            const int index = base_index + (thread_base_2 + i) * *line_length
                            + (thread_base_2 + i < 0 ? *line_length/2 : 0) * *line_length;
            arg2 [index] = tmp2 [lid_1 + i];
          }

          ///////////
          // part: HIGH
          ///////////
          for (i = 0; i < half_bs - lsize_1; i += lsize_1)
          {
            const int index = base_index + (lid_1 + i) * *line_length
                            + (lid_1 + i < 0 ? *line_length : *line_length/2) * *line_length;
            arg2 [index] = tmp2 [lid_1 + i + half_bs];
          }
          if (i + lid_1 < half_bs)
          {
            const int index = base_index + (lid_1 + i) * *line_length
                            + (lid_1 + i < 0 ? *line_length : *line_length/2) * *line_length;
            arg2 [index] = tmp2 [lid_1 + i + half_bs];
          }
          
        } // if: choose active threads in third dimension
        
      } // loop over blocks in second dimension
    
    } // loop over blocks in first dimension
    
  } // if: choose active threads
}