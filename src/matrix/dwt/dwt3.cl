// created on Nov 11, 2013

# ifndef GROUP_SIZE
  # define GROUP_SIZE 128
# endif

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



void
filter                   (const int local_c1,
                          __local A_type * tmp, __local A_type * tmp2,
                          __constant A_type * _lpf, __constant A_type * _hpf,
                          const int block_size,
                          const int border_block_size)
{

  const int group_size = get_local_size (2);

  // COLUMNS

  if (local_c1 < group_size/2)  // lowpass filter
  {

    // reused index parts
    const int part_index = 2 * local_c1;
    const int part_index2 = local_c1;

    // i: loop over slices per thread
    int i;
    for (i = 0; i < block_size - group_size; i += group_size)
    {
      const int index = part_index + i;
      const int index2 = part_index2 + i/2;
      tmp2 [index2] = conv_step_lo (index, _lpf, tmp, 1);
    }
    if (i + local_c1 < block_size)
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
    const int part_index = 2 * (local_c1 - group_size / 2) - 1 + start_hi;
    const int part_index2 = local_c1 - group_size / 2 + block_size / 2;

    // i: loop over slices per thread
    int i;
    for (i = 0; i < block_size - group_size; i += group_size)
    {
      const int index = part_index + i;
      const int index2 = part_index2 + i/2;
      tmp2 [index2] = conv_step_hi (index, _hpf, tmp, 1);
    }
    if (i + local_c1 - group_size / 2 < block_size)
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
kernel void dwt3 (__global A_type * arg1,
          __constant A_type * _lpf,
          __constant A_type * _hpf,
          __global A_type * arg2,
          __local A_type * loc_mem,
          __constant int * n,
          __global int * m,
          __global int * k,
          __constant int * line_length,
          __global int * loc_mem_size)
{


        if (get_global_id (2) < *line_length
            && get_global_id (1) < *line_length
            && get_global_id (0) < *line_length)
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
    
    const int tid = (lid_0 + lsize_0 * lid_1);
    
    const int active_threads_0 = min (*line_length, (int) get_global_size (0));
    const int active_threads_1 = min (*line_length, (int) get_global_size (1));
    const int active_threads_2 = min (*line_length, (int) get_global_size (2));

    const int block_size = *line_length / (active_threads_2 / get_local_size (2));
    const int border_block_size = block_size + offset;

    const int num_blocks_0 = *line_length / active_threads_0;
    const int num_blocks_1 = *line_length / active_threads_1;

    __local A_type * tmp = & loc_mem [tid * (border_block_size + block_size)];
    __local A_type * tmp2 = & loc_mem [tid * (border_block_size + block_size) + border_block_size];

    const int slice = LDA * LDB;
    const int half_bs = block_size / 2;
    
    const int shift = -offset / 2;

    const int index_base = get_group_id (0) * lsize_0 + lid_0
                         + (get_group_id (1) * lsize_1 + lid_1) * *n;
    const int thread_base_1 = lid_2 - offset;
    const int thread_base_2 = lid_2 + shift;


    for (int d0 = 0; d0 < num_blocks_0; d0 ++) // loop over global blocks in first dimension
    {
        
      for (int d1 = 0; d1 < num_blocks_1; d1 ++) // loop over global blocks in second dimension
      {

        const int base_index = index_base
                             + d0 * active_threads_0
                             + (d1 * active_threads_1) * *n;
                            
          // read: global2local
          int i;
          for (i = 0; i < border_block_size - lsize_2; i += lsize_2)
          {
            const int index = base_index + (i + thread_base_1) * slice
                            + (i + thread_base_1 < 0 ? *line_length : 0) * slice;
            tmp [lid_2 + i] = arg1 [index];
          }
          if (i + lid_2 < border_block_size)
          {
            const int index = base_index + (i + thread_base_1) * slice
                            + (i + thread_base_1 < 0 ? *line_length : 0) * slice;
            tmp [lid_2 + i] = arg1 [index];
          }


        barrier (CLK_LOCAL_MEM_FENCE);

        // perform calculation

        // choose active threads
        filter (lid_2, tmp, tmp2, _lpf, _hpf, block_size, border_block_size);

        barrier (CLK_LOCAL_MEM_FENCE);

        // write back

        // choose active threads in third dimension
        if (lid_2 < half_bs)
        {    
          
          ///////////
          // part: LOW
          ///////////
          int i;
          for (i = 0; i < half_bs - lsize_2; i += lsize_2)
          {
            const int index = base_index + (thread_base_2 + i) * slice
                            + (thread_base_2 + i < 0 ? *line_length/2 : 0) * slice;
            arg2 [index] = tmp2 [lid_2 + i];
          }
          if (i + lid_2 < half_bs)
          {
            const int index = base_index + (thread_base_2 + i) * slice
                            + (thread_base_2 + i < 0 ? *line_length/2 : 0) * slice;
            arg2 [index] = tmp2 [lid_2 + i];
          }

          ///////////
          // part: HIGH
          ///////////
          for (i = 0; i < half_bs - lsize_2; i += lsize_2)
          {
            const int index = base_index + (lid_2 + i) * slice
                            + (lid_2 + i < 0 ? *line_length : *line_length/2) * slice;
            arg2 [index] = tmp2 [lid_2 + i + half_bs];
          }
          if (i + lid_2 < half_bs)
          {
            const int index = base_index + (lid_2 + i) * slice
                            + (lid_2 + i < 0 ? *line_length : *line_length/2) * slice;
            arg2 [index] = tmp2 [lid_2 + i + half_bs];
          }
          
        } // if: choose active threads in third dimension
        
      } // loop over blocks in second dimension
    
    } // loop over blocks in first dimension
    
  } // if: choose active threads
}