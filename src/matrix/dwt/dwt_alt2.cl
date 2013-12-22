// created on Dec 19, 2013

# ifndef LOC_MEM_LINE
 # define LOC_MEM_LINE 32
# endif

# ifndef OFFSET
  __constant const int offset = FL-2;
  # define OFFSET
# endif


/**
 * @author djoergens
 */
kernel void dwt_1_alt (__global A_type * arg1,
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
  
  const int shift = -offset/2;

  const int padding = offset;

  ///////////////////
  // READING CONFIG
  ///////////////////
  const int pad_line_length = *line_length + padding;
  const int pad_slice = pad_line_length * *line_length;

  ///////////////////
  // WRITE CONFIG
  ///////////////////
  const int slice = *line_length * *line_length;

  ///////////////
  // THREAD
  ///////////////
  const int lid_0 = get_local_id (0);
  const int lid_1 = get_local_id (1);
  const int lid_2 = get_local_id (2);

  ///////////////
  // WORK GROUP
  ///////////////
  const int gid_0 = get_group_id (0);
  const int gid_1 = get_group_id (1);
  const int gid_2 = get_group_id (2);
  const int lsize_0 = get_local_size (0);
  const int lsize_1 = get_local_size (1);
  const int lsize_2 = get_local_size (2);

  //////////////
  // GLOBAL
  //////////////
  const int ng_0 = get_num_groups (0);
  const int gsize_1 = get_global_size (1);
  const int gsize_2 = get_global_size (2);

  //////////////////
  // LOCAL MEMORY
  //////////////////

  const int loc_mem_line_IN = LOC_MEM_LINE + 1;
  const int loc_mem_line_OUT = LOC_MEM_LINE - offset + 1;

  __local A_type * input = loc_mem;
  __local A_type * output = loc_mem + get_local_size (0) * get_local_size (1) * get_local_size (2) * loc_mem_line_IN;

  //////////////////////
  // ALGORITHM
  //////////////////////

  // loop over blocks in third dimension
  for (int d2 = gid_2 * lsize_2; d2 + lid_2 < *chunk_size; d2 += gsize_2)
  {

    barrier (CLK_LOCAL_MEM_FENCE);

    // loop over blocks in second dimension
    for (int d1 = gid_1 * lsize_1; d1 + lid_1 < *line_length; d1 += gsize_1)
    {

      barrier (CLK_LOCAL_MEM_FENCE);

      // loop over blocks in first dimension
      for (int d0 = gid_0 * (LOC_MEM_LINE - offset); d0 + lid_0 < *line_length; d0 += ng_0 * (LOC_MEM_LINE - offset))
      {

        //////////////
        // READ: global -> local
        //////////////

        const int pad_index_base = (d2 + lid_2) * pad_slice                   // slice
                                 + (d1 + lid_1) * (*line_length + padding);   // line in particular slice

        // local memory
        const int line = get_local_id (1) + get_local_id (2) * get_local_size (1);
        __local A_type * tmp = input + line * loc_mem_line_IN;

        // copy
        const int index = pad_index_base + d0 + lid_0 - offset + padding
                        + (d0 + lid_0 - offset < 0 ? *line_length : 0);
        tmp [lid_0]           = arg1 [index];
        tmp [lid_0 + lsize_0] = arg1 [index + lsize_0];

        //////////////
        // WRITE: local -> global
        //////////////
        
        const int index_base = (d2 + lid_2) * pad_slice //slice
                             + (d1 + lid_1) * pad_line_length;//*line_length;
        
        if (lid_0 < (LOC_MEM_LINE - offset)/2)
        {
          const int index2 = index_base + d0 + lid_0 + padding;
          arg2 [index2] = tmp[lid_0 + offset];
        if (d0 + lid_0 + (LOC_MEM_LINE-offset)/2 < *line_length)
          arg2 [index2 + (LOC_MEM_LINE - offset)/2] = tmp[lid_0 + offset + (LOC_MEM_LINE-offset)/2];
        }
      } // loop over first dimension

    } // loop over second dimension

  } // loop over third dimension

}


/**
 * @author djoergens
 */
kernel void dwt_2_alt (__global A_type * arg1,
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
  
  const int shift = -offset/2;

  const int padding = offset;

  ///////////////////
  // READING CONFIG
  ///////////////////
  const int pad_line_length = *line_length + padding;
  const int pad_slice = pad_line_length * *line_length;

  ///////////////////
  // WRITE CONFIG
  ///////////////////
  const int slice = *line_length * *line_length;

  ///////////////
  // THREAD
  ///////////////
  const int lid_0 = get_local_id (0);
  const int lid_1 = get_local_id (1);
  const int lid_2 = get_local_id (2);

  ///////////////
  // WORK GROUP
  ///////////////
  const int gid_0 = get_group_id (0);
  const int gid_1 = get_group_id (1);
  const int gid_2 = get_group_id (2);
  const int lsize_0 = get_local_size (0);
  const int lsize_1 = get_local_size (1);
  const int lsize_2 = get_local_size (2);

  //////////////
  // GLOBAL
  //////////////
  const int ng_1 = get_num_groups (1);
  const int gsize_0 = get_global_size (0);
  const int gsize_2 = get_global_size (2);

  //////////////////
  // LOCAL MEMORY
  //////////////////

  const int loc_mem_line_IN = LOC_MEM_LINE + 1;
  const int loc_mem_line_OUT = LOC_MEM_LINE - offset + 1;

  __local A_type * input = loc_mem;
  __local A_type * output = loc_mem + get_local_size (0) * get_local_size (1) * get_local_size (2) * loc_mem_line_IN;

  //////////////////////
  // ALGORITHM
  //////////////////////

  // loop over blocks in third dimension
  for (int d2 = gid_2 * lsize_2; d2 + lid_2 < *chunk_size; d2 += gsize_2)
  {

    barrier (CLK_LOCAL_MEM_FENCE);

    // loop over blocks in second dimension
    for (int d0 = gid_0 * lsize_0; d0 + lid_0 < *line_length; d0 += gsize_0)
    {

      barrier (CLK_LOCAL_MEM_FENCE);

      // loop over blocks in first dimension
      for (int d1 = gid_1 * (LOC_MEM_LINE - offset); d1 + lid_1 < *line_length; d1 += ng_1 * (LOC_MEM_LINE - offset))
      {

        barrier (CLK_LOCAL_MEM_FENCE);

        //////////////
        // READ: global -> local
        //////////////

        const int pad_index_base = (d2 + lid_2) * pad_slice  // slice
                                 + (d0 + lid_0) + padding;   // line in particular slice

        // local memory
        const int line = get_local_id (0) + get_local_id (2) * get_local_size (0);
        __local A_type * tmp = input + line * loc_mem_line_IN;

        // copy
        const int index = pad_index_base + (d1 + lid_1 - offset) * pad_line_length
                        + (d1 + lid_1 - offset < 0 ? pad_slice : 0);
        tmp [lid_1]           = arg1 [index];
        tmp [lid_1 + lsize_1] = arg1 [index + lsize_1 * pad_line_length];

        barrier (CLK_LOCAL_MEM_FENCE);

        //////////////
        // WRITE: local -> global
        //////////////
        
        const int index_base = (d2 + lid_2) * slice
                             + (d0 + lid_0);
        
        if (lid_1 < (LOC_MEM_LINE - offset)/2)
        {
          const int index2 = index_base + (d1 + lid_1) * *line_length;
          arg2 [index2] = tmp [lid_1 + offset];
        if (d1 + lid_1 + (LOC_MEM_LINE-offset)/2 < *line_length)
          arg2 [index2 + (LOC_MEM_LINE - offset)/2 * *line_length] = tmp [lid_1 + offset + (LOC_MEM_LINE-offset)/2];
        }
      } // loop over first dimension

    } // loop over second dimension

  } // loop over third dimension

}



/**
 * @author djoergens
 */
kernel void dwt_3_alt (__global A_type * arg1,
          __constant B_type * _lpf,
          __constant B_type * _hpf,
          __global A_type * arg2,
          __local A_type * loc_mem,
          __constant int * n,
          __global int * m,
          __global int * k,
          __constant int * line_length,
          __constant int * chunk_size_0,
          __constant int * chunk_size_1,
          __global int * loc_mem_size)
{
  
  const int shift = -offset/2;

  const int padding = offset;

  ///////////////////
  // READING CONFIG
  ///////////////////
  const int pad_line_length = *chunk_size_0 + padding;
  const int pad_slice = pad_line_length * *chunk_size_1;

  ///////////////////
  // WRITE CONFIG
  ///////////////////
  const int slice = *chunk_size_0 * *chunk_size_1;

  ///////////////
  // THREAD
  ///////////////
  const int lid_0 = get_local_id (0);
  const int lid_1 = get_local_id (1);
  const int lid_2 = get_local_id (2);

  ///////////////
  // WORK GROUP
  ///////////////
  const int gid_0 = get_group_id (0);
  const int gid_1 = get_group_id (1);
  const int gid_2 = get_group_id (2);
  const int lsize_0 = get_local_size (0);
  const int lsize_1 = get_local_size (1);
  const int lsize_2 = get_local_size (2);

  //////////////
  // GLOBAL
  //////////////
  const int ng_2 = get_num_groups (2);
  const int gsize_0 = get_global_size (0);
  const int gsize_1 = get_global_size (1);

  //////////////////
  // LOCAL MEMORY
  //////////////////

  const int loc_mem_line_IN = LOC_MEM_LINE + 1;
  const int loc_mem_line_OUT = LOC_MEM_LINE - offset + 1;

  __local A_type * input = loc_mem;
  __local A_type * output = loc_mem + get_local_size (0) * get_local_size (1) * get_local_size (2) * loc_mem_line_IN;

  //////////////////////
  // ALGORITHM
  //////////////////////

  // loop over blocks in third dimension
  for (int d1 = gid_1 * lsize_1; d1 + lid_1 < *chunk_size_1; d1 += gsize_1)
  {

    barrier (CLK_LOCAL_MEM_FENCE);

    // loop over blocks in second dimension
    for (int d0 = gid_0 * lsize_0; d0 + lid_0 < *chunk_size_0; d0 += gsize_0)
    {

      barrier (CLK_LOCAL_MEM_FENCE);

      // loop over blocks in first dimension
      for (int d2 = gid_2 * (LOC_MEM_LINE - offset); d2 + lid_2 < *line_length; d2 += ng_2 * (LOC_MEM_LINE - offset))
      {

        barrier (CLK_LOCAL_MEM_FENCE);

        //////////////
        // READ: global -> local
        //////////////

        const int pad_index_base = (d1 + lid_1) * pad_line_length  // ...
                                 + (d0 + lid_0) + padding;         // line in particular slice

        // local memory
        const int line = get_local_id (0) + get_local_id (1) * get_local_size (0);
        __local A_type * tmp = input + line * loc_mem_line_IN;

        // copy
        const int index1 = pad_index_base + (d2 + lid_2 - offset) * pad_slice
                         + (d2 + lid_2 - offset < 0 ? *line_length * pad_slice : 0);
        tmp [lid_2]           = arg1 [index1];
        const int index2 = pad_index_base + (d2 + lid_2 + lsize_2 - offset) * pad_slice
                         + (d2 + lid_2 + lsize_2 - offset < 0 ? *line_length * pad_slice : 0);
        if (index2 < *line_length * pad_slice)  // TODO: ???
          tmp [lid_2 + lsize_2] = arg1 [index2];

        barrier (CLK_LOCAL_MEM_FENCE);

        //////////////
        // WRITE: local -> global
        //////////////
        
        const int index_base = (d1 + lid_1) * *chunk_size_0
                             + (d0 + lid_0);
        
        if (lid_2 < (LOC_MEM_LINE - offset)/2)
        {
          const int index2 = index_base + (d2 + lid_2) * slice;
          arg2 [index2] = tmp [lid_2 + offset];
        if (d2 + lid_2 + (LOC_MEM_LINE-offset)/2 < *line_length)
          arg2 [index2 + (LOC_MEM_LINE - offset)/2 * slice] = tmp [lid_2 + offset + (LOC_MEM_LINE-offset)/2];
        }
      } // loop over first dimension

    } // loop over second dimension

  } // loop over third dimension

}