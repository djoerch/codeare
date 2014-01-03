// created on Dec 19, 2013

# ifndef OFFSET
  __constant const int offset = FL-2;
  # define OFFSET
# endif

# ifndef LOC_MEM_LINE
 # define LOC_MEM_LINE 32
# endif

# ifndef ROUND_TO
 # define ROUND_TO 32
# endif

# ifndef CHECKS
// # define CHECKS
# endif

int roundUp(const int numToRound, const int multiple) 
{ 
 if(multiple == 0) 
 { 
  return numToRound; 
 } 

 int remainder = numToRound % multiple;
 if (remainder == 0)
  return numToRound;
 return numToRound + multiple - remainder;
} 

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
  const int pad_line_length = roundUp (*line_length + padding, ROUND_TO);
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

  const int loc_mem_line_IN = LOC_MEM_LINE + offset + 1;
  const int loc_mem_line_OUT = LOC_MEM_LINE + 1;

  __local A_type * input = loc_mem;
  __local A_type * output = loc_mem + lsize_1 * lsize_2 * loc_mem_line_IN;

  //////////////////////
  // ALGORITHM
  //////////////////////

  // loop over blocks in third dimension
  for (int d2 = gid_2 * lsize_2; d2 < *chunk_size; d2 += gsize_2)
  {

    // loop over blocks in second dimension
    for (int d1 = gid_1 * lsize_1; d1 < *line_length; d1 += gsize_1)
    {

      // loop over blocks in first dimension
      for (int d0 = gid_0 * LOC_MEM_LINE; d0 < *line_length; d0 += ng_0 * LOC_MEM_LINE)
      {

        //////////////
        // READ: global -> local
        //////////////
        
        # ifdef CHECKS
        if (d2 + lid_2 < *chunk_size
          && d1 + lid_1 < *line_length
          && d0 + lid_0 < *line_length)
        # endif
        {

          const int pad_index_base = (d2 + lid_2) * pad_slice                   // slice
                                   + (d1 + lid_1) * pad_line_length;   // line in particular slice

          // local memory
          const int line = get_local_id (1) + get_local_id (2) * get_local_size (1);
          __local A_type * tmp = input + line * loc_mem_line_IN;

          // copy
          const int index = pad_index_base + d0 + lid_0 - offset + padding
                          + ((d0 + lid_0 - offset) < 0 ? *line_length : 0);
          tmp [lid_0]           = arg1 [index];
          if (d0 + lsize_0 + lid_0 - offset < *line_length)
            tmp [lid_0 + lsize_0] = arg1 [pad_index_base + d0 + lid_0 - offset + padding + lsize_0];
          if (d0 + 2 * lsize_0 + lid_0 - offset < *line_length && lid_0 < offset)
            tmp [lid_0 + 2 * lsize_0] = arg1 [pad_index_base + d0 + lid_0 - offset + padding + 2 * lsize_0];
        
        }

        //////////////////
        // COMPUTE: local
        //////////////////
        barrier (CLK_LOCAL_MEM_FENCE);
        
        # ifdef CHECKS
        if (d2 + lid_2 < *chunk_size
          && d1 + lid_0 < *line_length
          && d0 + lid_1 < *line_length)
        if (2 * lid_1 < LOC_MEM_LINE
         && lid_2 * lsize_0 < lsize_2 * lsize_1
         && 2 * lid_1 + d0 < *line_length)
        # endif
        {
        
          // local memory
          const int line_in = get_local_id (0) + get_local_id (2) * get_local_size(0);
          __local A_type * tmp_in = input + line_in * loc_mem_line_IN;
          __local A_type * tmp_out = output + line_in * loc_mem_line_IN;
        
          A_type sum = 0;
          # pragma unroll
          for (int k = FL-1; k >= 0; k--)
          {
            sum += tmp_in [2 * lid_1 + k] * _lpf [k];
          }
          tmp_out [lid_1] = sum;
        
          A_type sum2 = 0;
          # pragma unroll
          for (int k = FL-1; k >= 0; k--)
          {
            sum2 += tmp_in [2 * lid_1 + FL-1 - k] * _hpf [k];
          }
          tmp_out [lid_1 + LOC_MEM_LINE/2] = sum2;
        
        }
        
        //////////////
        // WRITE: local -> global
        //////////////
        barrier (CLK_LOCAL_MEM_FENCE);
        
        # ifdef CHECKS
        if (d2 + lid_2 < *chunk_size
          && d1 + lid_1 < *line_length
          && d0 + lid_0 < *line_length)
        if (2 * lid_0 + d0 < *line_length
         && 2 * lid_0 < LOC_MEM_LINE)
        # endif
        {
            
          // local memory
          const int line_out = get_local_id (1) + get_local_id (2) * get_local_size (1);
          __local A_type * tmp_out = output + line_out * loc_mem_line_IN;

          const int index_base = (d2 + lid_2) * pad_slice
                               + (d1 + lid_1) * pad_line_length;

          const int index2 = index_base + d0/2 + lid_0 + padding;
          arg2 [index2 + shift + (d0/2 + lid_0 + shift < 0 ? *line_length/2 : 0)] = tmp_out [lid_0];
          arg2 [index2 + *line_length/2] = tmp_out [lid_0 + LOC_MEM_LINE/2];

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
  const int pad_line_length = roundUp (*line_length + padding, ROUND_TO);
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

  const int loc_mem_line_IN = LOC_MEM_LINE + offset + 1;
  const int loc_mem_line_OUT = LOC_MEM_LINE + 1;

  __local A_type * input = loc_mem;
  __local A_type * output = loc_mem + lsize_0 * lsize_2 * loc_mem_line_IN;

  //////////////////////
  // ALGORITHM
  //////////////////////

  // loop over blocks in third dimension
  for (int d2 = gid_2 * lsize_2; d2 < *chunk_size; d2 += gsize_2)
  {

    // loop over blocks in second dimension
    for (int d0 = gid_0 * lsize_0; d0 < *line_length; d0 += gsize_0)
    {

      // loop over blocks in first dimension
      for (int d1 = gid_1 * LOC_MEM_LINE; d1 < *line_length; d1 += ng_1 * LOC_MEM_LINE)
      {

        //////////////
        // READ: global -> local
        //////////////

        # ifdef CHECKS
        if (d2 + lid_2 < *chunk_size
         && d0 + lid_0 < *line_length
         && d1 + lid_1 < *line_length)
        # endif
        {

        const int pad_index_base = (d2 + lid_2) * pad_slice  // slice
                                 + (d0 + lid_0) + padding;   // line in particular slice

        // local memory
        const int line = get_local_id (0) + get_local_id (2) * get_local_size (0);
        __local A_type * tmp = input + line * loc_mem_line_IN;

        // copy
        const int index = pad_index_base + (d1 + lid_1 - offset) * pad_line_length
                        + (d1 + lid_1 - offset < 0 ? pad_slice : 0);
        tmp [lid_1]           = arg1 [index];
        if (d1 + lid_1 + lsize_1 - offset < *line_length)
          tmp [lid_1 + lsize_1] = arg1 [pad_index_base + (d1 + lid_1 - offset + lsize_1) * pad_line_length];
        if (d1 + lid_1 + 2 * lsize_1 - offset < *line_length && lid_1 < offset)
          tmp [lid_1 + 2 * lsize_1] = arg1 [pad_index_base + (d1 + lid_1 - offset + 2 * lsize_1) * pad_line_length];

        }

        //////////////////
        // COMPUTE: local
        //////////////////
        barrier (CLK_LOCAL_MEM_FENCE);
        
        # ifdef CHECKS
        if (d2 + lid_2 < *chunk_size
         && d0 + lid_0 < *line_length
         && d1 + lid_1 < *line_length)
        if (2 * lid_1 < LOC_MEM_LINE)
        # endif
        {
        
          // local memory
          const int line_in = get_local_id (0) + get_local_id (2) * get_local_size(0);
          __local A_type * tmp_in = input + line_in * loc_mem_line_IN;
          __local A_type * tmp_out = output + line_in * loc_mem_line_IN;
        
          A_type sum = 0;
          # pragma unroll
          for (int k = FL-1; k >= 0; k--)
          {
            sum += tmp_in [2 * lid_1 + k] * _lpf [k];
          }
          tmp_out [lid_1] = sum;
        
          sum = 0;
          # pragma unroll
          for (int k = FL-1; k >= 0; k--)
          {
            sum += tmp_in [2 * lid_1 + FL-1 - k] * _hpf [k];
          }
          tmp_out [lid_1 + LOC_MEM_LINE/2] = sum;
        
        }

        //////////////
        // WRITE: local -> global
        //////////////
        barrier (CLK_LOCAL_MEM_FENCE);
        
        # ifdef CHECKS
        if (d2 + lid_2 < *chunk_size
         && d0 + lid_0 < *line_length
         && d1 + lid_1 < *line_length)
        if (lid_1 + d1/2 < *line_length/2
         && lid_1 < LOC_MEM_LINE/2)
        # endif
        {
        
          // local memory
          const int line_out = get_local_id (0) + get_local_id (2) * get_local_size (0);
          __local A_type * tmp_out = output + line_out * loc_mem_line_IN;
        
          const int index_base = (d2 + lid_2) * slice
                               + (d0 + lid_0);
        
          const int index2 = index_base + (d1/2 + lid_1) * *line_length;
          arg2 [index2 + (shift + (d1/2 + lid_1 + shift < 0 ? *line_length/2 : 0)) * *line_length] = tmp_out [lid_1];
          arg2 [index2 + *line_length/2 * *line_length] = tmp_out [lid_1 + LOC_MEM_LINE/2];
        
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
  const int pad_line_length = roundUp (*chunk_size_0 + padding, ROUND_TO);
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

  const int loc_mem_line_IN = LOC_MEM_LINE + offset + 1;
  const int loc_mem_line_OUT = LOC_MEM_LINE + 1;

  __local A_type * input = loc_mem;
  __local A_type * output = loc_mem + lsize_0 * lsize_1 * loc_mem_line_IN;

  //////////////////////
  // ALGORITHM
  //////////////////////

  // loop over blocks in third dimension
  for (int d1 = gid_1 * lsize_1; d1 < *chunk_size_1; d1 += gsize_1)
  {

    // loop over blocks in second dimension
    for (int d0 = gid_0 * lsize_0; d0 < *chunk_size_0; d0 += gsize_0)
    {

      // loop over blocks in first dimension
      for (int d2 = gid_2 * LOC_MEM_LINE; d2 < *line_length; d2 += ng_2 * LOC_MEM_LINE)
      {

        //////////////
        // READ: global -> local
        //////////////

        # ifdef CHECKS
        if (d1 + lid_1 < *chunk_size_1
         && d0 + lid_0 < *chunk_size_0
         && d2 + lid_2 < *line_length)
        # endif
        {

        const int pad_index_base = (d1 + lid_1) * pad_line_length  // ...
                                 + (d0 + lid_0) + padding;         // line in particular slice

        // local memory
        const int line = get_local_id (0) + get_local_id (1) * get_local_size (0);
        __local A_type * tmp = input + line * loc_mem_line_IN;

        // copy
        const int index1 = pad_index_base + (d2 + lid_2 - offset) * pad_slice
                         + (d2 + lid_2 - offset < 0 ? *line_length * pad_slice : 0);
        tmp [lid_2]           = arg1 [index1];
        if (d2 + lid_2 - offset + lsize_2 < *line_length)
          tmp [lid_2 + lsize_2] = arg1 [pad_index_base + (d2 + lid_2 - offset + lsize_2) * pad_slice];
        if (d2 + lid_2 - offset + 2 * lsize_2 < *line_length && lid_2 < offset)
          tmp [lid_2 + 2 * lsize_2] = arg1 [pad_index_base + (d2 + lid_2 - offset + 2 * lsize_2) * pad_slice];

        }

        //////////////////
        // COMPUTE: local
        //////////////////
        barrier (CLK_LOCAL_MEM_FENCE);

        # ifdef CHECKS
        if (d1 + lid_1 < *chunk_size_1
         && d0 + lid_0 < *chunk_size_0
         && d2 + lid_2 < *line_length)
        if (2 * lid_2 < LOC_MEM_LINE)
        # endif
        {
        
          // local memory
          const int line_in = get_local_id (0) + get_local_id (1) * get_local_size(0);
          __local A_type * tmp_in = input + line_in * loc_mem_line_IN;
          __local A_type * tmp_out = output + line_in * loc_mem_line_IN;
        
          A_type sum = 0;
          # pragma unroll
          for (int k = FL-1; k >= 0; k--)
          {
            sum += tmp_in [2 * lid_2 + k] * _lpf [k];
          }
          tmp_out [lid_2] = sum;
        
          sum = 0;
          # pragma unroll
          for (int k = FL-1; k >= 0; k--)
          {
            sum += tmp_in [2 * lid_2 + FL-1 - k] * _hpf [k];
          }
          tmp_out [lid_2 + LOC_MEM_LINE/2] = sum;
        
        }

        //////////////
        // WRITE: local -> global
        //////////////
        barrier (CLK_LOCAL_MEM_FENCE);

        # ifdef CHECKS
        if (d1 + lid_1 < *chunk_size_1
         && d0 + lid_0 < *chunk_size_0
         && d2 + lid_2 < *line_length)
        if (lid_2 + d2/2 < *line_length/2
         && lid_2 < LOC_MEM_LINE/2)
        # endif
        {
        
          // local memory
          const int line_out = get_local_id (0) + get_local_id (1) * get_local_size (0);
          __local A_type * tmp_out = output + line_out * loc_mem_line_IN;
        
          const int index_base = (d1 + lid_1) * *chunk_size_0
                               + (d0 + lid_0);
        
          const int index2 = index_base + (d2/2 + lid_2) * slice;
          arg2 [index2 + (shift + (d2/2 + lid_2 + shift < 0 ? *line_length/2 : 0))*slice] = tmp_out [lid_2];
          arg2 [index2 + *line_length/2 * slice] = tmp_out [lid_2 + LOC_MEM_LINE/2];
        
        }
        
      } // loop over first dimension

    } // loop over second dimension

  } // loop over third dimension

}