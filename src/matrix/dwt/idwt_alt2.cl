// created on Jan 5, 2014

#pragma OPENCL EXTENSION cl_nv_pragma_unroll : enable

# ifndef I_OFFSET
  __constant const int i_offset = FL-1;
  # define I_OFFSET
# endif

# ifndef iLOC_MEM_LINE
 # define iLOC_MEM_LINE 64
# endif

# ifndef CHECKS
// # define CHECKS
# endif

/**
 * @author djoergens
 */
kernel void idwt_1_alt (__global A_type * arg1,
          __constant B_type * _lpf,
          __constant B_type * _hpf,
          __global A_type * arg2,
          __local A_type * loc_mem,
          __constant int * n,
          __global int * m,
          __global int * k,
          __constant int * line_length,
          __constant int * chunk_size)
{

  ///////////////////
  // WRITE CONFIG
  ///////////////////
  const int slice = *n * *n;

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

  const int loc_mem_line_IN = iLOC_MEM_LINE + 2 * i_offset;
  const int loc_mem_line_OUT = iLOC_MEM_LINE + 1;

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
      for (int d0 = gid_0 * iLOC_MEM_LINE/2; d0 < *line_length/2; d0 += ng_0 * iLOC_MEM_LINE/2)
      {

        const int half_current_line_length = min (iLOC_MEM_LINE, *line_length - 2 * d0)/2;

        //////////////
        // READ: global -> local
        //////////////
        
        # ifdef CHECKS
        if (d2 + lid_2 < *chunk_size
          && d1 + lid_1 < *line_length
          && d0 + lid_0 < *line_length/2)
        # endif
        {

          const int index = (d2 + lid_2) * slice                   // slice
                          + (d1 + lid_1) * *n           // line in particular slice
                          + d0 + lid_0;
          
          // local memory
          const int line = get_local_id (1) + get_local_id (2) * get_local_size (1);

          __local A_type * tmp = input + line * loc_mem_line_IN;

          // copy
          const int constraint = *line_length/2 - d0 - lid_0;
          for (int i = 0; i < iLOC_MEM_LINE/2; i += lsize_0)
            if (i < constraint)
            {
              tmp [lid_0 + i + i_offset] = arg1 [index + i];
              tmp [lid_0 + i + i_offset + iLOC_MEM_LINE/2] = arg1 [index + i + *line_length/2];
            }
          if (lid_0 < i_offset)
          {
            tmp [lid_0] = arg1 [index - i_offset + ((d0 + lid_0 - i_offset) < 0 ? *line_length/2 : 0)];
            tmp [lid_0 + i_offset + iLOC_MEM_LINE/2 + half_current_line_length] = arg1 [index + half_current_line_length + ((d0 + lid_0 + half_current_line_length + 1) > *line_length/2 ? 0 : *line_length/2)];
          }
                
        }

        //////////////////
        // COMPUTE: local
        //////////////////
        barrier (CLK_LOCAL_MEM_FENCE);
        
        # ifdef CHECKS
        if (d2 + lid_2 < *chunk_size
          && d1 + lid_0 < *line_length
          && d0 + lid_1 < *line_length/2)
        if (lid_1 < iLOC_MEM_LINE/2
         && lid_2 * lsize_0 < lsize_2 * lsize_1
         && lid_1 + d0 < *line_length/2)
        # endif
        {
                  
          // local memory
          const int line_in = get_local_id (0) + get_local_id (2) * get_local_size(0);

          __local A_type * tmp_in = input + line_in * loc_mem_line_IN;
          __local A_type * tmp_out = output + line_in * loc_mem_line_OUT;

          for (int i = 0; i < iLOC_MEM_LINE/2; i += lsize_1)
          {

            A_type sum1 = 0, sum2 = 0;
            # pragma unroll
            for (int k = 0; k < FL/2; k++)
            {
              const A_type value = tmp_in [i_offset + i + lid_1 - k];
              sum1 += value * _lpf [2 * k];
              sum2 += value * _lpf [2 * k + 1];
            }

            # pragma unroll
            for (int k = 0; k < FL/2; k++)
            {
              const A_type value = tmp_in [i_offset + iLOC_MEM_LINE/2 + i + lid_1 + k];
              sum1 += value * _hpf [2 * k + 1];
              sum2 += value * _hpf [2 * k];
            }
            
            tmp_out [2 * (i + lid_1)] = sum1;
            tmp_out [2 * (i + lid_1) + 1] = sum2;

          }

        }
        
        //////////////
        // WRITE: local -> global
        //////////////
        barrier (CLK_LOCAL_MEM_FENCE);
        
        # ifdef CHECKS
        if (d2 + lid_2 < *chunk_size
          && d1 + lid_1 < *line_length
          && 2 * d0 + lid_0 < *line_length)
        if (lid_0 + 2 * d0 < *line_length
         && lid_0 < iLOC_MEM_LINE)
        # endif
        {
          
          // local memory
          const int line_out = get_local_id (1) + get_local_id (2) * get_local_size (1);

          __local A_type * tmp_out = output + line_out * loc_mem_line_OUT;

          const int index2 = (d2 + lid_2) * slice
                           + (d1 + lid_1) * *n
                           + 2 * d0 + lid_0;

          const int constraint = *line_length - 2 * d0 - lid_0;
          for (int i = 0; i < iLOC_MEM_LINE; i += lsize_0)
          {
            if (i < constraint)
            {
              arg2 [index2 + i] = tmp_out [lid_0 + i];
            }
          }

        }
        
      } // loop over first dimension

    } // loop over second dimension

  } // loop over third dimension

}


/**
 * @author djoergens
 */
kernel void idwt_2_alt (__global A_type * arg1,
          __constant B_type * _lpf,
          __constant B_type * _hpf,
          __global A_type * arg2,
          __local A_type * loc_mem,
          __constant int * n,
          __global int * m,
          __global int * k,
          __constant int * line_length,
          __constant int * chunk_size)
{

  ///////////////////
  // WRITE CONFIG
  ///////////////////
  const int slice = *n * *m;

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

  const int loc_mem_line_IN = iLOC_MEM_LINE + 2 * i_offset;
  const int loc_mem_line_OUT = iLOC_MEM_LINE + 1;

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
      for (int d1 = gid_1 * iLOC_MEM_LINE/2; d1 < *line_length/2; d1 += ng_1 * iLOC_MEM_LINE/2)
      {

        const int half_current_line_length = min (iLOC_MEM_LINE, *line_length - 2 * d1) / 2;

        //////////////
        // READ: global -> local
        //////////////
        
        # ifdef CHECKS
        if (d2 + lid_2 < *chunk_size
          && d0 + lid_0 < *line_length
          && d1 + lid_1 < *line_length/2)
        # endif
        {

          const int index = (d2 + lid_2) * slice                   // slice
                          + (d0 + lid_0)           // line in particular slice
                          + (d1 + lid_1) * *n;
          
          // local memory
          const int line = get_local_id (0) + get_local_id (2) * get_local_size (0);

          __local A_type * tmp = input + line * loc_mem_line_IN;

          // copy
          const int constraint = *line_length/2 - d1 - lid_1;
          for (int i = 0; i < iLOC_MEM_LINE/2; i += lsize_1)
            if (i < constraint)
            {
              tmp [lid_1 + i + i_offset] = arg1 [index + i * *n];
              tmp [lid_1 + i + i_offset + iLOC_MEM_LINE/2] = arg1 [index + (i + *line_length/2) * *n];
            }
          if (lid_1 < i_offset)
          {
            tmp [lid_1] = arg1 [index - i_offset * *n + ((d1 + lid_1 - i_offset) < 0 ? *line_length/2 : 0) * *n];
            tmp [lid_1 + i_offset + iLOC_MEM_LINE/2 + half_current_line_length] = arg1 [index + half_current_line_length * *n + ((d1 + lid_1 + half_current_line_length + 1) > *line_length/2 ? 0 : *line_length/2) * *n];
          }
        
        }

        //////////////////
        // COMPUTE: local
        //////////////////
        barrier (CLK_LOCAL_MEM_FENCE);
        
        # ifdef CHECKS
        if (d2 + lid_2 < *chunk_size
          && d0 + lid_0 < *line_length
          && d1 + lid_1 < *line_length/2)
        if (lid_1 < iLOC_MEM_LINE/2
//         && lid_2 * lsize_0 < lsize_2 * lsize_1
         && lid_1 + d1 < *line_length/2)
        # endif
        {
                  
          // local memory
          const int line_in = get_local_id (0) + get_local_id (2) * get_local_size(0);

          __local A_type * tmp_in = input + line_in * loc_mem_line_IN;
          __local A_type * tmp_out = output + line_in * loc_mem_line_OUT;

          for (int i = 0; i < iLOC_MEM_LINE/2; i += lsize_1)
          {

            A_type sum1 = 0, sum2 = 0;
            # pragma unroll
            for (int k = 0; k < FL/2; k++)
            {
              const A_type value = tmp_in [i_offset + i + lid_1 - k];
              sum1 += value * _lpf [2 * k];
              sum2 += value * _lpf [2 * k + 1];
            }

            # pragma unroll
            for (int k = 0; k < FL/2; k++)
            {
              const A_type value = tmp_in [i_offset + iLOC_MEM_LINE/2 + i + lid_1 + k];
              sum1 += value * _hpf [2 * k + 1];
              sum2 += value * _hpf [2 * k];
            }

            tmp_out [2 * (i + lid_1)] = sum1;
            tmp_out [2 * (i + lid_1) + 1] = sum2;

          }

        }
        
        //////////////
        // WRITE: local -> global
        //////////////
        barrier (CLK_LOCAL_MEM_FENCE);
        
        # ifdef CHECKS
        if (d2 + lid_2 < *chunk_size
          && d0 + lid_0 < *line_length
          && 2 * d1 + lid_1 < *line_length)
        if (lid_1 + 2 * d1 < *line_length
         && lid_1 < iLOC_MEM_LINE)
        # endif
        {
          
          // local memory
          const int line_out = get_local_id (0) + get_local_id (2) * get_local_size (0);

          __local A_type * tmp_out = output + line_out * loc_mem_line_OUT;

          const int index2 = (d2 + lid_2) * slice
                           + (2 * d1 + lid_1) * *n
                           + d0 + lid_0;

          const int constraint = *line_length - 2 * d1 - lid_1;
          for (int i = 0; i < iLOC_MEM_LINE; i += lsize_1)
          {
            if (i < constraint)
            {
              arg2 [index2 + i * *n] = tmp_out [lid_1 + i];
            }
          }

        }
        
      } // loop over first dimension

    } // loop over second dimension

  } // loop over third dimension

}


/**
 * @author djoergens
 */
kernel void idwt_3_alt (__global A_type * arg1,
          __constant B_type * _lpf,
          __constant B_type * _hpf,
          __global A_type * arg2,
          __local A_type * loc_mem,
          __constant int * n,
          __global int * m,
          __global int * k,
          __constant int * line_length,
          __constant int * chunk_size_0,
          __constant int * chunk_size_1)
{

  ///////////////////
  // WRITE CONFIG
  ///////////////////
  const int slice = *n * *m;

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

  const int loc_mem_line_IN = iLOC_MEM_LINE + 2 * i_offset;
  const int loc_mem_line_OUT = iLOC_MEM_LINE + 1;

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
      for (int d2 = gid_2 * iLOC_MEM_LINE/2; d2 < *line_length/2; d2 += ng_2 * iLOC_MEM_LINE/2)
      {

        const int half_current_line_length = min (iLOC_MEM_LINE, *line_length - 2 * d2) / 2;

        //////////////
        // READ: global -> local
        //////////////
        
        # ifdef CHECKS
        if (d1 + lid_1 < *chunk_size_1
          && d0 + lid_0 < *chunk_size_0
          && d2 + lid_2 < *line_length/2)
        # endif
        {

          const int index_base = (d1 + lid_1) * *n  // 
                               + (d0 + lid_0);                // line in particular slice
          
          // local memory
          const int line = get_local_id (0) + get_local_id (1) * get_local_size (0);

          __local A_type * tmp = input + line * loc_mem_line_IN;

          // copy
          const int index = index_base + (d2 + lid_2) * slice;

          const int constraint = *line_length/2 - d2 - lid_2;
          for (int i = 0; i < iLOC_MEM_LINE/2; i += lsize_2)
            if (i < constraint)
            {
              tmp [lid_2 + i + i_offset] = arg1 [index + i * slice];
              tmp [lid_2 + i + i_offset + iLOC_MEM_LINE/2] = arg1 [index + (i + *line_length/2) * slice];
            }
          if (lid_2 < i_offset)
          {
            tmp [lid_2] = arg1 [index - i_offset * slice + ((d2 + lid_2 - i_offset) < 0 ? *line_length/2 : 0) * slice];
            tmp [lid_2 + i_offset + iLOC_MEM_LINE/2 + half_current_line_length] = arg1 [index + half_current_line_length * slice + ((d2 + lid_2 + half_current_line_length + 1) > *line_length/2 ? 0 : *line_length/2) * slice];
          }
        
        }

        //////////////////
        // COMPUTE: local
        //////////////////
        barrier (CLK_LOCAL_MEM_FENCE);
        
        # ifdef CHECKS
        if (d1 + lid_1 < *chunk_size_1
          && d0 + lid_0 < *chunk_size_0
          && d2 + lid_2 < *line_length/2)
        if (lid_2 < iLOC_MEM_LINE/2
//         && lid_2 * lsize_0 < lsize_2 * lsize_1
         && lid_2 + d2 < *line_length/2)
        # endif
        {
           
          // local memory
          const int line_in = get_local_id (0) + get_local_id (1) * get_local_size(0);

          __local A_type * tmp_in = input + line_in * loc_mem_line_IN;
          __local A_type * tmp_out = output + line_in * loc_mem_line_OUT;

          for (int i = 0; i < iLOC_MEM_LINE/2; i += lsize_2)
          {

            A_type sum1 = 0, sum2 = 0;
            # pragma unroll
            for (int k = 0; k < FL/2; k++)
            {
              const A_type value = tmp_in [i_offset + i + lid_2 - k];
              sum1 += value * _lpf [2 * k];
              sum2 += value * _lpf [2 * k + 1];
            }

            # pragma unroll
            for (int k = 0; k < FL/2; k++)
            {
              const A_type value = tmp_in [i_offset + iLOC_MEM_LINE/2 + i + lid_2 + k];
              sum1 += value * _hpf [2 * k + 1];
              sum2 += value * _hpf [2 * k];
            }

            tmp_out [2 * (i + lid_2)] = sum1;
            tmp_out [2 * (i + lid_2) + 1] = sum2;

          }

        }
        
        //////////////
        // WRITE: local -> global
        //////////////
        barrier (CLK_LOCAL_MEM_FENCE);
        
        # ifdef CHECKS
        if (d1 + lid_1 < *chunk_size_1
          && d0 + lid_0 < *chunk_size_0
          && 2 * d2 + lid_2 < *line_length)
        if (lid_2 + 2 * d2 < *line_length
         && lid_2 < iLOC_MEM_LINE)
        # endif
        {
          
          // local memory
          const int line_out = get_local_id (0) + get_local_id (1) * get_local_size (0);

          __local A_type * tmp_out = output + line_out * loc_mem_line_OUT;

          const int index2 = (2 * d2 + lid_2) * slice
                           + (d1 + lid_1) * *n
                           + d0 + lid_0;

          const int constraint = *line_length - 2 * d2 - lid_2;
          for (int i = 0; i < iLOC_MEM_LINE; i += lsize_2)
          {
            if (i < constraint)
            {
              arg2 [index2 + i * slice] = tmp_out [lid_2 + i];
            }
          }

        }
        
      } // loop over first dimension

    } // loop over second dimension

  } // loop over third dimension

}


/**
 * @author djoergens
 */
kernel void idwt_final_alt (__global A_type * arg1,
          __constant B_type * _lpf,
          __constant B_type * _hpf,
          __global A_type * arg2,
          __local A_type * loc_mem,
          __constant int * n,
          __global int * m,
          __global int * k,
          __constant int * line_length,
          __constant int * chunk_size)
{
  
  ///////////////////
  // READING CONFIG
  ///////////////////
  const int pad_line_length = roundUp (*line_length + PADDING, ROUND_TO);
  const int pad_slice = pad_line_length * *line_length;

  ///////////////////
  // WRITE CONFIG
  ///////////////////
  const int slice = *n * *n;

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
  const int gsize_0 = get_global_size (0);
  const int gsize_1 = get_global_size (1);
  const int gsize_2 = get_global_size (2);

  //////////////////////
  // ALGORITHM
  //////////////////////
  # ifdef NON_SQUARE_GROUP
  const int num_mem_lines = max (lsize_0, lsize_1);
  # endif

  // loop over blocks in third dimension
  for (int d2 = gid_2 * lsize_2; d2 < *chunk_size; d2 += gsize_2)
  {

    // loop over blocks in second dimension
    for (int d1 = gid_1 * lsize_1; d1 < *line_length; d1 += gsize_1)
    {

      // loop over blocks in first dimension
      for (int d0 = gid_0 * lsize_0; d0 < *line_length; d0 += gsize_0)
      {

        //////////////
        // COPY: global -> global
        //////////////
        
        if (d2 + lid_2 < *chunk_size
          && d1 + lid_1 < *line_length
          && d0 + lid_0 < *line_length)
        {

          const int index = (d2 + lid_2) * slice                   // slice
                          + (d1 + lid_1) * *n   // line in particular slice
                          + d0 + lid_0;
          
          // copy
          arg2 [index] = arg1 [index];
          
        }
        
      } // loop over first dimension

    } // loop over second dimension

  } // loop over third dimension

}
