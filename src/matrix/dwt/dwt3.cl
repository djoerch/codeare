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

    const int start_lo = 0;

    // reused index parts
    const int part_index = 2 * local_c1 + start_lo;
    const int part_index2 = local_c1;

    // i: loop over rows per thread -> c1
    int i;
    for (i = 0; i < block_size / group_size; i++)
    {
      const int index = part_index + i * group_size;
      const int index2 = part_index2 + i * group_size/2;
      tmp2 [index2] = conv_step_lo (index, _lpf, tmp, 1);
    }
    if (i * group_size + local_c1 < block_size)
    {
      const int index = part_index + i * group_size;
      const int index2 = part_index2 + i * group_size/2;
      tmp2 [index2] = conv_step_lo (index, _lpf, tmp, 1);
    }

  } // end of lowpass filter

  else  // highpass filter
  {

    const int start_hi = FL;

    // reused index parts
    const int part_index = 2 * (local_c1 - group_size / 2) - 1 + start_hi;
    const int part_index2 = local_c1 - group_size / 2 + block_size / 2;

    // i: loop over elements per column per thread
    int i;
    for (i = 0; i < block_size / group_size; i++)
    {
      const int index = part_index + i * group_size;
      const int index2 = part_index2 + i * group_size / 2;
      tmp2 [index2] = conv_step_hi (index, _hpf, tmp, 1);
    }
    if (i * group_size + local_c1 - group_size / 2 < block_size)
    {
      const int index = part_index + i * group_size;
      const int index2 = part_index2 + i * group_size / 2;
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

  const int block_size = *line_length / (min (*line_length, (int) get_global_size (2))/get_local_size (2));
  const int border_block_size = block_size + offset;

  __local A_type * tmp = & loc_mem [0];
  __local A_type * tmp2 = & loc_mem [border_block_size];

  const int base_index = get_group_id (0) + get_group_id (1) * *n;


  // loop over size of local memory (over slices)
    
    // load GROUP_SIZE elements to local memory
    const int index = base_index + (get_local_id (2) < offset ? (LDC - get_local_id (2)) : (get_local_id (2) - offset)) * LDA * LDB;
    tmp [get_local_id (2)] = arg1 [index];
    
    barrier (CLK_LOCAL_MEM_FENCE);
    
    // perform calculation
    filter (get_local_id (2), tmp, tmp2, _lpf, _hpf, block_size, border_block_size);

//    if (get_local_id (2) < GROUP_SIZE / 2)
//    {
//        tmp2 [get_local_id (2)] = conv_step_lo (2 * get_local_id (2), _lpf, tmp, 1);
//    }
//    else
//    {
//        tmp2 [get_local_id (2)] = conv_step_hi (2 * (get_local_id (2) - GROUP_SIZE/2), _hpf, tmp, 1);
//    }
    
    barrier (CLK_LOCAL_MEM_FENCE);
    
    const int shift = - offset / 2;
    
    // write back
    arg2 [base_index + get_local_id (2) * LDA * LDB] = tmp2 [get_local_id (2)];
    
    arg2 [0] = 44.4;
    
}