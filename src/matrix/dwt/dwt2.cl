// created on Aug 28, 2013


#pragma OPENCL EXTENSION cl_amd_printf: enable

//int printf(constant char * restrict format, ...);

//typedef float A_type;

# ifndef GROUP_SIZE
  # define GROUP_SIZE 128
# endif

# ifndef NUM_GROUPS
  # define NUM_GROUPS 4
# endif

# ifndef LINE_LENGTH
  # define LINE_LENGTH 512
# endif

# ifndef FL
  # define FL 4
# endif


# define BLOCK_SIZE LINE_LENGTH/NUM_GROUPS
# define BORDER_BLOCK_SIZE LINE_LENGTH/NUM_GROUPS+FL
# define OFFSET FL


A_type
conv_step           (const int index, __constant A_type * _filter, __local A_type * tmp)
{
  A_type sum = 0;
  # pragma unroll
  for (int k = FL-1; k >= 0; k--)
  {
    sum += tmp [index - k] * _filter [k];
  }
  return sum;
}


void
filtering           (const int local_c1, const int local_c2,
                     __local A_type * tmp, __local A_type * tmp2,
                     __constant A_type * _lpf, __constant A_type * _hpf)
{

    // COLUMNS

    if (local_c1 < GROUP_SIZE/2)  // lowpass filter
    {

      // reused index parts
      const int part_index = (local_c2 + FL) * BORDER_BLOCK_SIZE + 2 * local_c1 + FL;
      const int part_index2 = local_c2 * BLOCK_SIZE + local_c1;

      // j: loop over columns per thread
      // i: loop over rows per thread
      int j;
      # pragma unroll
      for (j = 0; j < BLOCK_SIZE-GROUP_SIZE; j += GROUP_SIZE)
      {
        int i;
        for (i = 0; i < BLOCK_SIZE/GROUP_SIZE-1; i++)
        {
          const int index = part_index + j * BORDER_BLOCK_SIZE + i * GROUP_SIZE;
          const int index2 = part_index2 + j * BLOCK_SIZE + i * GROUP_SIZE/2;
          tmp2 [index2] = conv_step (index, _lpf, tmp);
        }
        if (i*GROUP_SIZE + local_c1 < BLOCK_SIZE)
        {
          const int index = part_index + j * BORDER_BLOCK_SIZE + i * GROUP_SIZE;
          const int index2 = part_index2 + j * BLOCK_SIZE + i * GROUP_SIZE/2;
          tmp2 [index2] = conv_step (index, _lpf, tmp);
        }
      }
    
      // for "odd" block sizes
      if (j + local_c2 < BLOCK_SIZE)
      {
        int i;
        for (i = 0; i < BLOCK_SIZE/GROUP_SIZE-1; i++)
        {
          const int index = part_index + j * BORDER_BLOCK_SIZE + i * GROUP_SIZE;
          const int index2 = part_index2 + j * BLOCK_SIZE + i * GROUP_SIZE/2;
          tmp2 [index2] = conv_step (index, _lpf, tmp);
        }
        if (i * GROUP_SIZE + local_c1 < BLOCK_SIZE)
        {
          const int index = part_index + j * BORDER_BLOCK_SIZE + i * GROUP_SIZE;
          const int index2 = part_index2 + j * BLOCK_SIZE + i * GROUP_SIZE/2;
          tmp2 [index2] = conv_step (index, _lpf, tmp);
        }
      }
    
    } // end of lowpass filter
    
    else  // highpass filter
    { 
        
      // reused index parts
      const int part_index = (local_c2 + FL) * BORDER_BLOCK_SIZE + 2 * (local_c1 - GROUP_SIZE/2) + 1 + FL;
      const int part_index2 = local_c2 * BLOCK_SIZE + local_c1 - GROUP_SIZE/2 + BLOCK_SIZE/2;
        
      // j: loop over columns per thread
      // i: loop over rows per thread
      int j;
      # pragma unroll
      for (j = 0; j < BLOCK_SIZE-GROUP_SIZE; j += GROUP_SIZE)
      {
        int i;
        for (i = 0; i < BLOCK_SIZE/GROUP_SIZE-1; i++)
        {
          const int index = part_index + j * BORDER_BLOCK_SIZE + i * GROUP_SIZE;
          const int index2 = part_index2 + j * BLOCK_SIZE + i * GROUP_SIZE/2;
          tmp2 [index2] = conv_step (index, _hpf, tmp);
        }
        if (i*GROUP_SIZE + local_c1 - GROUP_SIZE/2 < BLOCK_SIZE)
        {
          const int index = part_index + j * BORDER_BLOCK_SIZE + i * GROUP_SIZE;
          const int index2 = part_index2 + j * BLOCK_SIZE + i * GROUP_SIZE/2;
          tmp2 [index2] = conv_step (index, _hpf, tmp);          
        }
      }
    
      // for "odd" block sizes
      if (j + local_c2 < BLOCK_SIZE)
      {
        int i;
        for (i = 0; i < BLOCK_SIZE/GROUP_SIZE-1; i++)
        {
          const int index = part_index + j * BORDER_BLOCK_SIZE + i * GROUP_SIZE;
          const int index2 = part_index2 + j * BLOCK_SIZE + i * GROUP_SIZE/2;
          tmp2 [index2] = conv_step (index, _hpf, tmp);
        }
        if (i*GROUP_SIZE + local_c1 - GROUP_SIZE/2 < BLOCK_SIZE)
        {
          const int index = part_index + j * BORDER_BLOCK_SIZE + i * GROUP_SIZE;
          const int index2 = part_index2 + j * BLOCK_SIZE + i * GROUP_SIZE/2;
          tmp2 [index2] = conv_step (index, _hpf, tmp);          
        }        
      }
    
    } // end of highpass filter

}


/**
 * @author djoergens
 */
kernel void dwt2 (__global A_type * arg1,
          __constant A_type * _lpf,
          __constant A_type * _hpf,
          __global A_type * arg2,
          __local A_type * loc_mem,
          __constant int * n,
          __global int * m,
          __global int * k,
          __global int * fl,
          __global int * loc_mem_size)
{

  const int local_c1 = get_local_id (0);
  const int local_c2 = get_local_id (1);

//  const int block_size = LINE_LENGTH / NUM_GROUPS;
//  const int border_block_size = LINE_LENGTH / NUM_GROUPS + FL;
//  const int offset = FL;

  __local A_type * tmp  = & loc_mem [0];
  __local A_type * tmp2 = & loc_mem [BORDER_BLOCK_SIZE * BORDER_BLOCK_SIZE];

  const int upper_left = get_group_id (1) * BLOCK_SIZE * LINE_LENGTH
                       + get_group_id (0) * BLOCK_SIZE
                       - OFFSET * LINE_LENGTH
                       - OFFSET;
  const int upper_left2 = get_group_id (1) * BLOCK_SIZE * LINE_LENGTH
                        + get_group_id (0) * BLOCK_SIZE / 2;

  /////////////////////////////
  // load block to local memory
  /////////////////////////////

  int j;
  for (j = 0; j < BORDER_BLOCK_SIZE-GROUP_SIZE; j += GROUP_SIZE)
  {
    int i;
    for (i = 0; i < BORDER_BLOCK_SIZE-GROUP_SIZE; i += GROUP_SIZE)
    {
//      if (upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i < 0)
//      {
//        tmp [(local_c2 + j) * border_block_size + local_c1 + i]
//            = arg1 [upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i];
//        continue;
//      }
      tmp [(local_c2 + j) * BORDER_BLOCK_SIZE + local_c1 + i]
          = arg1 [upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i];
    }
    if (i + local_c1 < BORDER_BLOCK_SIZE)
      tmp [(local_c2 + j) * BORDER_BLOCK_SIZE + local_c1 + i]
          = arg1 [upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i];
  }
  if (j + local_c2 < BORDER_BLOCK_SIZE)
  {
    int i;
    for (i = 0; i < BORDER_BLOCK_SIZE-GROUP_SIZE; i += GROUP_SIZE)
    {
//      if (upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i < 0)
//      {
//        tmp [(local_c2 + j) * border_block_size + local_c1 + i]
//            = arg1 [upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i];
//        continue;
//      }
      tmp [(local_c2 + j) * BORDER_BLOCK_SIZE + local_c1 + i]
          = arg1 [upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i];
    }
    if (i + local_c1 < BORDER_BLOCK_SIZE)
      tmp [(local_c2 + j) * BORDER_BLOCK_SIZE + local_c1 + i]
          = arg1 [upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i];
  }
    
  barrier (CLK_LOCAL_MEM_FENCE);

  // filter operations
  filtering (local_c1, local_c2, tmp, tmp2, _lpf, _hpf);

  barrier (CLK_LOCAL_MEM_FENCE);

  //////////////////////////////
  // write back to global memory
  //////////////////////////////

  // lowpass part
  for (j = 0; j < BLOCK_SIZE-GROUP_SIZE; j += GROUP_SIZE)
  {
    int i;
    for (i = 0; i < BLOCK_SIZE/2-GROUP_SIZE; i += GROUP_SIZE)
      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i]
           = tmp2 [(local_c2 + j) * BLOCK_SIZE + local_c1 + i];
    if (i + local_c1 < BLOCK_SIZE/2)
      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i]
           = tmp2 [(local_c2 + j) * BLOCK_SIZE + local_c1 + i];
  }
  if (j + local_c2 < BLOCK_SIZE)
  {
    int i;
    for (i = 0; i < BLOCK_SIZE/2-GROUP_SIZE; i += GROUP_SIZE)
      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i]
           = tmp2 [(local_c2 + j) * BLOCK_SIZE + local_c1 + i];
    if (i + local_c1 < BLOCK_SIZE/2)
      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i]
           = tmp2 [(local_c2 + j) * BLOCK_SIZE + local_c1 + i];
  }


  // highpass part
  for (j = 0; j < BLOCK_SIZE-GROUP_SIZE; j += GROUP_SIZE)
  {
    int i;
    for (i = 0; i < BLOCK_SIZE/2-GROUP_SIZE; i += GROUP_SIZE)
      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i + LINE_LENGTH/2]
           = tmp2 [(local_c2 + j) * BLOCK_SIZE + local_c1 + i + BLOCK_SIZE/2];
    if (i + local_c1 < BLOCK_SIZE/2)
      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i + LINE_LENGTH/2]
           = tmp2 [(local_c2 + j) * BLOCK_SIZE + local_c1 + i + BLOCK_SIZE/2];
  }
  if (j + local_c2 < BLOCK_SIZE)
  {
    int i;
    for (i = 0; i < BLOCK_SIZE/2-GROUP_SIZE; i += GROUP_SIZE)
      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i + LINE_LENGTH/2]
           = tmp2 [(local_c2 + j) * BLOCK_SIZE + local_c1 + i + BLOCK_SIZE/2];
    if (i + local_c1 < BLOCK_SIZE/2)
      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i + LINE_LENGTH/2]
           = tmp2 [(local_c2 + j) * BLOCK_SIZE + local_c1 + i + BLOCK_SIZE/2];
  }

}




kernel void idwt2 (__global A_type * arg1,
          __constant A_type * _lpf,
          __constant A_type * _hpf,
          __global A_type * arg2,
          __local A_type * loc_mem,
          __constant int * n,
          __global int * m,
          __global int * k,
          __global int * fl,
          __global int * loc_mem_size)
{

  const int local_c1 = get_local_id (0);
  const int local_c2 = get_local_id (1);

  const int block_size = LINE_LENGTH / NUM_GROUPS;
  const int border_block_size = LINE_LENGTH / NUM_GROUPS + FL;
  const int offset = FL;

  __local A_type * tmp  = & loc_mem [0];
  __local A_type * tmp2 = & loc_mem [border_block_size * border_block_size];

  const int upper_left = get_group_id (1) * block_size * LINE_LENGTH
                       + get_group_id (0) * block_size
                       - offset * LINE_LENGTH
                       - offset;
  const int upper_left2 = get_group_id (1) * block_size * LINE_LENGTH
                        + get_group_id (0) * block_size / 2;

  /////////////////////////////
  // load block to local memory
  /////////////////////////////

  int j;
  for (j = 0; j < border_block_size-GROUP_SIZE; j += GROUP_SIZE)
  {
    int i;
    for (i = 0; i < border_block_size-GROUP_SIZE; i += GROUP_SIZE)
    {
//      if (upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i < 0)
//      {
//        tmp [(local_c2 + j) * border_block_size + local_c1 + i]
//            = arg1 [upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i];
//        continue;
//      }
      tmp [(local_c2 + j) * border_block_size + local_c1 + i]
          = arg1 [upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i];
    }
    if (i + local_c1 < border_block_size)
      tmp [(local_c2 + j) * border_block_size + local_c1 + i]
          = arg1 [upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i];
  }
  if (j + local_c2 < border_block_size)
  {
    int i;
    for (i = 0; i < border_block_size-GROUP_SIZE; i += GROUP_SIZE)
    {
//      if (upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i < 0)
//      {
//        tmp [(local_c2 + j) * border_block_size + local_c1 + i]
//            = arg1 [upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i];
//        continue;
//      }
      tmp [(local_c2 + j) * border_block_size + local_c1 + i]
          = arg1 [upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i];
    }
    if (i + local_c1 < border_block_size)
      tmp [(local_c2 + j) * border_block_size + local_c1 + i]
          = arg1 [upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i];
  }
    
  barrier (CLK_LOCAL_MEM_FENCE);

  // filter operations
  filtering (local_c1, local_c2, tmp, tmp2, _lpf, _hpf);

  barrier (CLK_LOCAL_MEM_FENCE);

  //////////////////////////////
  // write back to global memory
  //////////////////////////////

  // lowpass part
  for (j = 0; j < block_size-GROUP_SIZE; j += GROUP_SIZE)
  {
    int i;
    for (i = 0; i < block_size/2-GROUP_SIZE; i += GROUP_SIZE)
      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i]
           = tmp2 [(local_c2 + j) * block_size + local_c1 + i];
    if (i + local_c1 < block_size/2)
      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i]
           = tmp2 [(local_c2 + j) * block_size + local_c1 + i];
  }
  if (j + local_c2 < block_size)
  {
    int i;
    for (i = 0; i < block_size/2-GROUP_SIZE; i += GROUP_SIZE)
      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i]
           = tmp2 [(local_c2 + j) * block_size + local_c1 + i];
    if (i + local_c1 < block_size/2)
      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i]
           = tmp2 [(local_c2 + j) * block_size + local_c1 + i];
  }


  // highpass part
  for (j = 0; j < block_size-GROUP_SIZE; j += GROUP_SIZE)
  {
    int i;
    for (i = 0; i < block_size/2-GROUP_SIZE; i += GROUP_SIZE)
      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i + LINE_LENGTH/2]
           = tmp2 [(local_c2 + j) * block_size + local_c1 + i + block_size/2];
    if (i + local_c1 < block_size/2)
      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i + LINE_LENGTH/2]
           = tmp2 [(local_c2 + j) * block_size + local_c1 + i + block_size/2];
  }
  if (j + local_c2 < block_size)
  {
    int i;
    for (i = 0; i < block_size/2-GROUP_SIZE; i += GROUP_SIZE)
      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i + LINE_LENGTH/2]
           = tmp2 [(local_c2 + j) * block_size + local_c1 + i + block_size/2];
    if (i + local_c1 < block_size/2)
      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i + LINE_LENGTH/2]
           = tmp2 [(local_c2 + j) * block_size + local_c1 + i + block_size/2];
  }

}