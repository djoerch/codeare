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


  __constant const int offset = FL-1;
  __constant const int block_size = LINE_LENGTH / NUM_GROUPS;
  __constant const int border_block_size = LINE_LENGTH / NUM_GROUPS + offset;


A_type
conv_step           (const int index, __constant A_type * _filter, __local A_type * tmp,
                     const int increment)
{
  A_type sum = 0;
  # pragma unroll
  for (int k = FL-1; k >= 0; k--)
  {
    sum += tmp [index - k * increment] * _filter [k];
  }
  return sum;
}


void
global2local        (__global A_type * arg1, __local A_type * tmp,
                     const int upper_left, const int local_c1, const int local_c2)
{
  int j;
  for (j = 0; j < border_block_size-GROUP_SIZE; j += GROUP_SIZE)
  {
    int i;
    for (i = 0; i < border_block_size-GROUP_SIZE; i += GROUP_SIZE)
    {
      int index = upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i;
      if (upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i < 0)
      {
        index = index + (index < -offset ? LINE_LENGTH * LINE_LENGTH : 0)
                      + (index < 0 && index >= -offset ? LINE_LENGTH : 0);
        tmp [(local_c2 + j) * border_block_size + local_c1 + i]
            = arg1 [index];
        continue;
      }
      tmp [(local_c2 + j) * border_block_size + local_c1 + i] = arg1 [index];
    }
    if (i + local_c1 < border_block_size)
    {
      int index = upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i;
      if (upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i < 0)
      {
        index = index + (index < -offset ? LINE_LENGTH * LINE_LENGTH : 0)
                      + (index < 0 && index >= -offset ? LINE_LENGTH : 0);
        tmp [(local_c2 + j) * border_block_size + local_c1 + i]
            = arg1 [index];
        continue;
      }
      else
      {
        tmp [(local_c2 + j) * border_block_size + local_c1 + i] = arg1 [index];
      }
    }
  }
  if (j + local_c2 < border_block_size)
  {
    int i;
    for (i = 0; i < border_block_size-GROUP_SIZE; i += GROUP_SIZE)
    {
      tmp [(local_c2 + j) * border_block_size + local_c1 + i]
          = arg1 [upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i];
    }
    if (i + local_c1 < border_block_size)
      tmp [(local_c2 + j) * border_block_size + local_c1 + i]
          = arg1 [upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i];
  }
}


void
local2global        (__local A_type * tmp2, __global A_type * arg2,
                     const int upper_left2, const int local_c1, const int local_c2)
{
  
  ///////////
  // part: LL
  ///////////
  int j;
  for (j = 0; j < block_size/2-GROUP_SIZE; j += GROUP_SIZE)
  {
    int i;
    for (i = 0; i < block_size/2-GROUP_SIZE; i += GROUP_SIZE)
      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i]
           = tmp2 [(local_c2 + j) * block_size + local_c1 + i];
    if (i + local_c1 < block_size/2)
      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i]
           = tmp2 [(local_c2 + j) * block_size + local_c1 + i];
  }
  if (j + local_c2 < block_size/2)
  {
    int i;
    for (i = 0; i < block_size/2-GROUP_SIZE; i += GROUP_SIZE)
      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i]
           = tmp2 [(local_c2 + j) * block_size + local_c1 + i];
    if (i + local_c1 < block_size/2)
      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i]
           = tmp2 [(local_c2 + j) * block_size + local_c1 + i];
  }

  ///////////
  // part: LH
  ///////////
  for (j = block_size/2; j < block_size-GROUP_SIZE; j += GROUP_SIZE)
  {
    int i;
    for (i = 0; i < block_size/2-GROUP_SIZE; i += GROUP_SIZE)
      arg2 [upper_left2 + (LINE_LENGTH/2 + local_c2 + j - block_size/2) * LINE_LENGTH + local_c1 + i] = tmp2 [(local_c2 + j) * block_size + local_c1 + i];
    if (i + local_c1 < block_size/2)
      arg2 [upper_left2 + (LINE_LENGTH/2 + local_c2 + j - block_size/2) * LINE_LENGTH + local_c1 + i] = tmp2 [(local_c2 + j) * block_size + local_c1 + i];
  }
  if (j + local_c2 < block_size)
  {
    int i;
    for (i = 0; i < block_size/2-GROUP_SIZE; i += GROUP_SIZE)
      arg2 [upper_left2 + (LINE_LENGTH/2 + local_c2 + j - block_size/2) * LINE_LENGTH + local_c1 + i] = tmp2 [(local_c2 + j) * block_size + local_c1 + i];
    if (i + local_c1 < block_size/2)
      arg2 [upper_left2 + (LINE_LENGTH/2 + local_c2 + j - block_size/2) * LINE_LENGTH + local_c1 + i] = tmp2 [(local_c2 + j) * block_size + local_c1 + i];
  }

  ///////////
  // part: HL
  ///////////
  for (j = 0; j < block_size/2-GROUP_SIZE; j += GROUP_SIZE)
  {
    int i;
    for (i = 0; i < block_size/2-GROUP_SIZE; i += GROUP_SIZE)
      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i + LINE_LENGTH/2]
           = tmp2 [(local_c2 + j) * block_size + local_c1 + i + block_size/2];
    if (i + local_c1 < block_size/2)
      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i + LINE_LENGTH/2]
           = tmp2 [(local_c2 + j) * block_size + local_c1 + i + block_size/2];
  }
  if (j + local_c2 < block_size/2)
  {
    int i;
    for (i = 0; i < block_size/2-GROUP_SIZE; i += GROUP_SIZE)
      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i + LINE_LENGTH/2]
           = tmp2 [(local_c2 + j) * block_size + local_c1 + i + block_size/2];
    if (i + local_c1 < block_size/2)
      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i + LINE_LENGTH/2]
           = tmp2 [(local_c2 + j) * block_size + local_c1 + i + block_size/2];
  }

  ///////////
  // part: HH
  ///////////
  for (j = block_size/2; j < block_size-GROUP_SIZE; j += GROUP_SIZE)
  {
    int i;
    for (i = 0; i < block_size/2-GROUP_SIZE; i += GROUP_SIZE)
      arg2 [upper_left2 + (LINE_LENGTH/2 + local_c2 + j - block_size/2) * LINE_LENGTH + local_c1 + i + LINE_LENGTH/2] = tmp2 [(local_c2 + j) * block_size + local_c1 + i + block_size/2];
    if (i + local_c1 < block_size/2)
      arg2 [upper_left2 + (LINE_LENGTH/2 + local_c2 + j - block_size/2) * LINE_LENGTH + local_c1 + i + LINE_LENGTH/2] = tmp2 [(local_c2 + j) * block_size + local_c1 + i + block_size/2];
  }
  if (j + local_c2 < block_size)
  {
    int i;
    for (i = 0; i < block_size/2-GROUP_SIZE; i += GROUP_SIZE)
      arg2 [upper_left2 + (LINE_LENGTH/2 + local_c2 + j - block_size/2) * LINE_LENGTH + local_c1 + i + LINE_LENGTH/2] = tmp2 [(local_c2 + j) * block_size + local_c1 + i + block_size/2];
    if (i + local_c1 < block_size/2)
      arg2 [upper_left2 + (LINE_LENGTH/2 + local_c2 + j - block_size/2) * LINE_LENGTH + local_c1 + i + LINE_LENGTH/2] = tmp2 [(local_c2 + j) * block_size + local_c1 + i + block_size/2];
  }

}



void
filter_columns           (const int local_c1, const int local_c2,
                          __local A_type * tmp, __local A_type * tmp2,
                          __constant A_type * _lpf, __constant A_type * _hpf)
{

    // COLUMNS

    if (local_c1 < GROUP_SIZE/2)  // lowpass filter
    {

      // reused index parts
      const int part_index = local_c2 * border_block_size + 2 * local_c1 + offset;
      const int part_index2 = local_c2 * block_size + local_c1;

      // j: loop over columns per thread -> c2
      // i: loop over rows per thread -> c1
      int j;
      for (j = 0; j < border_block_size-GROUP_SIZE; j += GROUP_SIZE)
      {
        int i;
        for (i = 0; i < block_size/GROUP_SIZE-1; i++)
        {
          const int index = part_index + j * border_block_size + i * GROUP_SIZE;
          const int index2 = part_index2 + j * block_size + i * GROUP_SIZE/2;
          tmp2 [index2] = conv_step (index, _lpf, tmp, 1);
        }
        if (i*GROUP_SIZE + local_c1 < block_size)
        {
          const int index = part_index + j * border_block_size + i * GROUP_SIZE;
          const int index2 = part_index2 + j * block_size + i * GROUP_SIZE/2;
          tmp2 [index2] = conv_step (index, _lpf, tmp, 1);
        }
      }
    
      // for "odd" block sizes
      if (j + local_c2 < border_block_size)
      {
        int i;
        for (i = 0; i < block_size/GROUP_SIZE-1; i++)
        {
          const int index = part_index + j * border_block_size + i * GROUP_SIZE;
          const int index2 = part_index2 + j * block_size + i * GROUP_SIZE/2;
          tmp2 [index2] = conv_step (index, _lpf, tmp, 1);
        }
        if (i * GROUP_SIZE + local_c1 < block_size)
        {
          const int index = part_index + j * border_block_size + i * GROUP_SIZE;
          const int index2 = part_index2 + j * block_size + i * GROUP_SIZE/2;
          tmp2 [index2] = conv_step (index, _lpf, tmp, 1);
        }
      }
    
    } // end of lowpass filter
    
    else  // highpass filter
    { 
        
      // reused index parts
      const int part_index = local_c2 * border_block_size + 2 * (local_c1 - GROUP_SIZE/2) + 1 + offset;
      const int part_index2 = local_c2 * block_size + local_c1 - GROUP_SIZE/2 + block_size/2;
        
      // j: loop over columns per thread
      // i: loop over elements per column per thread
      int j;
      for (j = 0; j < border_block_size-GROUP_SIZE; j += GROUP_SIZE)
      {
        int i;
        for (i = 0; i < block_size/GROUP_SIZE-1; i++)
        {
          const int index = part_index + j * border_block_size + i * GROUP_SIZE;
          const int index2 = part_index2 + j * block_size + i * GROUP_SIZE/2;
          tmp2 [index2] = conv_step (index, _hpf, tmp, 1);
        }
        if (i*GROUP_SIZE + local_c1 - GROUP_SIZE/2 < block_size)
        {
          const int index = part_index + j * border_block_size + i * GROUP_SIZE;
          const int index2 = part_index2 + j * block_size + i * GROUP_SIZE/2;
          tmp2 [index2] = conv_step (index, _hpf, tmp, 1);          
        }
      }
    
      // for "odd" block sizes
      if (j + local_c2 < border_block_size)
      {
        int i;
        for (i = 0; i < block_size/GROUP_SIZE-1; i++)
        {
          const int index = part_index + j * border_block_size + i * GROUP_SIZE;
          const int index2 = part_index2 + j * block_size + i * GROUP_SIZE/2;
          tmp2 [index2] = conv_step (index, _hpf, tmp, 1);
        }
        if (i*GROUP_SIZE + local_c1 - GROUP_SIZE/2 < block_size)
        {
          const int index = part_index + j * border_block_size + i * GROUP_SIZE;
          const int index2 = part_index2 + j * block_size + i * GROUP_SIZE/2;
          tmp2 [index2] = conv_step (index, _hpf, tmp, 1);          
        }        
      }
    
    } // end of highpass filter

}



void
filter_rows           (const int local_c1, const int local_c2,
                       __local A_type * tmp, __local A_type * tmp2,
                       __constant A_type * _lpf, __constant A_type * _hpf)
{

    // ROWS

    if (local_c2 < GROUP_SIZE/2)  // lowpass filter
    {

      // reused index parts
      const int part_index = local_c1 + (offset + 2 * local_c2) * block_size;
      const int part_index2 = local_c1 + local_c2 * block_size;

      // j: loop over rows per thread -> c1
      // i: loop over elements per row per thread -> c2
      int j;
      for (j = 0; j < block_size-GROUP_SIZE; j += GROUP_SIZE)
      {
        int i;
        for (i = 0; i < block_size/GROUP_SIZE-1; i++)
        {
          const int index = part_index + j + i * GROUP_SIZE * block_size;
          const int index2 = part_index2 + j + i * GROUP_SIZE/2 * block_size;
          tmp2 [index2] = conv_step (index, _lpf, tmp, block_size);
        }
        if (i*GROUP_SIZE + local_c2 < block_size)
        {
          const int index = part_index + j + i * GROUP_SIZE * block_size;
          const int index2 = part_index2 + j + i * GROUP_SIZE/2 * block_size;
          tmp2 [index2] = conv_step (index, _lpf, tmp, block_size);
        }
      }
    
      // for "odd" block sizes
      if (j + local_c1 < block_size)
      {
        int i;
        for (i = 0; i < block_size/GROUP_SIZE-1; i++)
        {
          const int index = part_index + j + i * GROUP_SIZE * block_size;
          const int index2 = part_index2 + j + i * GROUP_SIZE/2 * block_size;
          tmp2 [index2] = conv_step (index, _lpf, tmp, block_size);
        }
        if (i * GROUP_SIZE + local_c2 < block_size)
        {
          const int index = part_index + j + i * GROUP_SIZE * block_size;
          const int index2 = part_index2 + j + i * GROUP_SIZE/2 * block_size;
          tmp2 [index2] = conv_step (index, _lpf, tmp, block_size);
        }
      }
    
    } // end of lowpass filter
    
    else  // highpass filter
    { 
        
      // reused index parts
      const int part_index = local_c1 + (offset + 2 * (local_c2 - GROUP_SIZE/2) + 1) * block_size;
      const int part_index2 = local_c1 + (local_c2 - GROUP_SIZE/2 + block_size/2) * block_size;
        
      // j: loop over columns per thread
      // i: loop over rows per thread
      int j;
      for (j = 0; j < block_size-GROUP_SIZE; j += GROUP_SIZE)
      {
        int i;
        for (i = 0; i < block_size/GROUP_SIZE-1; i++)
        {
          const int index = part_index + j + i * GROUP_SIZE * block_size;
          const int index2 = part_index2 + j + i * GROUP_SIZE/2 * block_size;
          tmp2 [index2] = conv_step (index, _hpf, tmp, block_size);
        }
        if (i*GROUP_SIZE + local_c2 - GROUP_SIZE/2 < block_size)
        {
          const int index = part_index + j + i * GROUP_SIZE * block_size;
          const int index2 = part_index2 + j + i * GROUP_SIZE/2 * block_size;
          tmp2 [index2] = conv_step (index, _hpf, tmp, block_size);          
        }
      }
    
      // for "odd" block sizes
      if (j + local_c1 < block_size)
      {
        int i;
        for (i = 0; i < block_size/GROUP_SIZE-1; i++)
        {
          const int index = part_index + j + i * GROUP_SIZE * block_size;
          const int index2 = part_index2 + j + i * GROUP_SIZE/2 * block_size;
          tmp2 [index2] = conv_step (index, _hpf, tmp, block_size);
        }
        if (i*GROUP_SIZE + local_c2 - GROUP_SIZE/2 < block_size)
        {
          const int index = part_index + j + i * GROUP_SIZE * block_size;
          const int index2 = part_index2 + j + i * GROUP_SIZE/2 * block_size;
          tmp2 [index2] = conv_step (index, _hpf, tmp, block_size);          
        }        
      }
    
    } // end of highpass filter

}


kernel void
perf_dwtFilter (__local A_type * loc_mem, __constant A_type * _lpf, __constant A_type * _hpf)
{

  const int local_c1 = get_local_id (0);
  const int local_c2 = get_local_id (1);

  __local A_type * tmp  = & loc_mem [0];
  __local A_type * tmp2 = & loc_mem [border_block_size * border_block_size];


  filter_columns (local_c1, local_c2, tmp, tmp2, _lpf, _hpf);
  barrier (CLK_LOCAL_MEM_FENCE);
  filter_rows (local_c1, local_c2, tmp2, tmp, _lpf, _hpf);

}


kernel void
perf_dwtGlobalToLocal (__local A_type * loc_mem, __global A_type * arg1)
{

  const int local_c1 = get_local_id (0);
  const int local_c2 = get_local_id (1);

  __local A_type * tmp  = & loc_mem [0];

  const int upper_left = get_group_id (1) * block_size * LINE_LENGTH
                       + get_group_id (0) * block_size
                       - offset * LINE_LENGTH
                       - offset;


  global2local (arg1, tmp, upper_left, local_c1, local_c2);

}



kernel void
perf_dwtLocalToGlobal (__local A_type * loc_mem, __global A_type * arg2)
{

  const int local_c1 = get_local_id (0);
  const int local_c2 = get_local_id (1);

  __local A_type * tmp  = & loc_mem [0];

  const int upper_left2 = get_group_id (1) * block_size / 2 * LINE_LENGTH
                        + get_group_id (0) * block_size / 2;


  local2global (tmp, arg2, upper_left2, local_c1, local_c2);

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

  __local A_type * tmp  = & loc_mem [0];
  __local A_type * tmp2 = & loc_mem [border_block_size * border_block_size];

  const int upper_left = get_group_id (1) * block_size * LINE_LENGTH
                       + get_group_id (0) * block_size
                       - offset * LINE_LENGTH
                       - offset;
  const int upper_left2 = get_group_id (1) * block_size / 2 * LINE_LENGTH
                        + get_group_id (0) * block_size / 2;

  /////////////////////////////
  // load block to local memory
  /////////////////////////////
  global2local (arg1, tmp, upper_left, local_c1, local_c2);

    
  barrier (CLK_LOCAL_MEM_FENCE); // local mem fence since work is performed on local memory !!!

  // filter operations
  filter_columns (local_c1, local_c2, tmp, tmp2, _lpf, _hpf);
  barrier (CLK_LOCAL_MEM_FENCE);
  filter_rows (local_c1, local_c2, tmp2, tmp, _lpf, _hpf);

  barrier (CLK_LOCAL_MEM_FENCE);

  //////////////////////////////
  // write back to global memory
  //////////////////////////////
  local2global (tmp, arg2, upper_left2, local_c1, local_c2);

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
  filter_columns (local_c1, local_c2, tmp, tmp2, _lpf, _hpf);

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