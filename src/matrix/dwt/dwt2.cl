// created on Aug 28, 2013


#pragma OPENCL EXTENSION cl_amd_printf: enable

//int printf(constant char * restrict format, ...);

//typedef float A_type;

# ifndef GROUP_SIZE_0
  # define GROUP_SIZE_0 128
# endif

# ifndef GROUP_SIZE_1
  # define GROUP_SIZE_1 128
# endif

# ifndef NUM_GROUPS_0
  # define NUM_GROUPS_0 4
# endif

# ifndef NUM_GROUPS_1
  # define NUM_GROUPS_1 4
# endif

# ifndef LDA
  # define LDA 512
# endif

# ifndef FL
  # define FL 4
# endif


# ifndef OFFSET
  __constant const int offset = FL-1;
  # define OFFSET
# endif


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
                     const int upper_left, const int local_c1, const int local_c2,
                     const int border_block_size_0,
                     const int border_block_size_1, __constant int * line_length)
{
  
  int j;
  for (j = 0; j < border_block_size_1-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < border_block_size_0-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left + (local_c2 + j) * LDA + local_c1 + i;
      if (upper_left + (local_c2 + j) * LDA + local_c1 + i < 0)
      {
        index = index + (index < -offset ? *line_length * LDA : 0)
                      + (index < 0 && index >= -offset ? LDA : 0);
        tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
            = arg1 [index];
        continue;
      }
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i] = arg1 [index];
    }
    if (i + local_c1 < border_block_size_0)
    {
      int index = upper_left + (local_c2 + j) * LDA + local_c1 + i;
      if (upper_left + (local_c2 + j) * LDA + local_c1 + i < 0)
      {
        index = index + (index < -offset ? *line_length * LDA : 0)
                      + (index < 0 && index >= -offset ? LDA : 0);
        tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
            = arg1 [index];
        continue;
      }
      else
      {
        tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i] = arg1 [index];
      }
    }
  }
  if (j + local_c2 < border_block_size_1)
  {
    int i;
    for (i = 0; i < border_block_size_0-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
          = arg1 [upper_left + (local_c2 + j) * LDA + local_c1 + i];
    }
    if (i + local_c1 < border_block_size_0)
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
          = arg1 [upper_left + (local_c2 + j) * LDA + local_c1 + i];
  }
}


void
local2global        (__local A_type * tmp2, __global A_type * arg2,
                     const int upper_left2, const int local_c1, const int local_c2,
                     const int block_size_0, const int block_size_1,
                     __constant int * line_length)
{
  
  ///////////
  // part: LL
  ///////////
  int j;
  for (j = 0; j < block_size_1/2-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
      arg2 [upper_left2 + (local_c2 + j) * LDA + local_c1 + i]
           = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
    if (i + local_c1 < block_size_0/2)
      arg2 [upper_left2 + (local_c2 + j) * LDA + local_c1 + i]
           = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
  }
  if (j + local_c2 < block_size_1/2)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
      arg2 [upper_left2 + (local_c2 + j) * LDA + local_c1 + i]
           = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
    if (i + local_c1 < block_size_0/2)
      arg2 [upper_left2 + (local_c2 + j) * LDA + local_c1 + i]
           = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
  }

  ///////////
  // part: LH
  ///////////
  for (j = block_size_1/2; j < block_size_1-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
      arg2 [upper_left2 + (*line_length/2 + local_c2 + j - block_size_1/2) * LDA + local_c1 + i] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
    if (i + local_c1 < block_size_0/2)
      arg2 [upper_left2 + (*line_length/2 + local_c2 + j - block_size_1/2) * LDA + local_c1 + i] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
  }
  if (j + local_c2 < block_size_1)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
      arg2 [upper_left2 + (*line_length/2 + local_c2 + j - block_size_1/2) * LDA + local_c1 + i] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
    if (i + local_c1 < block_size_0/2)
      arg2 [upper_left2 + (*line_length/2 + local_c2 + j - block_size_1/2) * LDA + local_c1 + i] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
  }

  ///////////
  // part: HL
  ///////////
  for (j = 0; j < block_size_1/2-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
      arg2 [upper_left2 + (local_c2 + j) * LDA + local_c1 + i + *line_length/2]
           = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i + block_size_0/2];
    if (i + local_c1 < block_size_0/2)
      arg2 [upper_left2 + (local_c2 + j) * LDA + local_c1 + i + *line_length/2]
           = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i + block_size_0/2];
  }
  if (j + local_c2 < block_size_1/2)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
      arg2 [upper_left2 + (local_c2 + j) * LDA + local_c1 + i + *line_length/2]
           = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i + block_size_0/2];
    if (i + local_c1 < block_size_0/2)
      arg2 [upper_left2 + (local_c2 + j) * LDA + local_c1 + i + *line_length/2]
           = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i + block_size_0/2];
  }

  ///////////
  // part: HH
  ///////////
  for (j = block_size_1/2; j < block_size_1-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
      arg2 [upper_left2 + (*line_length/2 + local_c2 + j - block_size_1/2) * LDA + local_c1 + i + *line_length/2] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i + block_size_0/2];
    if (i + local_c1 < block_size_0/2)
      arg2 [upper_left2 + (*line_length/2 + local_c2 + j - block_size_1/2) * LDA + local_c1 + i + *line_length/2] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i + block_size_0/2];
  }
  if (j + local_c2 < block_size_1)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
      arg2 [upper_left2 + (*line_length/2 + local_c2 + j - block_size_1/2) * LDA + local_c1 + i + *line_length/2] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i + block_size_0/2];
    if (i + local_c1 < block_size_0/2)
      arg2 [upper_left2 + (*line_length/2 + local_c2 + j - block_size_1/2) * LDA + local_c1 + i + *line_length/2] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i + block_size_0/2];
  }

}



void
filter_columns           (const int local_c1, const int local_c2,
                          __local A_type * tmp, __local A_type * tmp2,
                          __constant A_type * _lpf, __constant A_type * _hpf,
                          const int block_size_0, const int block_size_1,
                          const int border_block_size_0, const int border_block_size_1)
{

    // COLUMNS

    if (local_c1 < GROUP_SIZE_0/2)  // lowpass filter
    {

      // reused index parts
      const int part_index = local_c2 * border_block_size_0 + 2 * local_c1 + offset;
      const int part_index2 = local_c2 * block_size_0 + local_c1;

      // j: loop over columns per thread -> c2
      // i: loop over rows per thread -> c1
      int j;
      for (j = 0; j < border_block_size_1-GROUP_SIZE_1; j += GROUP_SIZE_1)
      {
        int i;
        for (i = 0; i < block_size_0/GROUP_SIZE_0-1; i++)
        {
          const int index = part_index + j * border_block_size_0 + i * GROUP_SIZE_0;
          const int index2 = part_index2 + j * block_size_0 + i * GROUP_SIZE_0/2;
          tmp2 [index2] = conv_step (index, _lpf, tmp, 1);
        }
        if (i*GROUP_SIZE_0 + local_c1 < block_size_0)
        {
          const int index = part_index + j * border_block_size_0 + i * GROUP_SIZE_0;
          const int index2 = part_index2 + j * block_size_0 + i * GROUP_SIZE_0/2;
          tmp2 [index2] = conv_step (index, _lpf, tmp, 1);
        }
      }
    
      // for "odd" block sizes
      if (j + local_c2 < border_block_size_1)
      {
        int i;
        for (i = 0; i < block_size_0/GROUP_SIZE_0-1; i++)
        {
          const int index = part_index + j * border_block_size_0 + i * GROUP_SIZE_0;
          const int index2 = part_index2 + j * block_size_0 + i * GROUP_SIZE_0/2;
          tmp2 [index2] = conv_step (index, _lpf, tmp, 1);
        }
        if (i * GROUP_SIZE_0 + local_c1 < block_size_0)
        {
          const int index = part_index + j * border_block_size_0 + i * GROUP_SIZE_0;
          const int index2 = part_index2 + j * block_size_0 + i * GROUP_SIZE_0/2;
          tmp2 [index2] = conv_step (index, _lpf, tmp, 1);
        }
      }
    
    } // end of lowpass filter
    
    else  // highpass filter
    { 
        
      // reused index parts
      const int part_index = local_c2 * border_block_size_0 + 2 * (local_c1 - GROUP_SIZE_0/2) + 1 + offset;
      const int part_index2 = local_c2 * block_size_0 + local_c1 - GROUP_SIZE_0/2 + block_size_0/2;
        
      // j: loop over columns per thread
      // i: loop over elements per column per thread
      int j;
      for (j = 0; j < border_block_size_1-GROUP_SIZE_1; j += GROUP_SIZE_1)
      {
        int i;
        for (i = 0; i < block_size_0/GROUP_SIZE_0-1; i++)
        {
          const int index = part_index + j * border_block_size_0 + i * GROUP_SIZE_0;
          const int index2 = part_index2 + j * block_size_0 + i * GROUP_SIZE_0/2;
          tmp2 [index2] = conv_step (index, _hpf, tmp, 1);
        }
        if (i*GROUP_SIZE_0 + local_c1 - GROUP_SIZE_0/2 < block_size_0)
        {
          const int index = part_index + j * border_block_size_0 + i * GROUP_SIZE_0;
          const int index2 = part_index2 + j * block_size_0 + i * GROUP_SIZE_0/2;
          tmp2 [index2] = conv_step (index, _hpf, tmp, 1);          
        }
      }
    
      // for "odd" block sizes
      if (j + local_c2 < border_block_size_1)
      {
        int i;
        for (i = 0; i < block_size_0/GROUP_SIZE_0-1; i++)
        {
          const int index = part_index + j * border_block_size_0 + i * GROUP_SIZE_0;
          const int index2 = part_index2 + j * block_size_0 + i * GROUP_SIZE_0/2;
          tmp2 [index2] = conv_step (index, _hpf, tmp, 1);
        }
        if (i*GROUP_SIZE_0 + local_c1 - GROUP_SIZE_0/2 < block_size_0)
        {
          const int index = part_index + j * border_block_size_0 + i * GROUP_SIZE_0;
          const int index2 = part_index2 + j * block_size_0 + i * GROUP_SIZE_0/2;
          tmp2 [index2] = conv_step (index, _hpf, tmp, 1);
        }        
      }
    
    } // end of highpass filter

}



void
filter_rows           (const int local_c1, const int local_c2,
                       __local A_type * tmp, __local A_type * tmp2,
                       __constant A_type * _lpf, __constant A_type * _hpf,
                       const int block_size_0, const int block_size_1)
{

    // ROWS

    if (local_c2 < GROUP_SIZE_1/2)  // lowpass filter
    {

      // reused index parts
      const int part_index = local_c1 + (offset + 2 * local_c2) * block_size_0;
      const int part_index2 = local_c1 + local_c2 * block_size_0;

      // j: loop over rows per thread -> c1
      // i: loop over elements per row per thread -> c2
      int j;
      for (j = 0; j < block_size_0-GROUP_SIZE_0; j += GROUP_SIZE_0)
      {
        int i;
        for (i = 0; i < block_size_1/GROUP_SIZE_1-1; i++)
        {
          const int index = part_index + j + i * GROUP_SIZE_1 * block_size_0;
          const int index2 = part_index2 + j + i * GROUP_SIZE_1/2 * block_size_0;
          tmp2 [index2] = conv_step (index, _lpf, tmp, block_size_0);
        }
        if (i*GROUP_SIZE_1 + local_c2 < block_size_1)
        {
          const int index = part_index + j + i * GROUP_SIZE_1 * block_size_0;
          const int index2 = part_index2 + j + i * GROUP_SIZE_1/2 * block_size_0;
          tmp2 [index2] = conv_step (index, _lpf, tmp, block_size_0);
        }
      }
    
      // for "odd" block sizes
      if (j + local_c1 < block_size_0)
      {
        int i;
        for (i = 0; i < block_size_1/GROUP_SIZE_1-1; i++)
        {
          const int index = part_index + j + i * GROUP_SIZE_1 * block_size_0;
          const int index2 = part_index2 + j + i * GROUP_SIZE_1/2 * block_size_0;
          tmp2 [index2] = conv_step (index, _lpf, tmp, block_size_0);
        }
        if (i * GROUP_SIZE_1 + local_c2 < block_size_1)
        {
          const int index = part_index + j + i * GROUP_SIZE_1 * block_size_0;
          const int index2 = part_index2 + j + i * GROUP_SIZE_1/2 * block_size_0;
          tmp2 [index2] = conv_step (index, _lpf, tmp, block_size_0);
        }
      }
    
    } // end of lowpass filter
    
    else  // highpass filter
    { 
        
      // reused index parts
      const int part_index = local_c1 + (offset + 2 * (local_c2 - GROUP_SIZE_1/2) + 1) * block_size_0;
      const int part_index2 = local_c1 + (local_c2 - GROUP_SIZE_1/2 + block_size_1/2) * block_size_0;
        
      // j: loop over rows per thread -> c1
      // i: loop over elements per row per thread -> c2
      int j;
      for (j = 0; j < block_size_0-GROUP_SIZE_0; j += GROUP_SIZE_0)
      {
        int i;
        for (i = 0; i < block_size_1/GROUP_SIZE_1-1; i++)
        {
          const int index = part_index + j + i * GROUP_SIZE_1 * block_size_0;
          const int index2 = part_index2 + j + i * GROUP_SIZE_1/2 * block_size_0;
          tmp2 [index2] = conv_step (index, _hpf, tmp, block_size_0);
        }
        if (i*GROUP_SIZE_1 + local_c2 - GROUP_SIZE_1/2 < block_size_1)
        {
          const int index = part_index + j + i * GROUP_SIZE_1 * block_size_0;
          const int index2 = part_index2 + j + i * GROUP_SIZE_1/2 * block_size_0;
          tmp2 [index2] = conv_step (index, _hpf, tmp, block_size_0);
        }
      }
    
      // for "odd" block sizes
      if (j + local_c1 < block_size_0)
      {
        int i;
        for (i = 0; i < block_size_1/GROUP_SIZE_1-1; i++)
        {
          const int index = part_index + j + i * GROUP_SIZE_1 * block_size_0;
          const int index2 = part_index2 + j + i * GROUP_SIZE_1/2 * block_size_0;
          tmp2 [index2] = conv_step (index, _hpf, tmp, block_size_0);
        }
        if (i*GROUP_SIZE_1 + local_c2 - GROUP_SIZE_1/2 < block_size_1)
        {
          const int index = part_index + j + i * GROUP_SIZE_1 * block_size_0;
          const int index2 = part_index2 + j + i * GROUP_SIZE_1/2 * block_size_0;
          tmp2 [index2] = conv_step (index, _hpf, tmp, block_size_0);
        }        
      }
    
    } // end of highpass filter

}


kernel void
perf_dwtFilter (__local A_type * loc_mem, __constant A_type * _lpf, __constant A_type * _hpf, __constant int * line_length)
{

  const int block_size_0 = *line_length / NUM_GROUPS_0;
  const int block_size_1 = *line_length / NUM_GROUPS_1;
  const int border_block_size_0 = *line_length / NUM_GROUPS_0 + offset;
  const int border_block_size_1 = *line_length / NUM_GROUPS_1 + offset;

  const int local_c1 = get_local_id (0);
  const int local_c2 = get_local_id (1);

  __local A_type * tmp  = & loc_mem [0];
  __local A_type * tmp2 = & loc_mem [border_block_size_0 * border_block_size_1];


  filter_columns (local_c1, local_c2, tmp, tmp2, _lpf, _hpf, block_size_0, block_size_1, border_block_size_0, border_block_size_1);
  barrier (CLK_LOCAL_MEM_FENCE);
  filter_rows (local_c1, local_c2, tmp2, tmp, _lpf, _hpf, block_size_0, block_size_1);

}


kernel void
perf_dwtGlobalToLocal (__local A_type * loc_mem, __global A_type * arg1, __constant int * line_length)
{

  const int block_size_0 = *line_length / NUM_GROUPS_0;
  const int block_size_1 = *line_length / NUM_GROUPS_1;
  const int border_block_size_0 = *line_length / NUM_GROUPS_0 + offset;
  const int border_block_size_1 = *line_length / NUM_GROUPS_1 + offset;

  const int local_c1 = get_local_id (0);
  const int local_c2 = get_local_id (1);

  __local A_type * tmp  = & loc_mem [0];

  const int upper_left = get_group_id (1) * block_size_1 * LDA
                       + get_group_id (0) * block_size_0
                       - offset * LDA
                       - offset;


  global2local (arg1, tmp, upper_left, local_c1, local_c2, border_block_size_0, border_block_size_1, line_length);

}



kernel void
perf_dwtLocalToGlobal (__local A_type * loc_mem, __global A_type * arg2, __constant int * line_length)
{

  const int block_size_0 = *line_length / NUM_GROUPS_0;
  const int block_size_1 = *line_length / NUM_GROUPS_1;

  const int local_c1 = get_local_id (0);
  const int local_c2 = get_local_id (1);

  __local A_type * tmp  = & loc_mem [0];

  const int upper_left2 = get_group_id (1) * block_size_1 / 2 * LDA
                        + get_group_id (0) * block_size_0 / 2;


  local2global (tmp, arg2, upper_left2, local_c1, local_c2, block_size_0, block_size_1, line_length);

}



kernel void dwt2_final (__global A_type * arg1,
                        __global A_type * arg2,
                        __constant int * n,
                        __global int * m,
                        __global int * k,
                        __constant int * line_length,
                        __constant int * num_levels)
{

  int block_size_0 = *line_length / (min (*line_length, (int) get_global_size (0))/GROUP_SIZE_0);
  int block_size_0_alt = (2 * *line_length) / (min (*line_length, (int) get_global_size (0))/GROUP_SIZE_0);
  int block_size_1 = *line_length / (min (*line_length, (int) get_global_size (1))/GROUP_SIZE_1);

  const int local_c1 = get_local_id (0);
  const int local_c2 = get_local_id (1);

  int upper_left  = get_group_id (1) * block_size_1 * LDA
                  + get_group_id (0) * block_size_0;
  int upper_left2 = get_group_id (1) * block_size_1 * LDA
                  + get_group_id (0) * block_size_0_alt;

  int l = 1;
  int current_line_length = *line_length;

  if (((*num_levels) & 1) == 0)
  {

  if (get_global_id (0) < current_line_length
    && get_global_id (1) < current_line_length)
  {

    // copy upper left corner
    int j = 0;
    for (; j < block_size_1 - GROUP_SIZE_1; j += GROUP_SIZE_1)
    {
      int i = 0;
      for (; i < block_size_0 - GROUP_SIZE_0; i += GROUP_SIZE_0)
      {
        int index = upper_left + (j + local_c2) * LDA + i + local_c1;
        arg2 [index] = arg1 [index];
      }
      if (i + local_c1 < block_size_0)
      {
        int index = upper_left + (j + local_c2) * LDA + i + local_c1;
        arg2 [index] = arg1 [index];
      }
    }
    if (j + local_c2 < block_size_1)
    {
      int i = 0;
      for (; i < block_size_0 - GROUP_SIZE_0; i += GROUP_SIZE_0)
      {
        int index = upper_left + (j + local_c2) * LDA + i + local_c1;
        arg2 [index] = arg1 [index];
      }
      if (i + local_c1 < block_size_0)
      {
        int index = upper_left + (j + local_c2) * LDA + i + local_c1;
        arg2 [index] = arg1 [index];
      }
    }
    
  }

  l += 2;
  current_line_length *= 2;

  }

  // loop over levels
  for (; l < *num_levels; l += 2)
  {
    
    if (get_global_id (1) < current_line_length
      && get_global_id (0) < current_line_length)
    {

    block_size_0 = current_line_length / (min (current_line_length, (int) get_global_size (0))/GROUP_SIZE_0);
    block_size_0_alt = (2 * current_line_length) / (min (current_line_length, (int) get_global_size (0))/GROUP_SIZE_0);
    block_size_1 = current_line_length / (min (current_line_length, (int) get_global_size (1))/GROUP_SIZE_1);
    
    upper_left  = get_group_id (1) * block_size_1 * LDA
                + get_group_id (0) * block_size_0;
    upper_left2 = get_group_id (1) * block_size_1 * LDA
                + get_group_id (0) * block_size_0_alt;
    
    // copy bottom left corner
    int j = 0;
    for (; j < block_size_1 - GROUP_SIZE_1; j += GROUP_SIZE_1)
    {
      int i = 0;
      for (; i < block_size_0 - GROUP_SIZE_0; i += GROUP_SIZE_0)
      {
        int index = upper_left + current_line_length + (j + local_c2) * LDA + i + local_c1;
        arg2 [index] = arg1 [index];
      }
      if (i + local_c1 < block_size_0)
      {
        int index = upper_left + current_line_length + (j + local_c2) * LDA + i + local_c1;
        arg2 [index] = arg1 [index];
      }
    }
    if (j + local_c2 < block_size_1)
    {
      int i = 0;
      for (; i < block_size_0 - GROUP_SIZE_0; i += GROUP_SIZE_0)
      {
        int index = upper_left + current_line_length + (j + local_c2) * LDA + i + local_c1;
        arg2 [index] = arg1 [index];
      }
      if (i + local_c1 < block_size_0)
      {
        int index = upper_left + current_line_length + (j + local_c2) * LDA + i + local_c1;
        arg2 [index] = arg1 [index];
      }
    }
    
    // copy parts on the right
    for (j = 0; j < block_size_1 - GROUP_SIZE_1; j += GROUP_SIZE_1)
    {
      int i = 0;
      for (; i < block_size_0_alt - GROUP_SIZE_0; i += GROUP_SIZE_0)
      {
        int index = upper_left2 + current_line_length*LDA + (j + local_c2) * LDA + i + local_c1;
        arg2 [index] = arg1 [index];
      }
      if (i + local_c1 < block_size_0_alt)
      {
        int index = upper_left2 + current_line_length*LDA + (j + local_c2) * LDA + i + local_c1;
        arg2 [index] = arg1 [index];
      }
    }
    if (j + local_c2 < block_size_1)
    {
      int i = 0;
      for (; i < block_size_0_alt - GROUP_SIZE_0; i += GROUP_SIZE_0)
      {
        int index = upper_left2 + current_line_length*LDA + (j + local_c2) * LDA + i + local_c1;
        arg2 [index] = arg1 [index];
      }
      if (i + local_c1 < block_size_0_alt)
      {
        int index = upper_left2 + current_line_length*LDA + (j + local_c2) * LDA + i + local_c1;
        arg2 [index] = arg1 [index];
      }
    }
    
    }
    
    current_line_length *= 4;
  }

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
          __constant int * line_length,
          __global int * loc_mem_size)
{

  if (get_global_id (0) < *line_length
    && get_global_id (1) < *line_length)
  {

  const int block_size_0 = *line_length / (min (*line_length, (int) get_global_size (0))/GROUP_SIZE_0);
  const int block_size_1 = *line_length / (min (*line_length, (int) get_global_size (1))/GROUP_SIZE_1);
  const int border_block_size_0 = block_size_0 + offset;
  const int border_block_size_1 = block_size_1 + offset;

  const int local_c1 = get_local_id (0);
  const int local_c2 = get_local_id (1);

  __local A_type * tmp  = & loc_mem [0];
  __local A_type * tmp2 = & loc_mem [border_block_size_0 * border_block_size_1];

  const int upper_left = get_group_id (1) * block_size_1 * LDA
                       + get_group_id (0) * block_size_0
                       - offset * LDA
                       - offset;
  const int upper_left2 = get_group_id (1) * block_size_1 / 2 * LDA
                        + get_group_id (0) * block_size_0 / 2;

  /////////////////////////////
  // load block to local memory
  /////////////////////////////
  global2local (arg1, tmp, upper_left, local_c1, local_c2, border_block_size_0, border_block_size_1, line_length);

    
  barrier (CLK_LOCAL_MEM_FENCE); // local mem fence since work is performed on local memory !!!

  // filter operations
  filter_columns (local_c1, local_c2, tmp, tmp2, _lpf, _hpf, block_size_0, block_size_1, border_block_size_0, border_block_size_1);
  barrier (CLK_LOCAL_MEM_FENCE);
  filter_rows (local_c1, local_c2, tmp2, tmp, _lpf, _hpf, block_size_0, block_size_1);

  barrier (CLK_LOCAL_MEM_FENCE);

  //////////////////////////////
  // write back to global memory
  //////////////////////////////
  local2global (tmp, arg2, upper_left2, local_c1, local_c2, block_size_0, block_size_1, line_length);

  }

}








//kernel void idwt2 (__global A_type * arg1,
//          __constant A_type * _lpf,
//          __constant A_type * _hpf,
//          __global A_type * arg2,
//          __local A_type * loc_mem,
//          __constant int * n,
//          __global int * m,
//          __global int * k,
//          __global int * fl,
//          __global int * loc_mem_size)
//{
//
//  const int local_c1 = get_local_id (0);
//  const int local_c2 = get_local_id (1);
//
//  __local A_type * tmp  = & loc_mem [0];
//  __local A_type * tmp2 = & loc_mem [border_block_size * border_block_size];
//
//  const int upper_left = get_group_id (1) * block_size * LINE_LENGTH
//                       + get_group_id (0) * block_size
//                       - offset * LINE_LENGTH
//                       - offset;
//  const int upper_left2 = get_group_id (1) * block_size * LINE_LENGTH
//                        + get_group_id (0) * block_size / 2;
//
//  /////////////////////////////
//  // load block to local memory
//  /////////////////////////////
//
//  int j;
//  for (j = 0; j < border_block_size-GROUP_SIZE; j += GROUP_SIZE)
//  {
//    int i;
//    for (i = 0; i < border_block_size-GROUP_SIZE; i += GROUP_SIZE)
//    {
////      if (upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i < 0)
////      {
////        tmp [(local_c2 + j) * border_block_size + local_c1 + i]
////            = arg1 [upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i];
////        continue;
////      }
//      tmp [(local_c2 + j) * border_block_size + local_c1 + i]
//          = arg1 [upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i];
//    }
//    if (i + local_c1 < border_block_size)
//      tmp [(local_c2 + j) * border_block_size + local_c1 + i]
//          = arg1 [upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i];
//  }
//  if (j + local_c2 < border_block_size)
//  {
//    int i;
//    for (i = 0; i < border_block_size-GROUP_SIZE; i += GROUP_SIZE)
//    {
////      if (upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i < 0)
////      {
////        tmp [(local_c2 + j) * border_block_size + local_c1 + i]
////            = arg1 [upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i];
////        continue;
////      }
//      tmp [(local_c2 + j) * border_block_size + local_c1 + i]
//          = arg1 [upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i];
//    }
//    if (i + local_c1 < border_block_size)
//      tmp [(local_c2 + j) * border_block_size + local_c1 + i]
//          = arg1 [upper_left + (local_c2 + j) * LINE_LENGTH + local_c1 + i];
//  }
//    
//  barrier (CLK_LOCAL_MEM_FENCE);
//
//  // filter operations
//  filter_columns (local_c1, local_c2, tmp, tmp2, _lpf, _hpf);
//
//  barrier (CLK_LOCAL_MEM_FENCE);
//
//  //////////////////////////////
//  // write back to global memory
//  //////////////////////////////
//
//  // lowpass part
//  for (j = 0; j < block_size-GROUP_SIZE; j += GROUP_SIZE)
//  {
//    int i;
//    for (i = 0; i < block_size/2-GROUP_SIZE; i += GROUP_SIZE)
//      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i]
//           = tmp2 [(local_c2 + j) * block_size + local_c1 + i];
//    if (i + local_c1 < block_size/2)
//      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i]
//           = tmp2 [(local_c2 + j) * block_size + local_c1 + i];
//  }
//  if (j + local_c2 < block_size)
//  {
//    int i;
//    for (i = 0; i < block_size/2-GROUP_SIZE; i += GROUP_SIZE)
//      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i]
//           = tmp2 [(local_c2 + j) * block_size + local_c1 + i];
//    if (i + local_c1 < block_size/2)
//      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i]
//           = tmp2 [(local_c2 + j) * block_size + local_c1 + i];
//  }
//
//
//  // highpass part
//  for (j = 0; j < block_size-GROUP_SIZE; j += GROUP_SIZE)
//  {
//    int i;
//    for (i = 0; i < block_size/2-GROUP_SIZE; i += GROUP_SIZE)
//      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i + LINE_LENGTH/2]
//           = tmp2 [(local_c2 + j) * block_size + local_c1 + i + block_size/2];
//    if (i + local_c1 < block_size/2)
//      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i + LINE_LENGTH/2]
//           = tmp2 [(local_c2 + j) * block_size + local_c1 + i + block_size/2];
//  }
//  if (j + local_c2 < block_size)
//  {
//    int i;
//    for (i = 0; i < block_size/2-GROUP_SIZE; i += GROUP_SIZE)
//      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i + LINE_LENGTH/2]
//           = tmp2 [(local_c2 + j) * block_size + local_c1 + i + block_size/2];
//    if (i + local_c1 < block_size/2)
//      arg2 [upper_left2 + (local_c2 + j) * LINE_LENGTH + local_c1 + i + LINE_LENGTH/2]
//           = tmp2 [(local_c2 + j) * block_size + local_c1 + i + block_size/2];
//  }
//
//}