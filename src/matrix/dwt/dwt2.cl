// created on Aug 28, 2013


#pragma OPENCL EXTENSION cl_amd_printf : enable

//int printf(const constant char * restrict format, ...);

//typedef float A_type;

# ifndef GROUP_SIZE_0
  # define GROUP_SIZE_0 16
# endif

# ifndef GROUP_SIZE_1
  # define GROUP_SIZE_1 8
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

# ifndef LDB
  # define LDB 512
# endif

# ifndef FL
  # define FL 4
# endif

# ifndef ODD_FILTER
  # define ODD_FILTER 1
# endif


# ifndef OFFSET
  __constant const int offset = FL-2;
  # define OFFSET
# endif


A_type
conv_step_hi        (const int index, __constant A_type * _filter, __local A_type * tmp,
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


A_type
conv_step_lo        (const int index, __constant A_type * _filter, __local A_type * tmp,
                     const int increment)
{
  A_type sum = 0;
  # pragma unroll
  for (int k = FL-1; k >= 0; k--)
  {
    sum += tmp [index + k * increment] * _filter [k];
  }
  return sum;
}


void
global2local        (__global A_type * arg1, __local A_type * tmp,
                     const int upper_left, const int local_c1, const int local_c2,
                     const int border_block_size_0,
                     const int border_block_size_1, __constant int * line_length)
{
  
  const int col_shift = *line_length;
  const int row_shift = *line_length * LDA;

  const int c1_base = get_group_id (0) * (border_block_size_0-offset) + local_c1 - offset;
  const int c2_base = get_group_id (1) * (border_block_size_1-offset) + local_c2 - offset;

  const int index_base = upper_left + local_c2 * LDA + local_c1;
  const int local_index_base = local_c2 * border_block_size_0 + local_c1;

  int j;
  for (j = 0; j < border_block_size_1-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < border_block_size_0-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      const int index = index_base + j * LDA + i
                      + (c1_base + i < 0 ? col_shift : 0)
                      + (c2_base + j < 0 ? row_shift : 0);
      tmp [local_index_base + j * border_block_size_0 + i] = arg1 [index];
    }
    if (i + local_c1 < border_block_size_0)
    {
      const int index = index_base + j * LDA + i
                      + (c1_base + i < 0 ? col_shift : 0)
                      + (c2_base + j < 0 ? row_shift : 0);
      tmp [local_index_base + j * border_block_size_0 + i] = arg1 [index];
    }
  }
  if (j + local_c2 < border_block_size_1)
  {
    int i;
    for (i = 0; i < border_block_size_0-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      const int index = index_base + j * LDA + i
                      + (c1_base + i < 0 ? col_shift : 0)
                      + (c2_base + j < 0 ? row_shift : 0);
      tmp [local_index_base + j * border_block_size_0 + i] = arg1 [index];
    }
    if (i + local_c1 < border_block_size_0)
    {
      const int index = index_base + j * LDA + i
                      + (c1_base + i < 0 ? col_shift : 0)
                      + (c2_base + j < 0 ? row_shift : 0);
      tmp [local_index_base + j * border_block_size_0 + i] = arg1 [index];
    }
  }
}


void
global2local__        (__global A_type * arg1, __local A_type * tmp,
                       const int upper_left, const int local_c1, const int local_c2,
                       const int border_block_size_0,
                       const int border_block_size_1, __constant int * line_length)
{
  
  const int c1_base = get_group_id (0) * (border_block_size_0-offset) + local_c1 - offset;
  const int c2_base = get_group_id (1) * (border_block_size_1-offset) + local_c2 - offset;

  int j;
  for (j = 0; j < border_block_size_1-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < border_block_size_0-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left + (local_c2 + j) * LDA + local_c1 + i;
//      index = index + (c1_base + i < 0 ? *line_length : 0)
//                    + (c2_base + j < 0 ? *line_length * LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
            = arg1 [index];
    }
    if (i + local_c1 < border_block_size_0)
    {
      int index = upper_left + (local_c2 + j) * LDA + local_c1 + i;
//      index = index + (c1_base + i < 0 ? *line_length : 0)
//                    + (c2_base + j < 0 ? *line_length * LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
            = arg1 [index];
    }
  }
  if (j + local_c2 < border_block_size_1)
  {
    int i;
    for (i = 0; i < border_block_size_0-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left + (local_c2 + j) * LDA + local_c1 + i;
//      index = index + (c1_base + i < 0 ? *line_length : 0)
//                    + (c2_base + j < 0 ? *line_length * LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
            = arg1 [index];
    }
    if (i + local_c1 < border_block_size_0)
    {
      int index = upper_left + (local_c2 + j) * LDA + local_c1 + i;
//      index = index + (c1_base + i < 0 ? *line_length : 0)
//                    + (c2_base + j < 0 ? *line_length * LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
            = arg1 [index];
    }
  }
}


void
local2global        (__local A_type * tmp2, __global A_type * arg2,
                     const int upper_left2, const int local_c1, const int local_c2,
                     const int block_size_0, const int block_size_1,
                     __constant int * line_length)
{
 
  const int shift = - offset/2;
  const int row_shift = *line_length/2 * LDA;
  const int col_shift = *line_length/2;

  const int c1_base = get_group_id (0) * block_size_0/2 + local_c1 + shift;
  const int c2_base = get_group_id (1) * block_size_1/2 + local_c2 + shift;

  const int index_base = upper_left2 + local_c2 * LDA + local_c1;
  const int local_index_base = local_c2 * block_size_0 + local_c1;

  ///////////
  // part: LL
  ///////////
  const int index_base_ll = index_base + shift * (LDA + 1);
  int j;
  for (j = 0; j < block_size_1/2-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      const int index = index_base_ll + j * LDA + i
                      + (c1_base + i < 0 ? col_shift : 0)
                      + (c2_base + j < 0 ? row_shift : 0);
      arg2 [index] = tmp2 [local_index_base + j * block_size_0 + i];
    }
    if (i + local_c1 < block_size_0/2)
    {
      const int index = index_base_ll + j * LDA + i
                      + (c1_base + i < 0 ? col_shift : 0)
                      + (c2_base + j < 0 ? row_shift : 0);
      arg2 [index] = tmp2 [local_index_base + j * block_size_0 + i];
    }
  }
  if (j + local_c2 < block_size_1/2)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      const int index = index_base_ll + j * LDA + i
                      + (c1_base + i < 0 ? col_shift : 0)
                      + (c2_base + j < 0 ? row_shift : 0);
      arg2 [index] = tmp2 [local_index_base + j * block_size_0 + i];
    }
    if (i + local_c1 < block_size_0/2)
    {
      const int index = index_base_ll + j * LDA + i
                      + (c1_base + i < 0 ? col_shift : 0)
                      + (c2_base + j < 0 ? row_shift : 0);
      arg2 [index] = tmp2 [local_index_base + j * block_size_0 + i];
    }
  }

  ///////////
  // part: LH
  ///////////
  const int index_base_lh = index_base + (*line_length/2 - block_size_1/2) * LDA + shift;
  for (j = block_size_1/2; j < block_size_1-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      const int index = index_base_lh + j * LDA + i
                      + (c1_base + i < 0 ? col_shift : 0)
                      + (c2_base + j -shift- block_size_1/2 < 0 ? row_shift : 0);
      arg2 [index] = tmp2 [local_index_base + j * block_size_0 + i];
    }
    if (i + local_c1 < block_size_0/2)
    {
      const int index = index_base_lh + j * LDA + i
                      + (c1_base + i < 0 ? col_shift : 0)
                      + (c2_base + j -shift- block_size_1/2 < 0 ? row_shift : 0);
      arg2 [index] = tmp2 [local_index_base + j * block_size_0 + i];
    }
  }
  if (j + local_c2 < block_size_1)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      const int index = index_base_lh + j * LDA + i
                      + (c1_base + i < 0 ? col_shift : 0)
                      + (c2_base + j -shift- block_size_1/2 < 0 ? row_shift : 0);
      arg2 [index] = tmp2 [local_index_base + j * block_size_0 + i];
    }
    if (i + local_c1 < block_size_0/2)
    {
      const int index = index_base_lh + j * LDA + i
                      + (c1_base + i < 0 ? col_shift : 0)
                      + (c2_base + j -shift- block_size_1/2 < 0 ? row_shift : 0);
      arg2 [index] = tmp2 [local_index_base + j * block_size_0 + i];
    }
  }

  ///////////
  // part: HL
  ///////////
  const int index_base_hl = index_base + shift * LDA + *line_length/2 - block_size_0/2;
  for (j = 0; j < block_size_1/2-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = block_size_0/2; i < block_size_0-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      const int index = index_base_hl + j * LDA + i
                      + (c1_base + i - block_size_0/2 -shift < 0 ? col_shift : 0)
                      + (c2_base + j < 0 ? row_shift : 0);
      arg2 [index] = tmp2 [local_index_base + j * block_size_0 + i];
    }
    if (i + local_c1 < block_size_0)
    {
      const int index = index_base_hl + j * LDA + i
                      + (c1_base + i - block_size_0/2 -shift < 0 ? col_shift : 0)
                      + (c2_base + j < 0 ? row_shift : 0);
      arg2 [index] = tmp2 [local_index_base + j * block_size_0 + i];
    }
  }
  if (j + local_c2 < block_size_1/2)
  {
    int i;
    for (i = block_size_0/2; i < block_size_0-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      const int index = index_base_hl + j * LDA + i
                      + (c1_base + i -block_size_0/2 -shift < 0 ? col_shift : 0)
                      + (c2_base + j < 0 ? row_shift : 0);
      arg2 [index] = tmp2 [local_index_base + j * block_size_0 + i];
    }
    if (i + local_c1 < block_size_0)
    {
      const int index = index_base_hl + j * LDA + i
                      + (c1_base + i -block_size_0/2 -shift < 0 ? col_shift : 0)
                      + (c2_base + j < 0 ? row_shift : 0);
      arg2 [index] = tmp2 [local_index_base + j * block_size_0 + i];
    }
  }

  ///////////
  // part: HH
  ///////////
  const int index_base_hh = index_base + (*line_length/2 - block_size_1/2) * LDA + *line_length/2 - block_size_0/2;
  for (j = block_size_1/2; j < block_size_1-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = block_size_0/2; i < block_size_0-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      const int index = index_base_hh + j * LDA + i
                      + (c1_base + i -block_size_0/2 -shift < 0 ? col_shift : 0)
                      + (c2_base + j -shift - block_size_1/2 < 0 ? row_shift : 0);
      arg2 [index] = tmp2 [local_index_base + j * block_size_0 + i];
    }
    if (i + local_c1 < block_size_0)
    {
      const int index = index_base_hh + j * LDA + i
                      + (c1_base + i -block_size_0/2 -shift < 0 ? col_shift : 0)
                      + (c2_base + j -shift - block_size_1/2 < 0 ? row_shift : 0);
      arg2 [index] = tmp2 [local_index_base + j * block_size_0 + i];
    }
  }
  if (j + local_c2 < block_size_1)
  {
    int i;
    for (i = block_size_0/2; i < block_size_0-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      const int index = index_base_hh + j * LDA + i
                      + (c1_base + i -block_size_0/2 -shift < 0 ? col_shift : 0)
                      + (c2_base + j -shift - block_size_1/2 < 0 ? row_shift : 0);
      arg2 [index] = tmp2 [local_index_base + j * block_size_0 + i];
    }
    if (i + local_c1 < block_size_0)
    {
      const int index = index_base_hh + j * LDA + i
                      + (c1_base + i -block_size_0/2 -shift < 0 ? col_shift : 0)
                      + (c2_base + j -shift - block_size_1/2 < 0 ? row_shift : 0);
      arg2 [index] = tmp2 [local_index_base + j * block_size_0 + i];
    }
  }

}



void
local2global__        (__local A_type * tmp2, __global A_type * arg2,
                     const int upper_left2, const int local_c1, const int local_c2,
                     const int block_size_0, const int block_size_1,
                     __constant int * line_length)
{
 
  const int shift = - offset/2;

  const int c1_base = get_group_id (0) * block_size_0/2 + local_c1 + shift;
  const int c2_base = get_group_id (1) * block_size_1/2 + local_c2 + shift;

  ///////////
  // part: LL
  ///////////
  int j;
  for (j = 0; j < block_size_1/2-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left2 + (local_c2 + j + shift) * LDA + local_c1 + i + shift;
//      index = index + (c1_base + i < 0 ? *line_length/2 : 0)
//                    + (c2_base + j < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
    }
    if (i + local_c1 < block_size_0/2)
    {
      int index = upper_left2 + (local_c2 + j + shift) * LDA + local_c1 + i + shift;
//      index = index + (c1_base + i < 0 ? *line_length/2 : 0)
//                    + (c2_base + j < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
    }
  }
  if (j + local_c2 < block_size_1/2)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left2 + (local_c2 + j + shift) * LDA + local_c1 + i + shift;
//      index = index + (c1_base + i < 0 ? *line_length/2 : 0)
//                    + (c2_base + j < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
    }
    if (i + local_c1 < block_size_0/2)
    {
      int index = upper_left2 + (local_c2 + j + shift) * LDA + local_c1 + i + shift;
//      index = index + (c1_base + i < 0 ? *line_length/2 : 0)
//                    + (c2_base + j < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
    }
  }

  ///////////
  // part: LH
  ///////////
  for (j = block_size_1/2; j < block_size_1-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left2 + (*line_length/2 + local_c2 + j - block_size_1/2 ) * LDA + local_c1 + i + shift;
//      index = index + (c1_base + i < 0 ? *line_length/2 : 0)
//                    + (c2_base + j -shift- block_size_1/2 < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
    }
    if (i + local_c1 < block_size_0/2)
    {
      int index = upper_left2 + (*line_length/2 + local_c2 + j - block_size_1/2) * LDA + local_c1 + i + shift;
//      index = index + (c1_base + i < 0 ? *line_length/2 : 0)
//                    + (c2_base + j -shift- block_size_1/2 < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
    }
  }
  if (j + local_c2 < block_size_1)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left2 + (*line_length/2 + local_c2 + j - block_size_1/2) * LDA + local_c1 + i + shift;
//      index = index + (c1_base + i < 0 ? *line_length/2 : 0)
//                    + (c2_base + j -shift- block_size_1/2 < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
    }
    if (i + local_c1 < block_size_0/2)
    {
      int index = upper_left2 + (*line_length/2 + local_c2 + j - block_size_1/2) * LDA + local_c1 + i + shift;
//      index = index + (c1_base + i < 0 ? *line_length/2 : 0)
//                    + (c2_base + j -shift - block_size_1/2 < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
    }
  }

  ///////////
  // part: HL
  ///////////
  for (j = 0; j < block_size_1/2-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left2 + (local_c2 + j + shift) * LDA + local_c1 + i + *line_length/2;
//      index = index + (c1_base + i -shift < 0 ? *line_length/2 : 0)
//                    + (c2_base + j < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i + block_size_0/2];
    }
    if (i + local_c1 < block_size_0/2)
    {
      int index = upper_left2 + (local_c2 + j + shift) * LDA + local_c1 + i + *line_length/2;
//      index = index + (c1_base + i -shift < 0 ? *line_length/2 : 0)
//                    + (c2_base + j < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i + block_size_0/2];
    }
  }
  if (j + local_c2 < block_size_1/2)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left2 + (local_c2 + j + shift) * LDA + local_c1 + i + *line_length/2;
//      index = index + (c1_base + i -shift < 0 ? *line_length/2 : 0)
//                    + (c2_base + j < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i + block_size_0/2];
    }
    if (i + local_c1 < block_size_0/2)
    {
      int index = upper_left2 + (local_c2 + j + shift) * LDA + local_c1 + i + *line_length/2;
//      index = index + (c1_base + i -shift < 0 ? *line_length/2 : 0)
//                    + (c2_base + j < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i + block_size_0/2];
    }
  }

  ///////////
  // part: HH
  ///////////
  for (j = block_size_1/2; j < block_size_1-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left2 + (*line_length/2 + local_c2 + j - block_size_1/2) * LDA + local_c1 + i + *line_length/2;
//      index = index + (c1_base + i -shift < 0 ? *line_length/2 : 0)
//                    + (c2_base + j -shift - block_size_1/2 < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i + block_size_0/2];
    }
    if (i + local_c1 < block_size_0/2)
    {
      int index = upper_left2 + (*line_length/2 + local_c2 + j - block_size_1/2) * LDA + local_c1 + i + *line_length/2;
//      index = index + (c1_base + i -shift < 0 ? *line_length/2 : 0)
//                    + (c2_base + j -shift - block_size_1/2 < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i + block_size_0/2];
    }
  }
  if (j + local_c2 < block_size_1)
  {
    int i;
    for (i = 0; i < block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left2 + (*line_length/2 + local_c2 + j - block_size_1/2) * LDA + local_c1 + i + *line_length/2;
//      index = index + (c1_base + i -shift < 0 ? *line_length/2 : 0)
//                    + (c2_base + j -shift - block_size_1/2 < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i + block_size_0/2];
    }
    if (i + local_c1 < block_size_0/2)
    {
      int index = upper_left2 + (*line_length/2 + local_c2 + j - block_size_1/2) * LDA + local_c1 + i + *line_length/2;
//      index = index + (c1_base -shift + i < 0 ? *line_length/2 : 0)
//                    + (c2_base -shift + j - block_size_1/2 < 0 ? *line_length/2 * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i + block_size_0/2];
    }
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
      const int part_index = local_c2 * border_block_size_0 + 2 * local_c1;
      const int part_index2 = local_c2 * block_size_0 + local_c1;

      // j: loop over columns per thread -> c2
      // i: loop over rows per thread -> c1
      int j;
      for (j = 0; j < border_block_size_1-GROUP_SIZE_1; j += GROUP_SIZE_1)
      {
        int i;
        for (i = 0; i < block_size_0 - GROUP_SIZE_0; i += GROUP_SIZE_0)
        {
          const int index = part_index + j * border_block_size_0 + i;
          const int index2 = part_index2 + j * block_size_0 + i/2;
          tmp2 [index2] = conv_step_lo (index, _lpf, tmp, 1);
        }
        if (i + local_c1 < block_size_0)
        {
          const int index = part_index + j * border_block_size_0 + i;
          const int index2 = part_index2 + j * block_size_0 + i/2;
          tmp2 [index2] = conv_step_lo (index, _lpf, tmp, 1);
        }
      }
    
      // for "odd" block sizes
      if (j + local_c2 < border_block_size_1)
      {
        int i;
        for (i = 0; i < block_size_0 - GROUP_SIZE_0; i += GROUP_SIZE_0)
        {
          const int index = part_index + j * border_block_size_0 + i;
          const int index2 = part_index2 + j * block_size_0 + i/2;
          tmp2 [index2] = conv_step_lo (index, _lpf, tmp, 1);
        }
        if (i + local_c1 < block_size_0)
        {
          const int index = part_index + j * border_block_size_0 + i;
          const int index2 = part_index2 + j * block_size_0 + i/2;
          tmp2 [index2] = conv_step_lo (index, _lpf, tmp, 1);
        }
      }
    
    } // end of lowpass filter
    
    else  // highpass filter
    {
     
      const int start_hi = FL;
    
      // reused index parts
      const int part_index = local_c2 * border_block_size_0 + 2 * (local_c1 - GROUP_SIZE_0/2) - 1 + start_hi;
      const int part_index2 = local_c2 * block_size_0 + local_c1 - GROUP_SIZE_0/2 + block_size_0/2;
        
      // j: loop over columns per thread
      // i: loop over elements per column per thread
      int j;
      for (j = 0; j < border_block_size_1-GROUP_SIZE_1; j += GROUP_SIZE_1)
      {
        int i;
        for (i = 0; i < block_size_0 - GROUP_SIZE_0; i += GROUP_SIZE_0)
        {
          const int index = part_index + j * border_block_size_0 + i;
          const int index2 = part_index2 + j * block_size_0 + i/2;
          tmp2 [index2] = conv_step_hi (index, _hpf, tmp, 1);
        }
        if (i + local_c1 - GROUP_SIZE_0/2 < block_size_0)
        {
          const int index = part_index + j * border_block_size_0 + i;
          const int index2 = part_index2 + j * block_size_0 + i/2;
          tmp2 [index2] = conv_step_hi (index, _hpf, tmp, 1);          
        }
      }
    
      // for "odd" block sizes
      if (j + local_c2 < border_block_size_1)
      {
        int i;
        for (i = 0; i < block_size_0 - GROUP_SIZE_0; i += GROUP_SIZE_0)
        {
          const int index = part_index + j * border_block_size_0 + i;
          const int index2 = part_index2 + j * block_size_0 + i/2;
          tmp2 [index2] = conv_step_hi (index, _hpf, tmp, 1);
        }
        if (i + local_c1 - GROUP_SIZE_0/2 < block_size_0)
        {
          const int index = part_index + j * border_block_size_0 + i;
          const int index2 = part_index2 + j * block_size_0 + i/2;
          tmp2 [index2] = conv_step_hi (index, _hpf, tmp, 1);
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

    const int half_bs_0 = block_size_0/2;

    // ROWS

    if (local_c2 < GROUP_SIZE_1/2)  // lowpass filter
    {

      // reused index parts
      const int part_index = local_c1 + (2 * local_c2) * block_size_0;
      const int part_index2 = local_c1 + local_c2 * block_size_0;

      // j: loop over rows per thread -> c1
      // i: loop over elements per row per thread -> c2
      int j;
      for (j = 0; j < block_size_0-GROUP_SIZE_0; j += GROUP_SIZE_0)
      {
        int i;
        for (i = 0; i < block_size_1 - GROUP_SIZE_1; i += GROUP_SIZE_1)
        {
          const int index = part_index + j + i * block_size_0;
          const int index2 = part_index2 + j + i * half_bs_0;
          tmp2 [index2] = conv_step_lo (index, _lpf, tmp, block_size_0);
        }
        if (i + local_c2 < block_size_1)
        {
          const int index = part_index + j + i * block_size_0;
          const int index2 = part_index2 + j + i * half_bs_0;
          tmp2 [index2] = conv_step_lo (index, _lpf, tmp, block_size_0);
        }
      }
    
      // for "odd" block sizes
      if (j + local_c1 < block_size_0)
      {
        int i;
        for (i = 0; i < block_size_1 - GROUP_SIZE_1; i += GROUP_SIZE_1)
        {
          const int index = part_index + j + i * block_size_0;
          const int index2 = part_index2 + j + i * half_bs_0;
          tmp2 [index2] = conv_step_lo (index, _lpf, tmp, block_size_0);
        }
        if (i + local_c2 < block_size_1)
        {
          const int index = part_index + j + i * block_size_0;
          const int index2 = part_index2 + j + i * half_bs_0;
          tmp2 [index2] = conv_step_lo (index, _lpf, tmp, block_size_0);
        }
      }
    
    } // end of lowpass filter
    
    else  // highpass filter
    {
        
      const int start_hi = FL;
        
      // reused index parts
      const int part_index = local_c1 + (start_hi - 1 + 2 * (local_c2 - GROUP_SIZE_1/2)) * block_size_0;
      const int part_index2 = local_c1 + (local_c2 - GROUP_SIZE_1/2 + block_size_1/2) * block_size_0;
        
      // j: loop over rows per thread -> c1
      // i: loop over elements per row per thread -> c2
      int j;
      for (j = 0; j < block_size_0-GROUP_SIZE_0; j += GROUP_SIZE_0)
      {
        int i;
        for (i = 0; i < block_size_1 - GROUP_SIZE_1; i += GROUP_SIZE_1)
        {
          const int index = part_index + j + i * block_size_0;
          const int index2 = part_index2 + j + i * half_bs_0;
          tmp2 [index2] = conv_step_hi (index, _hpf, tmp, block_size_0);
        }
        if (i + local_c2 - GROUP_SIZE_1/2 < block_size_1)
        {
          const int index = part_index + j + i * block_size_0;
          const int index2 = part_index2 + j + i * half_bs_0;
          tmp2 [index2] = conv_step_hi (index, _hpf, tmp, block_size_0);
        }
      }
    
      // for "odd" block sizes
      if (j + local_c1 < block_size_0)
      {
        int i;
        for (i = 0; i < block_size_1 - GROUP_SIZE_1; i += GROUP_SIZE_1)
        {
          const int index = part_index + j + i * block_size_0;
          const int index2 = part_index2 + j + i * half_bs_0;
          tmp2 [index2] = conv_step_hi (index, _hpf, tmp, block_size_0);
        }
        if (i + local_c2 - GROUP_SIZE_1/2 < block_size_1)
        {
          const int index = part_index + j + i * block_size_0;
          const int index2 = part_index2 + j + i * half_bs_0;
          tmp2 [index2] = conv_step_hi (index, _hpf, tmp, block_size_0);
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

    const int local_c1 = get_local_id (0);
    const int local_c2 = get_local_id (1);

    int l = 1;
    int current_line_length = *line_length;

    // first level (only for EVEN number of levels)
    if (((*num_levels) & 1) == 0)
    {

      const int block_size_0 = *line_length / (min (*line_length, (int) get_global_size (0))/GROUP_SIZE_0);
      const int block_size_0_alt = (2 * *line_length) / (min (*line_length, (int) get_global_size (0))/GROUP_SIZE_0);
      const int block_size_1 = *line_length / (min (*line_length, (int) get_global_size (1))/GROUP_SIZE_1);

      const int upper_left  = get_group_id (1) * block_size_1 * LDA
                            + get_group_id (0) * block_size_0;
      const int upper_left2 = get_group_id (1) * block_size_1 * LDA
                            + get_group_id (0) * block_size_0_alt;

      const int index_base = upper_left + local_c2 * LDA + local_c1;


      // choose active threads
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
            const int index = index_base + j * LDA + i;
            arg2 [index] = arg1 [index];
          }
          if (i + local_c1 < block_size_0)
          {
            const int index = index_base + j * LDA + i;
            arg2 [index] = arg1 [index];
          }
        }
        if (j + local_c2 < block_size_1)
        {
          int i = 0;
          for (; i < block_size_0 - GROUP_SIZE_0; i += GROUP_SIZE_0)
          {
            const int index = index_base + j * LDA + i;
            arg2 [index] = arg1 [index];
          }
          if (i + local_c1 < block_size_0)
          {
            const int index = index_base + j * LDA + i;
            arg2 [index] = arg1 [index];
          }
        }

      } // if: choose active threads

      l += 2;
      current_line_length *= 2;

    } // if: first level (case of EVEN number of levels)

    // loop over levels
    for (; l < *num_levels; l += 2)
    {

      if (get_global_id (1) < current_line_length
        && get_global_id (0) < current_line_length)
      {

        const int block_size_0 = current_line_length / (min (current_line_length, (int) get_global_size (0))/GROUP_SIZE_0);
        const int block_size_0_alt = (2 * current_line_length) / (min (current_line_length, (int) get_global_size (0))/GROUP_SIZE_0);
        const int block_size_1 = current_line_length / (min (current_line_length, (int) get_global_size (1))/GROUP_SIZE_1);

        const int upper_left  = get_group_id (1) * block_size_1 * LDA
                              + get_group_id (0) * block_size_0;
        const int upper_left2 = get_group_id (1) * block_size_1 * LDA
                              + get_group_id (0) * block_size_0_alt;

        const int index_base = upper_left + current_line_length + local_c2 * LDA + local_c1;
        const int index_base2 = upper_left2 + (current_line_length + local_c2) * LDA + local_c1;

        // copy bottom left corner
        int j = 0;
        for (; j < block_size_1 - GROUP_SIZE_1; j += GROUP_SIZE_1)
        {
          int i = 0;
          for (; i < block_size_0 - GROUP_SIZE_0; i += GROUP_SIZE_0)
          {
            const int index = index_base + j * LDA + i;
            arg2 [index] = arg1 [index];
          }
          if (i + local_c1 < block_size_0)
          {
            const int index = index_base + j * LDA + i;
            arg2 [index] = arg1 [index];
          }
        }
        if (j + local_c2 < block_size_1)
        {
          int i = 0;
          for (; i < block_size_0 - GROUP_SIZE_0; i += GROUP_SIZE_0)
          {
            const int index = index_base + j * LDA + i;
            arg2 [index] = arg1 [index];
          }
          if (i + local_c1 < block_size_0)
          {
            const int index = index_base + j * LDA + i;
            arg2 [index] = arg1 [index];
          }
        }

        // copy parts on the right
        for (j = 0; j < block_size_1 - GROUP_SIZE_1; j += GROUP_SIZE_1)
        {
          int i = 0;
          for (; i < block_size_0_alt - GROUP_SIZE_0; i += GROUP_SIZE_0)
          {
            const int index = index_base2 + j * LDA + i;
            arg2 [index] = arg1 [index];
          }
          if (i + local_c1 < block_size_0_alt)
          {
            const int index = index_base2 + j * LDA + i;
            arg2 [index] = arg1 [index];
          }
        }
        if (j + local_c2 < block_size_1)
        {
          int i = 0;
          for (; i < block_size_0_alt - GROUP_SIZE_0; i += GROUP_SIZE_0)
          {
            const int index = index_base2 + j * LDA + i;
            arg2 [index] = arg1 [index];
          }
          if (i + local_c1 < block_size_0_alt)
          {
            const int index = index_base2 + j * LDA + i;
            arg2 [index] = arg1 [index];
          }
        }

      } // if: choose active threads

      current_line_length *= 4;
    
    } // loop over levels

}



/**
 * @author djoergens
 */
kernel void dwt2 (__global A_type * input,
          __constant A_type * _lpf,
          __constant A_type * _hpf,
          __global A_type * output,
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

    const int block_size_0 = *line_length / (min (*line_length, (int) get_global_size (0))/GROUP_SIZE_0);
    const int block_size_1 = *line_length / (min (*line_length, (int) get_global_size (1))/GROUP_SIZE_1);
    const int border_block_size_0 = block_size_0 + offset;
    const int border_block_size_1 = block_size_1 + offset;

    const int local_c1 = get_local_id (0);
    const int local_c2 = get_local_id (1);

    __local A_type * tmp  = & loc_mem [get_local_id (2) * (border_block_size_0 * border_block_size_1 + block_size_0 * border_block_size_1)];
    __local A_type * tmp2 = & loc_mem [get_local_id (2) * (border_block_size_0 * border_block_size_1 + block_size_0 * border_block_size_1)
                                       + border_block_size_0 * border_block_size_1];

    const int upper_left = get_group_id (1) * block_size_1 * LDA
                         + get_group_id (0) * block_size_0
                         - offset * LDA
                         - offset;
    const int upper_left2 = get_group_id (1) * block_size_1 / 2 * LDA
                          + get_group_id (0) * block_size_0 / 2;

    for (int slice = get_group_id (2) * get_local_size (2); slice < *num_slices; slice += get_global_size (2))
    {

      // update start address of current slice
      __global A_type * arg1 = input + slice * LDA * LDB;
      __global A_type * arg2 = output + slice * LDA * LDB;

      barrier (CLK_LOCAL_MEM_FENCE);

      /////////////////////////////
      // load block to local memory
      /////////////////////////////
      if (get_group_id (0) * get_group_id (1) == 0)
        global2local (arg1, tmp, upper_left, local_c1, local_c2, border_block_size_0, border_block_size_1, line_length);
      else
        global2local__ (arg1, tmp, upper_left, local_c1, local_c2, border_block_size_0, border_block_size_1, line_length);

      barrier (CLK_LOCAL_MEM_FENCE); // local mem fence since work is performed on local memory !!!

      // filter operations
      filter_columns (local_c1, local_c2, tmp, tmp2, _lpf, _hpf, block_size_0, block_size_1, border_block_size_0, border_block_size_1);

      barrier (CLK_LOCAL_MEM_FENCE);

      filter_rows (local_c1, local_c2, tmp2, tmp, _lpf, _hpf, block_size_0, block_size_1);

      barrier (CLK_LOCAL_MEM_FENCE);

      //////////////////////////////
      // write back to global memory
      //////////////////////////////
      if (get_group_id (0) * get_group_id (1) == 0)
        local2global (tmp, arg2, upper_left2, local_c1, local_c2, block_size_0, block_size_1, line_length);
      else
        local2global__ (tmp, arg2, upper_left2, local_c1, local_c2, block_size_0, block_size_1, line_length);

    } // loop over slices

  } // if: choose active threads

}








