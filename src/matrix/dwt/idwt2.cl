// created on Oct 2, 2013


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


void
iglobal2local       (__global A_type * arg1, __local A_type * tmp,
                     const int local_c1, const int local_c2,
                     const int block_size_0, const int block_size_1,
                     const int border_block_size_0,
                     const int border_block_size_1, __constant int * line_length)
{
  
  const int upper_left = get_group_id (1) * block_size_1 / 2 * LDA
                       + get_group_id (0) * block_size_0 / 2;
                    
  const c1_base = block_size_0/2 * get_group_id (0) + offset + local_c1;
  const c2_base = block_size_1/2 * get_group_id (1) + offset + local_c2;

  ///////////
  // part: LL
  ///////////
  int j;
  for (j = 0; j < border_block_size_1/2-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < border_block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
        = arg1 [upper_left + (local_c2 + j) * LDA + local_c1 + i];
    if (i + local_c1 < border_block_size_0/2)
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
        = arg1 [upper_left + (local_c2 + j) * LDA + local_c1 + i];
  }
  if (j + local_c2 < border_block_size_1/2)
  {
    int i;
    for (i = 0; i < border_block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
        = arg1 [upper_left + (local_c2 + j) * LDA + local_c1 + i];
    if (i + local_c1 < border_block_size_0/2)
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
        = arg1 [upper_left + (local_c2 + j) * LDA + local_c1 + i];
  }

  ///////////
  // part: LH
  ///////////
  for (j = border_block_size_1/2; j < border_block_size_1-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < border_block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left + (*line_length/2 + local_c2 + j - border_block_size_1/2) * LDA + local_c1 + i;
      index = index + (c1_base + i > *line_length/2 ? -*line_length/2 : 0)
                    + (c2_base + j - border_block_size_1/2 > *line_length/2 ? -*line_length/2*LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
        = arg1 [index];
    }
    if (i + local_c1 < border_block_size_0/2)
    {
      int index = upper_left + (*line_length/2 + local_c2 + j - border_block_size_1/2) * LDA + local_c1 + i;
      index = index + (c1_base + i > *line_length/2 ? -*line_length/2 : 0)
                    + (c2_base + j - border_block_size_1/2 > *line_length/2 ? -*line_length/2*LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
        = arg1 [index];
    }
  }
  if (j + local_c2 < border_block_size_1)
  {
    int i;
    for (i = 0; i < border_block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left + (*line_length/2 + local_c2 + j - border_block_size_1/2) * LDA + local_c1 + i;
      index = index + (c1_base + i > *line_length/2 ? -*line_length/2 : 0)
                    + (c2_base + j - border_block_size_1/2 > *line_length/2 ? -*line_length/2*LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
        = arg1 [index];
    }
    if (i + local_c1 < border_block_size_0/2)
    {
      int index = upper_left + (*line_length/2 + local_c2 + j - border_block_size_1/2) * LDA + local_c1 + i;
      index = index + (c1_base + i > *line_length/2 ? -*line_length/2 : 0)
                    + (c2_base + j - border_block_size_1/2 > *line_length/2 ? -*line_length/2*LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
        = arg1 [index];
    }
  }

  ///////////
  // part: HL
  ///////////
  for (j = 0; j < border_block_size_1/2-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < border_block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i + border_block_size_0/2]
        = arg1 [upper_left + (local_c2 + j) * LDA + local_c1 + i + *line_length/2];
    if (i + local_c1 < border_block_size_0/2)
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i + border_block_size_0/2]
        = arg1 [upper_left + (local_c2 + j) * LDA + local_c1 + i + *line_length/2];
  }
  if (j + local_c2 < border_block_size_1/2)
  {
    int i;
    for (i = 0; i < border_block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i + border_block_size_0/2]
        = arg1 [upper_left + (local_c2 + j) * LDA + local_c1 + i + *line_length/2];
    if (i + local_c1 < border_block_size_0/2)
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i + border_block_size_0/2]
        = arg1 [upper_left + (local_c2 + j) * LDA + local_c1 + i + *line_length/2];
  }

  ///////////
  // part: HH
  ///////////
  for (j = border_block_size_1/2; j < border_block_size_1-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < border_block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left + (*line_length/2 + local_c2 + j - border_block_size_1/2) * LDA + local_c1 + i + *line_length/2;
      index = index + (c1_base + i > *line_length/2 ? -*line_length/2 : 0)
                    + (c2_base + j - border_block_size_1/2 > *line_length/2 ? -*line_length/2*LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i + border_block_size_0/2]
        = arg1 [index];
    }
    if (i + local_c1 < border_block_size_0/2)
    {
      int index = upper_left + (*line_length/2 + local_c2 + j - border_block_size_1/2) * LDA + local_c1 + i + *line_length/2;
      index = index + (c1_base + i > *line_length/2 ? -*line_length/2 : 0)
                    + (c2_base + j - border_block_size_1/2 > *line_length/2 ? -*line_length/2*LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i + border_block_size_0/2]
        = arg1 [index];
    }
  }
  if (j + local_c2 < border_block_size_1)
  {
    int i;
    for (i = 0; i < border_block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left + (*line_length/2 + local_c2 + j - border_block_size_1/2) * LDA + local_c1 + i + *line_length/2;
      index = index + (c1_base + i > *line_length/2 ? -*line_length/2 : 0)
                    + (c2_base + j - border_block_size_1/2 > *line_length/2 ? -*line_length/2*LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i + border_block_size_0/2]
        = arg1 [index];
    }
    if (i + local_c1 < border_block_size_0/2)
    {
      int index = upper_left + (*line_length/2 + local_c2 + j - border_block_size_1/2) * LDA + local_c1 + i + *line_length/2;
      index = index + (c1_base + i > *line_length/2 ? -*line_length/2 : 0)
                    + (c2_base + j - border_block_size_1/2 > *line_length/2 ? -*line_length/2*LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i + border_block_size_0/2]
        = arg1 [index];
    }
  }

}


void
ilocal2global       (__global A_type * arg2, __local A_type * tmp2,
                     const int upper_left2, const int local_c1, const int local_c2,
                     const int block_size_0,
                     const int block_size_1, __constant int * line_length)
{
  
  int j;
  for (j = 0; j < block_size_1-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < block_size_0-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left2 + (local_c2 + j) * LDA + local_c1 + i;
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
    }
    if (i + local_c1 < block_size_0)
    {
      int index = upper_left2 + (local_c2 + j) * LDA + local_c1 + i;
      arg2 [index] = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
    }
  }
  if (j + local_c2 < block_size_1)
  {
    int i;
    for (i = 0; i < block_size_0-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      arg2 [upper_left2 + (local_c2 + j) * LDA + local_c1 + i]
        = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
    }
    if (i + local_c1 < block_size_0)
      arg2 [upper_left2 + (local_c2 + j) * LDA + local_c1 + i]
        = tmp2 [(local_c2 + j) * block_size_0 + local_c1 + i];
  }
}


void
ifiltering1              (const int local_c1, const int local_c2,
                          __local A_type * tmp, __local A_type * tmp2,
                          __constant A_type * _lpf, __constant A_type * _hpf,
                          const int block_size_0, const int block_size_1,
                          const int border_block_size_0, const int border_block_size_1)
{

  int j;
  for (j = 0; j < block_size_1 - GROUP_SIZE_1*2; j += GROUP_SIZE_1*2)
  {
    int i;
    for (i = 0; i < block_size_0 - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      tmp2 [(j + 2*local_c2) * block_size_0 + i + local_c1]
        = tmp [(j/2 + local_c2) * border_block_size_0 + i/2 + (local_c1>>1)];
      tmp2 [(j + 2*local_c2 + 1) * block_size_0 + i + local_c1]
        = tmp [(j/2 + local_c2) * border_block_size_0 + i/2 + (local_c1>>1)];
    }
    if (i + local_c1 < block_size_0)
    {
      tmp2 [(j + 2*local_c2) * block_size_0 + i + local_c1]
        = tmp [(j/2 + local_c2) * border_block_size_0 + i/2 + (local_c1>>1)];
      tmp2 [(j + 2*local_c2 + 1) * block_size_0 + i + local_c1]
        = tmp [(j/2 + local_c2) * border_block_size_0 + i/2 + (local_c1>>1)];
    }
  }
  if (j + 2*local_c2 < block_size_1)
  {
    int i;
    for (i = 0; i < block_size_0 - GROUP_SIZE_0; i += GROUP_SIZE_0)
      tmp2 [(j + 2*local_c2) * block_size_0 + i + local_c1]
        = tmp [(j/2 + local_c2) * border_block_size_0 + i/2 + (local_c1>>1)];
    if (i + local_c1 < block_size_0)
      tmp2 [(j + 2*local_c2) * block_size_0 + i + local_c1]
        = tmp [(j/2 + local_c2) * border_block_size_0 + i/2 + (local_c1>>1)];
  }
  if (j + 2*local_c2 + 1 < block_size_1)
  {
    int i;
    for (i = 0; i < block_size_0 - GROUP_SIZE_0; i += GROUP_SIZE_0)
      tmp2 [(j + 2*local_c2 + 1) * block_size_0 + i + local_c1]
        = tmp [(j/2 + local_c2) * border_block_size_0 + i/2 + (local_c1>>1)];
    if (i + local_c1 < block_size_0)
      tmp2 [(j + 2*local_c2 + 1) * block_size_0 + i + local_c1]
        = tmp [(j/2 + local_c2) * border_block_size_0 + i/2 + (local_c1>>1)];
  }

}


void
ifiltering               (const int local_c1, const int local_c2,
                          __local A_type * tmp, __local A_type * tmp2,
                          __constant A_type * _lpf, __constant A_type * _hpf,
                          const int block_size_0, const int block_size_1,
                          const int border_block_size_0, const int border_block_size_1,
                          const int base_index)
{

  int j;
  for (j = 0; j < block_size_1 - GROUP_SIZE_1*2; j += GROUP_SIZE_1*2)
  {
    int i;
    for (i = 0; i < block_size_0 - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      tmp2 [(j + 2*local_c2) * block_size_0 + i + local_c1]
        += tmp [(j/2 + local_c2) * border_block_size_0 + i/2 + (local_c1>>1) + base_index];
      tmp2 [(j + 2*local_c2 + 1) * block_size_0 + i + local_c1]
        += tmp [(j/2 + local_c2) * border_block_size_0 + i/2 + (local_c1>>1) + base_index];
    }
    if (i + local_c1 < block_size_0)
    {
      tmp2 [(j + 2*local_c2) * block_size_0 + i + local_c1]
        += tmp [(j/2 + local_c2) * border_block_size_0 + i/2 + (local_c1>>1) + base_index];
      tmp2 [(j + 2*local_c2 + 1) * block_size_0 + i + local_c1]
        += tmp [(j/2 + local_c2) * border_block_size_0 + i/2 + (local_c1>>1) + base_index];
    }
  }
  if (j + 2*local_c2 < block_size_1)
  {
    int i;
    for (i = 0; i < block_size_0 - GROUP_SIZE_0; i += GROUP_SIZE_0)
      tmp2 [(j + 2*local_c2) * block_size_0 + i + local_c1]
        += tmp [(j/2 + local_c2) * border_block_size_0 + i/2 + (local_c1>>1) + base_index];
    if (i + local_c1 < block_size_0)
      tmp2 [(j + 2*local_c2) * block_size_0 + i + local_c1]
        += tmp [(j/2 + local_c2) * border_block_size_0 + i/2 + (local_c1>>1) + base_index];
  }
  if (j + 2*local_c2 + 1 < block_size_1)
  {
    int i;
    for (i = 0; i < block_size_0 - GROUP_SIZE_0; i += GROUP_SIZE_0)
      tmp2 [(j + 2*local_c2 + 1) * block_size_0 + i + local_c1]
        += tmp [(j/2 + local_c2) * border_block_size_0 + i/2 + (local_c1>>1) + base_index];
    if (i + local_c1 < block_size_0)
      tmp2 [(j + 2*local_c2 + 1) * block_size_0 + i + local_c1]
        += tmp [(j/2 + local_c2) * border_block_size_0 + i/2 + (local_c1>>1) + base_index];
  }

}



/**
 * @author djoergens
 */
kernel void idwt2 (__global A_type * arg1,
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


  const int block_size_0 = *line_length / (min (*line_length, (int) get_global_size (0))/GROUP_SIZE_0);
  const int block_size_1 = *line_length / (min (*line_length, (int) get_global_size (1))/GROUP_SIZE_1);
  const int border_block_size_0 = block_size_0 + offset;
  const int border_block_size_1 = block_size_1 + offset;

  const int local_c1 = get_local_id (0);
  const int local_c2 = get_local_id (1);

  __local A_type * tmp  = & loc_mem [0];
  __local A_type * tmp2 = & loc_mem [border_block_size_0 * border_block_size_1];

  const int upper_left = get_group_id (1) * block_size_1 / 2 * LDA
                       + get_group_id (0) * block_size_0 / 2;
  const int upper_left2 = get_group_id (1) * block_size_1 * LDA
                        + get_group_id (0) * block_size_0;

  iglobal2local (arg1, tmp, local_c1, local_c2, block_size_0, block_size_1, border_block_size_0, border_block_size_1, line_length);

  barrier (CLK_LOCAL_MEM_FENCE);
 
  ifiltering1    (local_c1, local_c2, tmp, tmp2, _lpf, _hpf, block_size_0, block_size_1,
                  border_block_size_0, border_block_size_1);
  int base_index = border_block_size_0/2;
  ifiltering     (local_c1, local_c2, tmp, tmp2, _lpf, _hpf, block_size_0, block_size_1,
                  border_block_size_0, border_block_size_1, base_index);
  base_index = border_block_size_1/2 * border_block_size_0;
  ifiltering     (local_c1, local_c2, tmp, tmp2, _lpf, _hpf, block_size_0, block_size_1,
                  border_block_size_0, border_block_size_1, base_index);
  base_index = border_block_size_1/2 * border_block_size_0 + border_block_size_0/2;
  ifiltering     (local_c1, local_c2, tmp, tmp2, _lpf, _hpf, block_size_0, block_size_1,
                  border_block_size_0, border_block_size_1, base_index);

  barrier (CLK_LOCAL_MEM_FENCE);

  ilocal2global (arg2, tmp2, upper_left2, local_c1, local_c2, block_size_0, block_size_1, line_length);

}