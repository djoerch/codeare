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

# ifndef LDB
  # define LDB 512
# endif

# ifndef FL
  # define FL 4
# endif

# ifndef ODD_FILTER
  # define ODD_FILTER 1
# endif

# ifndef I_OFFSET
  __constant const int i_offset = FL-1;
  # define I_OFFSET
# endif



void
iglobal2local (__global A_type * arg1, __local A_type * tmp,
                      const int local_c1, const int local_c2,
                      const int block_size_0, const int block_size_1,
                      const int border_block_size_0,
                      __constant int * line_length)
{
  
  const int j_max = block_size_1 / 2 + i_offset;
  const int i_max = block_size_0 / 2 + i_offset;

  const int shift_lo = - i_offset;

  const int upper_left = get_group_id (1) * block_size_1 / 2 * LDA
                       + get_group_id (0) * block_size_0 / 2;
                    
  const int c1_base_lo = block_size_0/2 * get_group_id (0) + local_c1 + shift_lo;
  const int c2_base_lo = block_size_1/2 * get_group_id (1) + local_c2 + shift_lo;

  const int c1_base_hi = block_size_0/2 * get_group_id (0) + local_c1 + 1;
  const int c2_base_hi = block_size_1/2 * get_group_id (1) + local_c2 + 1;

  const int half_line = *line_length/2;

  const int index_base = upper_left + local_c2 * LDA + local_c1;
  const int local_base_1 = local_c2 * border_block_size_0 + local_c1;
  const int local_base_2 = local_base_1 + i_max;

  ///////////
  // part: LL
  ///////////
  const int index_base_ll = index_base + shift_lo * LDA + shift_lo;
  int j;
  for (j = 0; j < j_max - GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      const int index = index_base_ll + j * LDA + i
                      + ((c1_base_lo + i < 0)
                      +  (c2_base_lo + j < 0) * LDA) * half_line;
      tmp [local_base_1 + j * border_block_size_0 + i] = arg1 [index];
    }
    if (i + local_c1 < i_max)
    {
      const int index = index_base_ll + j * LDA + i
                      + ((c1_base_lo + i < 0)
                      +  (c2_base_lo + j < 0) * LDA) * half_line;
      tmp [local_base_1 + j * border_block_size_0 + i] = arg1 [index];
    }
  }
  if (j + local_c2 < j_max)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      const int index = index_base_ll + j * LDA + i
                      + ((c1_base_lo + i < 0)
                      +  (c2_base_lo + j < 0) * LDA) * half_line;
      tmp [local_base_1 + j * border_block_size_0 + i] = arg1 [index];
    }
    if (i + local_c1 < i_max)
    {
      const int index = index_base_ll + j * LDA + i
                      + ((c1_base_lo + i < 0)
                      +  (c2_base_lo + j < 0) * LDA) * half_line;
      tmp [local_base_1 + j * border_block_size_0 + i] = arg1 [index];
    }
  }

  ///////////
  // part: LH
  ///////////
  const int index_base_lh = index_base + *line_length/2 * LDA + shift_lo;
  for (j = 0; j < j_max - GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      const int index = index_base_lh + j * LDA + i
                      + ((c1_base_lo + i < 0)
                      -  (c2_base_hi + j > half_line) * LDA) * half_line;
      tmp [local_base_1 + (j+j_max) * border_block_size_0 + i] = arg1 [index];
    }
    if (i + local_c1 < i_max)
    {
      const int index = index_base_lh + j * LDA + i
                      + ((c1_base_lo + i < 0)
                      -  (c2_base_hi + j > half_line) * LDA) * half_line;
      tmp [local_base_1 + (j+j_max) * border_block_size_0 + i] = arg1 [index];
    }
  }
  if (j + local_c2 < j_max)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      const int index = index_base_lh + j * LDA + i
                      + ((c1_base_lo + i < 0)
                      -  (c2_base_hi + j > half_line) * LDA) * half_line;
      tmp [local_base_1 + (j+j_max) * border_block_size_0 + i] = arg1 [index];
    }
    if (i + local_c1 < i_max)
    {
      const int index = index_base_lh + j * LDA + i
                      + ((c1_base_lo + i < 0)
                      -  (c2_base_hi + j > half_line) * LDA) * half_line;
      tmp [local_base_1 + (j+j_max) * border_block_size_0 + i] = arg1 [index];
    }
  }

  ///////////
  // part: HL
  ///////////
  const int index_base_hl = index_base + shift_lo * LDA + half_line;
  for (j = 0; j < j_max - GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      const int index = index_base_hl + j * LDA + i
                      + ((c2_base_lo + j < 0) * LDA
                      -  (c1_base_hi + i > half_line)) * half_line;
      tmp [local_base_2 + j * border_block_size_0 + i] = arg1 [index];
    }
    if (i + local_c1 < i_max)
    {
      const int index = index_base_hl + j * LDA + i
                      + ((c2_base_lo + j < 0) * LDA
                      -  (c1_base_hi + i > half_line)) * half_line;
      tmp [local_base_2 + j * border_block_size_0 + i] = arg1 [index];
    }
  }
  if (j + local_c2 < j_max)
  {
    int i;
    for (i = 0; i < i_max-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      const int index = index_base_hl + j * LDA + i
                      + ((c2_base_lo + j < 0) * LDA
                      -  (c1_base_hi + i > half_line)) * half_line;
      tmp [local_base_2 + j * border_block_size_0 + i] = arg1 [index];
    }
    if (i + local_c1 < i_max)
    {
      const int index = index_base_hl + j * LDA + i
                      + ((c2_base_lo + j < 0) * LDA
                      -  (c1_base_hi + i > half_line)) * half_line;
      tmp [local_base_2 + j * border_block_size_0 + i] = arg1 [index];
    }
  }

  ///////////
  // part: HH
  ///////////
  const int index_base_hh = index_base + half_line * LDA + half_line;
  for (j = 0; j < j_max - GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      const int index = index_base_hh + j * LDA + i
                      - ((c1_base_hi + i > half_line)
                      +  (c2_base_hi + j > half_line) * LDA) * half_line;
      tmp [local_base_2 + (j+j_max) * border_block_size_0 + i] = arg1 [index];
    }
    if (i + local_c1 < i_max)
    {
      const int index = index_base_hh + j * LDA + i
                      - ((c1_base_hi + i > half_line)
                      +  (c2_base_hi + j > half_line) * LDA) * half_line;
      tmp [local_base_2 + (j+j_max) * border_block_size_0 + i] = arg1 [index];
    }
  }
  if (j + local_c2 < j_max)
  {
    int i;
    for (i = 0; i < border_block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      const int index = index_base_hh + j * LDA + i
                      - ((c1_base_hi + i > half_line)
                      +  (c2_base_hi + j > half_line) * LDA) * half_line;
      tmp [local_base_2 + (j+j_max) * border_block_size_0 + i] = arg1 [index];
    }
    if (i + local_c1 < i_max)
    {
      const int index = index_base_hh + j * LDA + i
                      - ((c1_base_hi + i > half_line)
                      +  (c2_base_hi + j > half_line) * LDA) * half_line;
      tmp [local_base_2 + (j+j_max) * border_block_size_0 + i] = arg1 [index];
    }
  }

}




void
ilocal2global       (__global A_type * arg2, __local A_type * tmp2,
                     const int upper_left2, const int local_c1, const int local_c2,
                     const int block_size_0,
                     const int block_size_1, __constant int * line_length)
{
  
  const int c1_base = get_group_id (0) * block_size_0 + local_c1;
  const int c2_base = get_group_id (1) * block_size_1 + local_c2;

  const int index_base = upper_left2 + local_c2 * LDA + local_c1;
  const int local_base = local_c2 * block_size_0 + local_c1;

  int j;
  for (j = 0; j < block_size_1-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < block_size_0-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      const int index = index_base + j * LDA + i;
//                      + ((c1_base + i < 0)
//                      +  (c2_base + j < 0) * LDA) * *line_length;
      arg2 [index] = tmp2 [local_base + j * block_size_0 + i];
    }
    if (i + local_c1 < block_size_0)
    {
      const int index = index_base + j * LDA + i;
//                      + ((c1_base + i < 0)
//                      +  (c2_base + j < 0) * LDA) * *line_length;
      arg2 [index] = tmp2 [local_base + j * block_size_0 + i];
    }
  }
  if (j + local_c2 < block_size_1)
  {
    int i;
    for (i = 0; i < block_size_0-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      const int index = index_base + j * LDA + i;
//                      + ((c1_base + i < 0)
//                      +  (c2_base + j < 0) * LDA) * *line_length;
      arg2 [index] = tmp2 [local_base + j * block_size_0 + i];
    }
    if (i + local_c1 < block_size_0)
    {
      const int index = index_base + j * LDA + i;
//                      + ((c1_base + i < 0)
//                      +  (c2_base + j < 0) * LDA) * *line_length;
      arg2 [index] = tmp2 [local_base + j * block_size_0 + i];
    }
  }

}


void
iconv_step1_lo              (const int index1, const int index2,
                             __constant B_type * filter,
                             __local A_type * tmp, __local A_type * tmp2,
                             const int increment)
{
  
  A_type sum1 = 0, sum2 = 0;
  # pragma unroll
  for (int k = 0; k < FL/2; k ++)
  {
    const A_type val = tmp [index1 - k * increment];
    sum1 += val * filter [2 * k];
    sum2 += val * filter [(2 * k + 1)];
  }

  if (ODD_FILTER == 1)
  {
    sum1 += tmp [index1 + (FL/2) * increment] * filter [0];
  }

  tmp2 [index2] = sum1;
  tmp2 [index2 + increment] = sum2;

}



void
iconv_step2_lo              (const int index1, const int index2,
                             __constant B_type * filter,
                             __local A_type * tmp, __local A_type * tmp2,
                             const int increment)
{

  A_type sum1 = 0, sum2 = 0;
  # pragma unroll
  for (int k = 0; k < FL/2; k ++)
  {
    const A_type val = tmp [index1 - k * increment];
    sum1 += val * filter [2 * k];
    sum2 += val * filter [(2 * k + 1)];
  }

  if (ODD_FILTER == 1)
  {
    sum1 += tmp [index1 + (FL/2) * increment] * filter [0];
  }

  tmp2 [index2] += sum1;
  tmp2 [index2 + increment] += sum2;

}


void
iconv_step2_hi              (const int index1, const int index2,
                             __constant B_type * filter,
                             __local A_type * tmp, __local A_type * tmp2,
                             const int increment)
{

  A_type sum1 = 0, sum2 = 0;
  # pragma unroll
  for (int k = 0; k < FL/2; k ++)
  {
    const A_type val = tmp [index1 + k * increment];
    sum1 += val * filter [2 * k];
    sum2 += val * filter [2 * k + 1];
  }

  if (ODD_FILTER == 1)
  {
    sum1 += tmp [index1 + (FL/2) * increment] * filter [FL-1];
  }

  tmp2 [index2] += sum2;
  tmp2 [index2 + increment] += sum1;

}



void
ifiltering1_rows         (const int local_c1, const int local_c2,
                          __local A_type * tmp, __local A_type * tmp2,
                          __constant B_type * _lpf, __constant B_type * _hpf,
                          const int block_size_0, const int block_size_1,
                          const int border_block_size_0, const int border_block_size_1)
{

  const int j_max = block_size_1 / 2;
  const int i_max = block_size_0 / 2 + i_offset;

  const int double_bbs_0 = 2 * border_block_size_0;

  // ROWS //

  const int start_lo = FL-1;

  const int local_base_1 = (local_c2 + start_lo) * border_block_size_0 + local_c1;
  const int local_base_2 = local_c2 * double_bbs_0 + local_c1;

  // LL //
  int j;
  for (j = 0; j < j_max - GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
        const int index1 = local_base_1 + j * border_block_size_0 + i;
        const int index2 = local_base_2 + j * double_bbs_0 + i;
        iconv_step1_lo (index1, index2, _lpf, tmp, tmp2, border_block_size_0);
    }
    if (i + local_c1 < i_max)
    {
        const int index1 = local_base_1 + j * border_block_size_0 + i;
        const int index2 = local_base_2 + j * double_bbs_0 + i;
        iconv_step1_lo (index1, index2, _lpf, tmp, tmp2, border_block_size_0);
    }
  }
  if (j + local_c2 < j_max)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
        const int index1 = local_base_1 + j * border_block_size_0 + i;
        const int index2 = local_base_2 + j * double_bbs_0 + i;
        iconv_step1_lo (index1, index2, _lpf, tmp, tmp2, border_block_size_0);
    }
    if (i + local_c1 < i_max)
    {
        const int index1 = local_base_1 + j * border_block_size_0 + i;
        const int index2 = local_base_2 + j * double_bbs_0 + i;
        iconv_step1_lo (index1, index2, _lpf, tmp, tmp2, border_block_size_0);
    }
  }

  // HL //

  const int i_start = i_max;
  const int i_end = 2 * i_max;

  for (j = 0; j < j_max - GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = i_start; i < i_end - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
        const int index1 = local_base_1 + j * border_block_size_0 + i;
        const int index2 = local_base_2 + j * double_bbs_0 + i;
        iconv_step1_lo (index1, index2, _lpf, tmp, tmp2, border_block_size_0);
    }
    if (i + local_c1 < i_end)
    {
        const int index1 = local_base_1 + j * border_block_size_0 + i;
        const int index2 = local_base_2 + j * double_bbs_0 + i;
        iconv_step1_lo (index1, index2, _lpf, tmp, tmp2, border_block_size_0);
    }
  }
  if (j + local_c2 < j_max)
  {
    int i;
    for (i = i_start; i < i_end - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
        const int index1 = local_base_1 + j * border_block_size_0 + i;
        const int index2 = local_base_2 + j * double_bbs_0 + i;
        iconv_step1_lo (index1, index2, _lpf, tmp, tmp2, border_block_size_0);
    }
    if (i + local_c1 < i_end)
    {
        const int index1 = local_base_1 + j * border_block_size_0 + i;
        const int index2 = local_base_2 + j * double_bbs_0 + i;
        iconv_step1_lo (index1, index2, _lpf, tmp, tmp2, border_block_size_0);
    }
  }

}





void
ifiltering_rows          (const int local_c1, const int local_c2,
                          __local A_type * tmp, __local A_type * tmp2,
                          __constant B_type * _lpf, __constant B_type * _hpf,
                          const int block_size_0, const int block_size_1,
                          const int border_block_size_0, const int border_block_size_1)
{

  const int j_max = block_size_1 / 2;
  const int i_max = block_size_0 / 2 + i_offset;

  const int double_bbs_0 = 2 * border_block_size_0;

  // ROWS //

  const int local_base_1 = (local_c2 + j_max + i_offset) * border_block_size_0 + local_c1;
  const int local_base_2 = local_c2 * double_bbs_0 + local_c1;

  // LH //
  int j;
  for (j = 0; j < j_max - GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
        const int index1 = local_base_1 + j * border_block_size_0 + i;
        const int index2 = local_base_2 + j * double_bbs_0 + i;
        iconv_step2_hi (index1, index2, _hpf, tmp, tmp2, border_block_size_0);
    }
    if (i + local_c1 < i_max)
    {
        const int index1 = local_base_1 + j * border_block_size_0 + i;
        const int index2 = local_base_2 + j * double_bbs_0 + i;
        iconv_step2_hi (index1, index2, _hpf, tmp, tmp2, border_block_size_0);
    }
  }
  if (j + local_c2 < j_max)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
        const int index1 = local_base_1 + j * border_block_size_0 + i;
        const int index2 = local_base_2 + j * double_bbs_0 + i;
        iconv_step2_hi (index1, index2, _hpf, tmp, tmp2, border_block_size_0);
    }
    if (i + local_c1 < i_max)
    {
        const int index1 = local_base_1 + j * border_block_size_0 + i;
        const int index2 = local_base_2 + j * double_bbs_0 + i;
        iconv_step2_hi (index1, index2, _hpf, tmp, tmp2, border_block_size_0);
    }
  }

  // HH //

  const int i_start = i_max;
  const int i_end = 2 * i_max;

  for (j = 0; j < j_max - GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = i_start; i < i_end - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
        const int index1 = local_base_1 + j * border_block_size_0 + i;
        const int index2 = local_base_2 + j * double_bbs_0 + i;
        iconv_step2_hi (index1, index2, _hpf, tmp, tmp2, border_block_size_0);
    }
    if (i + local_c1 < i_end)
    {
        const int index1 = local_base_1 + j * border_block_size_0 + i;
        const int index2 = local_base_2 + j * double_bbs_0 + i;
        iconv_step2_hi (index1, index2, _hpf, tmp, tmp2, border_block_size_0);
    }
  }
  if (j + local_c2 < j_max)
  {
    int i;
    for (i = i_start; i < i_end - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
        const int index1 = local_base_1 + j * border_block_size_0 + i;
        const int index2 = local_base_2 + j * double_bbs_0 + i;
        iconv_step2_hi (index1, index2, _hpf, tmp, tmp2, border_block_size_0);
    }
    if (i + local_c1 < i_end)
    {
        const int index1 = local_base_1 + j * border_block_size_0 + i;
        const int index2 = local_base_2 + j * double_bbs_0 + i;
        iconv_step2_hi (index1, index2, _hpf, tmp, tmp2, border_block_size_0);
    }
  }

}



void
ifiltering1_cols         (const int local_c1, const int local_c2,
                          __local A_type * tmp, __local A_type * tmp2,
                          __constant B_type * _lpf, __constant B_type * _hpf,
                          const int block_size_0, const int block_size_1,
                          const int border_block_size_0, const int border_block_size_1)
{

  const int j_max = block_size_1;
  const int i_max = block_size_0 / 2;

  // COLS //

  const int start_lo = FL-1;

  const int local_base_1 = local_c2 * border_block_size_0 + local_c1 + start_lo;
  const int local_base_2 = local_c2 * block_size_0 + 2 * local_c1;

  // L //
  int j;
  for (j = 0; j < j_max - GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
        const int index1 = local_base_1 + j * border_block_size_0 + i;
        const int index2 = local_base_2 + j * block_size_0 + 2 * i;
        iconv_step1_lo (index1, index2, _lpf, tmp2, tmp, 1);
    }
    if (i + local_c1 < i_max)
    {
        const int index1 = local_base_1 + j * border_block_size_0 + i;
        const int index2 = local_base_2 + j * block_size_0 + 2 * i;
        iconv_step1_lo (index1, index2, _lpf, tmp2, tmp, 1);
    }
  }
  if (j + local_c2 < j_max)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
        const int index1 = local_base_1 + j * border_block_size_0 + i;
        const int index2 = local_base_2 + j * block_size_0 + 2 * i;
        iconv_step1_lo (index1, index2, _lpf, tmp2, tmp, 1);
    }
    if (i + local_c1 < i_max)
    {
        const int index1 = local_base_1 + j * border_block_size_0 + i;
        const int index2 = local_base_2 + j * block_size_0 + 2 * i;
        iconv_step1_lo (index1, index2, _lpf, tmp2, tmp, 1);
    }
  }

}




void
ifiltering_cols          (const int local_c1, const int local_c2,
                          __local A_type * tmp, __local A_type * tmp2,
                          __constant B_type * _lpf, __constant B_type * _hpf,
                          const int block_size_0, const int block_size_1,
                          const int border_block_size_0, const int border_block_size_1)
{

  const int j_max = block_size_1;
  const int i_max = block_size_0 / 2;

  // COLS //

  const int local_base_1 = local_c2 * border_block_size_0 + local_c1 + i_max + i_offset;
  const int local_base_2 = local_c2 * block_size_0 + 2 * local_c1;

  // H //

  int j;
  for (j = 0; j < j_max - GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      const int index1 = local_base_1 + j * border_block_size_0 + i;
      const int index2 = local_base_2 + j * block_size_0 + 2 * i;
      iconv_step2_hi (index1, index2, _hpf, tmp2, tmp, 1);
    }
    if (i + local_c1 < i_max)
    {
      const int index1 = local_base_1 + j * border_block_size_0 + i;
      const int index2 = local_base_2 + j * block_size_0 + 2 * i;
      iconv_step2_hi (index1, index2, _hpf, tmp2, tmp, 1);
    }
  }
  if (j + local_c2 < j_max)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      const int index1 = local_base_1 + j * border_block_size_0 + i;
      const int index2 = local_base_2 + j * block_size_0 + 2 * i;
      iconv_step2_hi (index1, index2, _hpf, tmp2, tmp, 1);
    }
    if (i + local_c1 < i_max)
    {
      const int index1 = local_base_1 + j * border_block_size_0 + i;
      const int index2 = local_base_2 + j * block_size_0 + 2 * i;
      iconv_step2_hi (index1, index2, _hpf, tmp2, tmp, 1);
    }
  }

}




kernel void idwt2_prepare (__global A_type * input,
                           __global A_type * output,
                           __constant int * n,
                           __global int * m,
                           __global int * k,
                           __constant int * line_length,
                           __constant int * num_slices)
{

  // choose active threads
  if (get_global_id (0) < *line_length
    && get_global_id (1) < *line_length)
  {

    const int block_size_0 = *line_length / (min (*line_length, (int) get_global_size (0))/GROUP_SIZE_0);
    const int block_size_1 = *line_length / (min (*line_length, (int) get_global_size (1))/GROUP_SIZE_1);
    const int local_c1 = get_local_id (0);
    const int local_c2 = get_local_id (1);

    const int upper_left = get_group_id (1) * block_size_1 * LDA
                         + get_group_id (0) * block_size_0;

    const int index_base = upper_left + local_c2 * LDA + local_c1;

    // loop over slices
    for (int slice = get_group_id (2); slice < *num_slices; slice += get_global_size (2))
    {

      // start addresses of current slice
      __global A_type * arg1 = input + slice * LDA * LDB;
      __global A_type * arg2 = output + slice * LDA * LDB;

      // copy approximation part from arg1 to arg2
      int j;
      for (j = 0; j < block_size_1 - GROUP_SIZE_1; j += GROUP_SIZE_1)
      {
        int i;
        for (i = 0; i < block_size_0 - GROUP_SIZE_0; i += GROUP_SIZE_0)
        {
          const int index = index_base + j * LDA + i;
          arg1 [index] = arg2 [index];
        }
        if (i + local_c1 < block_size_0)
        {
          const int index = index_base + j * LDA + i;
          arg1 [index] = arg2 [index];
        }
      }
      if (j + local_c2 < block_size_1)
      {
        int i;
        for (i = 0; i < block_size_0 - GROUP_SIZE_0; i += GROUP_SIZE_0)
        {
          const int index = index_base + j * LDA + i;
          arg1 [index] = arg2 [index];
        }
        if (i + local_c1 < block_size_0)
        {
          const int index = index_base + j * LDA + i;
          arg1 [index] = arg2 [index];
        }
      }

    } // loop over slices

  } // if: choose active threads
    
}



/**
 * @author djoergens
 */
kernel void idwt2 (__global A_type * input,
          __constant B_type * _lpf,
          __constant B_type * _hpf,
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
    && get_global_id (2) < *num_slices)
  {

    const int block_size_0 = *line_length / (min (*line_length, (int) get_global_size (0))/GROUP_SIZE_0);
    const int block_size_1 = *line_length / (min (*line_length, (int) get_global_size (1))/GROUP_SIZE_1);
    const int border_block_size_0 = block_size_0 + 2*i_offset;
    const int border_block_size_1 = block_size_1 + 2*i_offset;

    const int local_c1 = get_local_id (0);
    const int local_c2 = get_local_id (1);

    __local A_type * tmp  = & loc_mem [get_local_id (2) * (border_block_size_0 * border_block_size_1 + border_block_size_0 * block_size_1)];
    __local A_type * tmp2 = & loc_mem [get_local_id (2) * (border_block_size_0 * border_block_size_1 + border_block_size_0 * block_size_1)
                                       + border_block_size_0 * border_block_size_1];

    const int upper_left = get_group_id (1) * block_size_1 / 2 * LDA
                         + get_group_id (0) * block_size_0 / 2;
    const int upper_left2 = get_group_id (1) * block_size_1 * LDA
                          + get_group_id (0) * block_size_0;

    // loop over slices
    for (int slice = get_group_id (2); slice < *num_slices; slice += get_global_size (2))
    {

      // start addresses of current slice
      __global A_type * arg1 = input + slice * LDA * LDB;
      __global A_type * arg2 = output + slice * LDA * LDB;

      barrier (CLK_LOCAL_MEM_FENCE);

      /////
      // memory transfer: global -> local
      /////
      iglobal2local (arg1, tmp, local_c1, local_c2, block_size_0, block_size_1, border_block_size_0, line_length);

      barrier (CLK_LOCAL_MEM_FENCE);

      /////
      // filter operations on rows
      /////
      ifiltering1_rows (local_c1, local_c2, tmp, tmp2, _lpf, _hpf, block_size_0, block_size_1,
                        border_block_size_0, border_block_size_1);
      ifiltering_rows (local_c1, local_c2, tmp, tmp2, _lpf, _hpf, block_size_0, block_size_1,
                       border_block_size_0, border_block_size_1);

      barrier (CLK_LOCAL_MEM_FENCE);

      /////
      // filter operations on columns
      /////
      ifiltering1_cols (local_c1, local_c2, tmp, tmp2, _lpf, _hpf, block_size_0, block_size_1,
                        border_block_size_0, border_block_size_1);
      ifiltering_cols (local_c1, local_c2, tmp, tmp2, _lpf, _hpf, block_size_0, block_size_1,
                       border_block_size_0, border_block_size_1);

      barrier (CLK_LOCAL_MEM_FENCE);

      /////
      // memory transfer: local -> global
      /////
      ilocal2global (arg2, tmp, upper_left2, local_c1, local_c2, block_size_0, block_size_1, line_length);

    } // loop over slices

  } // if: choose active threads

}