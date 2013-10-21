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

# ifndef ODD_FILTER
  # define ODD_FILTER 1
# endif

# ifndef I_OFFSET
  __constant const int i_offset = FL-1;
  # define I_OFFSET
# endif


void
iglobal2local       (__global A_type * arg1, __local A_type * tmp,
                     const int local_c1, const int local_c2,
                     const int block_size_0, const int block_size_1,
                     __constant int * line_length)
{
  
  const int j_max = block_size_1 / 2 + i_offset;
  const int i_max = block_size_0 / 2 + i_offset;

  const int border_block_size_0 = block_size_0 + 2*i_offset;

//  const int shift = - i_offset/2;

  const int shift_lo = - i_offset;
  const int shift_hi = 0;//i_offset;

  const int upper_left = get_group_id (1) * block_size_1 / 2 * LDA
                       + get_group_id (0) * block_size_0 / 2;
                    
  const int c1_base_lo = block_size_0/2 * get_group_id (0) + local_c1 + shift_lo;
  const int c2_base_lo = block_size_1/2 * get_group_id (1) + local_c2 + shift_lo;

  const int c1_base_hi = block_size_0/2 * get_group_id (0) + local_c1 + shift_hi + 1;
  const int c2_base_hi = block_size_1/2 * get_group_id (1) + local_c2 + shift_hi + 1;

//  if (local_c1 < i_max && local_c2 < j_max)
//  {

  ///////////
  // part: LL
  ///////////
  int j;
  for (j = 0; j < j_max - GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left + (local_c2 + j + shift_lo) * LDA + local_c1 + i + shift_lo;
      index = index + (c1_base_lo + i < 0 ? *line_length/2 : 0)
                    + (c2_base_lo + j < 0 ? *line_length/2*LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
        = arg1 [index];
    }
    if (i + local_c1 < i_max)
    {
      int index = upper_left + (local_c2 + j + shift_lo) * LDA + local_c1 + i + shift_lo;
      index = index + (c1_base_lo + i < 0 ? *line_length/2 : 0)
                    + (c2_base_lo + j < 0 ? *line_length/2*LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
        = arg1 [index];
    }
  }
  if (j + local_c2 < j_max)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left + (local_c2 + j + shift_lo) * LDA + local_c1 + i + shift_lo;
      index = index + (c1_base_lo + i < 0 ? *line_length/2 : 0)
                    + (c2_base_lo + j < 0 ? *line_length/2*LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
        = arg1 [index];
    }
    if (i + local_c1 < i_max)
    {
      int index = upper_left + (local_c2 + j + shift_lo) * LDA + local_c1 + i + shift_lo;
      index = index + (c1_base_lo + i < 0 ? *line_length/2 : 0)
                    + (c2_base_lo + j < 0 ? *line_length/2*LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
        = arg1 [index];
    }
  }

  ///////////
  // part: LH
  ///////////
  for (j = j_max; j < 2 * j_max - GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left + (*line_length/2 + local_c2 + j - j_max + shift_hi) * LDA + local_c1 + i + shift_lo;
      index = index + (c1_base_lo + i < 0 ? *line_length/2 : 0)
                    + (c2_base_hi + j - j_max > *line_length/2 ? -*line_length/2*LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
        = arg1 [index];
    }
    if (i + local_c1 < i_max)
    {
      int index = upper_left + (*line_length/2 + local_c2 + j - j_max + shift_hi) * LDA + local_c1 + i + shift_lo;
      index = index + (c1_base_lo + i < 0 ? *line_length/2 : 0)
                    + (c2_base_hi + j - j_max > *line_length/2 ? -*line_length/2*LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
        = arg1 [index];
    }
  }
  if (j + local_c2 < 2 * j_max)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left + (*line_length/2 + local_c2 + j - j_max + shift_hi) * LDA + local_c1 + i + shift_lo;
      index = index + (c1_base_lo + i < 0 ? *line_length/2 : 0)
                    + (c2_base_hi + j - j_max > *line_length/2 ? -*line_length/2*LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
        = arg1 [index];
    }
    if (i + local_c1 < i_max)
    {
      int index = upper_left + (*line_length/2 + local_c2 + j - j_max + shift_hi) * LDA + local_c1 + i + shift_lo;
      index = index + (c1_base_lo + i < 0 ? *line_length/2 : 0)
                    + (c2_base_hi + j - j_max > *line_length/2 ? -*line_length/2*LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i]
        = arg1 [index];
    }
  }

  ///////////
  // part: HL
  ///////////
  for (j = 0; j < j_max - GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left + (local_c2 + j + shift_lo) * LDA + local_c1 + i + *line_length/2 + shift_hi;
      index = index + (c1_base_hi + i > *line_length/2 ? -*line_length/2 : 0)
                    + (c2_base_lo + j < 0 ? *line_length/2*LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i + i_max]
        = arg1 [index];
    }
    if (i + local_c1 < i_max)
    {
      int index = upper_left + (local_c2 + j + shift_lo) * LDA + local_c1 + i + *line_length/2 + shift_hi;
      index = index + (c1_base_hi + i > *line_length/2 ? -*line_length/2 : 0)
                    + (c2_base_lo + j < 0 ? *line_length/2*LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i + i_max]
        = arg1 [index];
    }
  }
  if (j + local_c2 < j_max)
  {
    int i;
    for (i = 0; i < i_max-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left + (local_c2 + j + shift_lo) * LDA + local_c1 + i + *line_length/2 + shift_hi;
      index = index + (c1_base_hi + i > *line_length/2 ? -*line_length/2 : 0)
                    + (c2_base_lo + j < 0 ? *line_length/2*LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i + i_max]
        = arg1 [index];
    }
    if (i + local_c1 < i_max)
    {
      int index = upper_left + (local_c2 + j + shift_lo) * LDA + local_c1 + i + *line_length/2 + shift_hi;
      index = index + (c1_base_hi + i > *line_length/2 ? -*line_length/2 : 0)
                    + (c2_base_lo + j < 0 ? *line_length/2*LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i + i_max]
        = arg1 [index];
    }
  }

  ///////////
  // part: HH
  ///////////
  for (j = j_max; j < 2 * j_max - GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left + (*line_length/2 + local_c2 + j - j_max + shift_hi) * LDA + local_c1 + i + *line_length/2 + shift_hi;
      index = index + (c1_base_hi + i > *line_length/2 ? -*line_length/2 : 0)
                    + (c2_base_hi + j - j_max > *line_length/2 ? -*line_length/2*LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i + i_max]
        = arg1 [index];
    }
    if (i + local_c1 < i_max)
    {
      int index = upper_left + (*line_length/2 + local_c2 + j - j_max + shift_hi) * LDA + local_c1 + i + *line_length/2 + shift_hi;
      index = index + (c1_base_hi + i > *line_length/2 ? -*line_length/2 : 0)
                    + (c2_base_hi + j - j_max > *line_length/2 ? -*line_length/2*LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i + i_max]
        = arg1 [index];
    }
  }
  if (j + local_c2 < 2 * j_max)
  {
    int i;
    for (i = 0; i < border_block_size_0/2-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left + (*line_length/2 + local_c2 + j - j_max + shift_hi) * LDA + local_c1 + i + *line_length/2 + shift_hi;
      index = index + (c1_base_hi + i > *line_length/2 ? -*line_length/2 : 0)
                    + (c2_base_hi + j - j_max > *line_length/2 ? -*line_length/2*LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i + i_max]
        = arg1 [index];
    }
    if (i + local_c1 < i_max)
    {
      int index = upper_left + (*line_length/2 + local_c2 + j - j_max + shift_hi) * LDA + local_c1 + i + *line_length/2 + shift_hi;
      index = index + (c1_base_hi + i > *line_length/2 ? -*line_length/2 : 0)
                    + (c2_base_hi + j - j_max > *line_length/2 ? -*line_length/2*LDA : 0);
      tmp [(local_c2 + j) * border_block_size_0 + local_c1 + i + i_max]
        = arg1 [index];
    }
  }

//  }

}


void
ilocal2global       (__global A_type * arg2, __local A_type * tmp2,
                     const int upper_left2, const int local_c1, const int local_c2,
                     const int block_size_0,
                     const int block_size_1, __constant int * line_length)
{
  
  const int ld = block_size_0;

  const int shift = 0;//-(FL+1)/2;//0;//FL-1;

  const int c1_base = get_group_id (0) * block_size_0 + local_c1 + shift;
  const int c2_base = get_group_id (1) * block_size_1 + local_c2 + shift;

//  if (local_c1 < block_size_0 && local_c2 < block_size_1)
//  {

  int j;
  for (j = 0; j < block_size_1-GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < block_size_0-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left2 + (local_c2 + j + shift) * LDA + local_c1 + i + shift;
      index = index + (c1_base + i < 0 ? *line_length : 0)
                    + (c2_base + j < 0 ? *line_length * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * ld + local_c1 + i];
    }
    if (i + local_c1 < block_size_0)
    {
      int index = upper_left2 + (local_c2 + j + shift) * LDA + local_c1 + i + shift;
      index = index + (c1_base + i < 0 ? *line_length : 0)
                    + (c2_base + j < 0 ? *line_length * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * ld + local_c1 + i];
    }
  }
  if (j + local_c2 < block_size_1)
  {
    int i;
    for (i = 0; i < block_size_0-GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
      int index = upper_left2 + (local_c2 + j + shift) * LDA + local_c1 + i + shift;
      index = index + (c1_base + i < 0 ? *line_length : 0)
                    + (c2_base + j < 0 ? *line_length * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * ld + local_c1 + i];
    }
    if (i + local_c1 < block_size_0)
    {
      int index = upper_left2 + (local_c2 + j + shift) * LDA + local_c1 + i + shift;
      index = index + (c1_base + i < 0 ? *line_length : 0)
                    + (c2_base + j < 0 ? *line_length * LDA : 0);
      arg2 [index] = tmp2 [(local_c2 + j) * ld + local_c1 + i];
    }
  }

//  }

}



void
iconv_step1_lo              (const int index1, const int index2,
                             __constant A_type * filter,
                             __local A_type * tmp, __local A_type * tmp2,
                             const int increment)
{

  A_type sum1 = 0, sum2 = 0;
  # pragma unroll
  for (int k = 0; k < FL/2; k ++)
  {
    sum1 += tmp [index1 - k * increment] * filter [2 * k];
    sum2 += tmp [index1 - k * increment] * filter [(2 * k + 1)];
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
                             __constant A_type * filter,
                             __local A_type * tmp, __local A_type * tmp2,
                             const int increment)
{

  A_type sum1 = 0, sum2 = 0;
  # pragma unroll
  for (int k = 0; k < FL/2; k ++)
  {
    sum1 += tmp [index1 - k * increment] * filter [2 * k];
    sum2 += tmp [index1 - k * increment] * filter [(2 * k + 1)];
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
                             __constant A_type * filter,
                             __local A_type * tmp, __local A_type * tmp2,
                             const int increment)
{

  A_type sum1 = 0, sum2 = 0;
  # pragma unroll
  for (int k = 0; k < FL/2; k ++)
  {
    sum1 += tmp [index1 + k * increment] * filter [2 * k];
    sum2 += tmp [index1 + k * increment] * filter [2 * k + 1];
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
                          __constant A_type * _lpf, __constant A_type * _hpf,
                          const int block_size_0, const int block_size_1,
                          const int border_block_size_0, const int border_block_size_1)
{

  const int j_max = block_size_1 / 2;
  const int i_max = block_size_0 / 2 + i_offset;

  // ROWS //

//  const int shift = (FL>2 ? FL/2 - 1 : 0);

  const int start_lo = FL-1;
  const int start_hi = 0;

//  if (local_c1 < block_size_0/2 + i_offset && local_c2 < block_size_1/2)
//  {

  // LL //
  int j;
  for (j = 0; j < j_max - GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
        int index1 = (j + local_c2 + start_lo) * border_block_size_0 + i + local_c1;
        int index2 = (j + 2 * local_c2) * border_block_size_0 + i + local_c1;
        iconv_step1_lo (index1, index2, _lpf, tmp, tmp2, border_block_size_0);
    }
    if (i + local_c1 < i_max)
    {
        int index1 = (j + local_c2 + start_lo) * border_block_size_0 + i + local_c1;
        int index2 = (j + 2 * local_c2) * border_block_size_0 + i + local_c1;
        iconv_step1_lo (index1, index2, _lpf, tmp, tmp2, border_block_size_0);
    }
  }
  if (j + local_c2 < j_max)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
        int index1 = (j + local_c2 + start_lo) * border_block_size_0 + i + local_c1;
        int index2 = (j + 2 * local_c2) * border_block_size_0 + i + local_c1;
        iconv_step1_lo (index1, index2, _lpf, tmp, tmp2, border_block_size_0);
    }
    if (i + local_c1 < i_max)
    {
        int index1 = (j + local_c2 + start_lo) * border_block_size_0 + i + local_c1;
        int index2 = (j + 2 * local_c2) * border_block_size_0 + i + local_c1;
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
        int index1 = (j + local_c2 + start_lo) * border_block_size_0 + i + local_c1;
        int index2 = (j + 2 * local_c2) * border_block_size_0 + i + local_c1;
        iconv_step1_lo (index1, index2, _lpf, tmp, tmp2, border_block_size_0);
    }
    if (i + local_c1 < i_end)
    {
        int index1 = (j + local_c2 + start_lo) * border_block_size_0 + i + local_c1;
        int index2 = (j + 2 * local_c2) * border_block_size_0 + i + local_c1;
        iconv_step1_lo (index1, index2, _lpf, tmp, tmp2, border_block_size_0);
    }
  }
  if (j + local_c2 < j_max)
  {
    int i;
    for (i = i_start; i < i_end - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
        int index1 = (j + local_c2 + start_lo) * border_block_size_0 + i + local_c1;
        int index2 = (j + 2 * local_c2) * border_block_size_0 + i + local_c1;
        iconv_step1_lo (index1, index2, _lpf, tmp, tmp2, border_block_size_0);
    }
    if (i + local_c1 < i_end)
    {
        int index1 = (j + local_c2 + start_lo) * border_block_size_0 + i + local_c1;
        int index2 = (j + 2 * local_c2) * border_block_size_0 + i + local_c1;
        iconv_step1_lo (index1, index2, _lpf, tmp, tmp2, border_block_size_0);
    }
  }

//  }

}





void
ifiltering_rows          (const int local_c1, const int local_c2,
                          __local A_type * tmp, __local A_type * tmp2,
                          __constant A_type * _lpf, __constant A_type * _hpf,
                          const int block_size_0, const int block_size_1,
                          const int border_block_size_0, const int border_block_size_1)
{

  const int j_max = block_size_1 / 2;
  const int i_max = block_size_0 / 2 + i_offset;

  // ROWS //

//  if ((local_c1 < block_size_0/2 + i_offset) && (local_c2 < block_size_1/2))
//  {

//  const int start_lo = FL-1;
  const int start_hi = 0;

  // LH //
  int j;
  for (j = 0; j < j_max - GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < j_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
        int index1 = (j + local_c2 + (j_max + i_offset) + start_hi) * border_block_size_0 + i + local_c1;
        int index2 = (j + 2 * local_c2) * border_block_size_0 + i + local_c1;
        iconv_step2_hi (index1, index2, _hpf, tmp, tmp2, border_block_size_0);
    }
    if (i + local_c1 < i_max)
    {
        int index1 = (j + local_c2 + (j_max + i_offset) + start_hi) * border_block_size_0 + i + local_c1;
        int index2 = (j + 2 * local_c2) * border_block_size_0 + i + local_c1;
        iconv_step2_hi (index1, index2, _hpf, tmp, tmp2, border_block_size_0);
    }
  }
  if (j + local_c2 < j_max)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
        int index1 = (j + local_c2 + (j_max + i_offset) + start_hi) * border_block_size_0 + i + local_c1;
        int index2 = (j + 2 * local_c2) * border_block_size_0 + i + local_c1;
        iconv_step2_hi (index1, index2, _hpf, tmp, tmp2, border_block_size_0);
    }
    if (i + local_c1 < i_max)
    {
        int index1 = (j + local_c2 + (j_max + i_offset) + start_hi) * border_block_size_0 + i + local_c1;
        int index2 = (j + 2 * local_c2) * border_block_size_0 + i + local_c1;
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
        int index1 = (j + local_c2 + (j_max + i_offset) + start_hi) * border_block_size_0 + i + local_c1;
        int index2 = (j + 2 * local_c2) * border_block_size_0 + i + local_c1;
        iconv_step2_hi (index1, index2, _hpf, tmp, tmp2, border_block_size_0);
    }
    if (i + local_c1 < i_end)
    {
        int index1 = (j + local_c2 + (j_max + i_offset) + start_hi) * border_block_size_0 + i + local_c1;
        int index2 = (j + 2 * local_c2) * border_block_size_0 + i + local_c1;
        iconv_step2_hi (index1, index2, _hpf, tmp, tmp2, border_block_size_0);
    }
  }
  if (j + local_c2 < j_max)
  {
    int i;
    for (i = i_start; i < i_end - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
        int index1 = (j + local_c2 + (j_max + i_offset) + start_hi) * border_block_size_0 + i + local_c1;
        int index2 = (j + 2 * local_c2) * border_block_size_0 + i + local_c1;
        iconv_step2_hi (index1, index2, _hpf, tmp, tmp2, border_block_size_0);
    }
    if (i + local_c1 < i_end)
    {
        int index1 = (j + local_c2 + (j_max + i_offset) + start_hi) * border_block_size_0 + i + local_c1;
        int index2 = (j + 2 * local_c2) * border_block_size_0 + i + local_c1;
        iconv_step2_hi (index1, index2, _hpf, tmp, tmp2, border_block_size_0);
    }
  }

//  }

}



void
ifiltering1_cols         (const int local_c1, const int local_c2,
                          __local A_type * tmp, __local A_type * tmp2,
                          __constant A_type * _lpf, __constant A_type * _hpf,
                          const int block_size_0, const int block_size_1,
                          const int border_block_size_0, const int border_block_size_1)
{

  const int j_max = block_size_1;
  const int i_max = block_size_0 / 2;

  // COLS //

//  if (local_c1 < block_size_0/2 && local_c2 < block_size_1)
//  {

//  const int shift = (FL>2 ? FL/2 - 1 : 0);

  const int start_lo = FL-1;
  const int start_hi = 0;

  // L //
  int j;
  for (j = 0; j < j_max - GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
        int index1 = (j + local_c2) * border_block_size_0 + i + local_c1 + start_lo;
        int index2 = (j + local_c2) * block_size_0 + i + 2 * (local_c1);
        iconv_step1_lo (index1, index2, _lpf, tmp2, tmp, 1);
//        tmp2 [index2] = tmp2 [index2 + 1] = tmp [index1];
    }
    if (i + local_c1 < i_max)
    {
        int index1 = (j + local_c2) * border_block_size_0 + i + local_c1 + start_lo;
        int index2 = (j + local_c2) * block_size_0 + i + 2 * (local_c1);
        iconv_step1_lo (index1, index2, _lpf, tmp2, tmp, 1);
//        tmp2 [index2] = tmp2 [index2 + 1] = tmp [index1];
    }
  }
  if (j + local_c2 < j_max)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
        int index1 = (j + local_c2) * border_block_size_0 + i + local_c1 + start_lo;
        int index2 = (j + local_c2) * block_size_0 + i + 2 * (local_c1);
        iconv_step1_lo (index1, index2, _lpf, tmp2, tmp, 1);
//        tmp2 [index2] = tmp2 [index2 + 1] = tmp [index1];
    }
    if (i + local_c1 < i_max)
    {
        int index1 = (j + local_c2) * border_block_size_0 + i + local_c1 + start_lo;
        int index2 = (j + local_c2) * block_size_0 + i + 2 * (local_c1);
        iconv_step1_lo (index1, index2, _lpf, tmp2, tmp, 1);
//        tmp2 [index2] = tmp2 [index2 + 1] = tmp [index1];
    }
  }

//  }

}




void
ifiltering_cols          (const int local_c1, const int local_c2,
                          __local A_type * tmp, __local A_type * tmp2,
                          __constant A_type * _lpf, __constant A_type * _hpf,
                          const int block_size_0, const int block_size_1,
                          const int border_block_size_0, const int border_block_size_1)
{

  const int j_max = block_size_1;
  const int i_max = block_size_0 / 2;

  // COLS //

//  if (local_c1 < block_size_0/2 && local_c2 < block_size_1)
//  {

  const int start_hi = 0;

  // H //

  int j;
  for (j = 0; j < j_max - GROUP_SIZE_1; j += GROUP_SIZE_1)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
        int index1 = (j + local_c2) * border_block_size_0 + i + local_c1 + (i_max + i_offset) + start_hi;
        int index2 = (j + local_c2) * block_size_0 + i + 2 * (local_c1);
        iconv_step2_hi (index1, index2, _hpf, tmp2, tmp, 1);
    }
    if (i + local_c1 < i_max)
    {
        int index1 = (j + local_c2) * border_block_size_0 + i + local_c1 + (i_max + i_offset) + start_hi;
        int index2 = (j + local_c2) * block_size_0 + i + 2 * (local_c1);
        iconv_step2_hi (index1, index2, _hpf, tmp2, tmp, 1);
    }
  }
  if (j + local_c2 < j_max)
  {
    int i;
    for (i = 0; i < i_max - GROUP_SIZE_0; i += GROUP_SIZE_0)
    {
        int index1 = (j + local_c2) * border_block_size_0 + i + local_c1 + (i_max + i_offset) + start_hi;
        int index2 = (j + local_c2) * block_size_0 + i + 2 * (local_c1);
        iconv_step2_hi (index1, index2, _hpf, tmp2, tmp, 1);
    }
    if (i + local_c1 < i_max)
    {
        int index1 = (j + local_c2) * border_block_size_0 + i + local_c1 + (i_max + i_offset) + start_hi;
        int index2 = (j + local_c2) * block_size_0 + i + 2 * (local_c1);
        iconv_step2_hi (index1, index2, _hpf, tmp2, tmp, 1);
    }
  }

//  }

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

  if (get_global_id (0) < 2* *line_length
    && get_global_id (1) < 2* *line_length)
  {

  const int block_size_0 = *line_length / (min (*line_length, (int) get_global_size (0))/GROUP_SIZE_0);
  const int block_size_1 = *line_length / (min (*line_length, (int) get_global_size (1))/GROUP_SIZE_1);
  const int border_block_size_0 = block_size_0 + 2*i_offset;
  const int border_block_size_1 = block_size_1 + 2*i_offset;

  const int local_c1 = get_local_id (0);
  const int local_c2 = get_local_id (1);

  __local A_type * tmp  = & loc_mem [0];
  __local A_type * tmp2 = & loc_mem [(block_size_0 + 2*i_offset) * (block_size_1 + 2*i_offset)];

  const int upper_left = get_group_id (1) * block_size_1 / 2 * LDA
                       + get_group_id (0) * block_size_0 / 2;
  const int upper_left2 = get_group_id (1) * block_size_1 * LDA
                        + get_group_id (0) * block_size_0;

  iglobal2local (arg1, tmp, local_c1, local_c2, block_size_0, block_size_1, line_length);

  barrier (CLK_LOCAL_MEM_FENCE);
 
  ifiltering1_rows (local_c1, local_c2, tmp, tmp2, _lpf, _hpf, block_size_0, block_size_1,
                    border_block_size_0, border_block_size_1);
  ifiltering_rows (local_c1, local_c2, tmp, tmp2, _lpf, _hpf, block_size_0, block_size_1,
                   border_block_size_0, border_block_size_1);
                
  barrier (CLK_LOCAL_MEM_FENCE);
                
  ifiltering1_cols (local_c1, local_c2, tmp, tmp2, _lpf, _hpf, block_size_0, block_size_1,
                    border_block_size_0, border_block_size_1);
  ifiltering_cols (local_c1, local_c2, tmp, tmp2, _lpf, _hpf, block_size_0, block_size_1,
                   border_block_size_0, border_block_size_1);

  barrier (CLK_LOCAL_MEM_FENCE);

  ilocal2global (arg2, tmp, upper_left2, local_c1, local_c2, block_size_0, block_size_1, line_length);

  }

}