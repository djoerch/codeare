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
    
  const int gid0 = get_group_id (0); // leading dimension
  const int gid1 = get_group_id (1);
  
  __local A_type * tmp = & loc_mem [0];
  __local A_type * tmp2 = & loc_mem [get_local_size (2)];

  const int base_index = gid0 + gid1 * *n;


  // loop over size of local memory (over slices)
    
    // load GROUP_SIZE elements to local memory
    const int index = base_index + (get_local_id (0) < offset ? (LDC-get_local_id (0)) : get_local_id (0) - offset) * LDA * LDB;
    tmp [get_local_id (2)] = arg1 [base_index + get_local_id (2)];//index];
    
    // perform calculation
    if (get_local_id (2) < GROUP_SIZE / 2)
    {
        tmp2 [get_local_id (2)] = tmp [get_local_id (2)];// + offset];
    }
    else
    {
        tmp2 [get_local_id (2)] = tmp [get_local_id (2)];// + offset];
    }
    
    // write back
    arg2 [base_index + get_local_id (2) * LDA * LDB] = 33.3;//get_local_id (0); //tmp2 [get_local_id (0)];
    
    arg2 [0] = 44.4;
    
}