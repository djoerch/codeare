// created on Nov 11, 2013

# ifndef GROUP_SIZE
  # define GROUP_SIZE 32
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
  
  A_type * tmp = & loc_mem [0];
  A_type * tmp2 = & loc_mem [4 * GROUP_SIZE];

  const int start_pos = gid0 + gid1 * n;


  // loop over size of local memory (over slices)
    
    // load GROUP_SIZE elements to local memory
    if (
    tmp [get_local_id (0)] = arg1 [start_pos + get_local_id (0)];
    
    // perform calculation
    if (get_local_id (0) < GROUP_SIZE / 2)
    {
        
    }
    else
    {
        
    }
    
    // write back
    arg2 [start_pos + get_local_id (0)] = tmp [get_local_id (0)];
}