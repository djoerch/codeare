/* 
 * File:   oclAccessPattern.hpp
 * Author: djoergens
 *
 * Created on November 24, 2013, 11:56 AM
 */

#ifndef __OCL_ACCESS_PATTERN_HPP__

  #define	__OCL_ACCESS_PATTERN_HPP__

  class oclAccessPattern
  {
    
    private:
      
      size_t origin_ [3];
      size_t row_pitch_;
      size_t slice_pitch_;
      size_t region_ [3];
      
      oclAccessPattern ();
      
    public:
      
      oclAccessPattern (const size_t num_bytes)
       : origin_ ({0, 0, 0}),
         row_pitch_ (num_bytes), slice_pitch_ (num_bytes),
         region_ ({num_bytes, 1, 1})
      { }
      
      oclAccessPattern (const size_t origin [3],
                        const size_t row_pitch, const size_t slice_pitch,
                        const size_t region [3])
       : origin_ ({origin [0], origin [1], origin [2]}),
         row_pitch_ (row_pitch), slice_pitch_ (slice_pitch),
         region_ ({region [0], region [1], region [2]})
      { }
      
      oclAccessPattern (const oclAccessPattern & ap)
       : origin_ ({ap.origin_ [0], ap.origin_ [1], ap.origin_ [2]}),
         row_pitch_ (ap.row_pitch_), slice_pitch_ (ap.slice_pitch_),
         region_ ({ap.region_ [0], ap.region_ [1], ap.region_ [2]})
      { }
      
      /*
       * SETTER
       */
         
      size_t &
      Origin           (const int i)
      { return origin_ [i]; }
      
      size_t &
      RowPitch         ()
      { return row_pitch_; }
      
      size_t &
      SlicePitch       ()
      { return slice_pitch_; }
      
      size_t &
      Region           (const int i)
      { return region_ [i]; }

      /*
       * GETTER
       */
      
      const cl::size_t <3>
      Origin           () const
      {
        cl::size_t <3> to_ret;
        memcpy (&to_ret, origin_, 3 * sizeof (size_t));
        return to_ret;
      }
      
      size_t
      RowPitch         () const
      { return row_pitch_; }
      
      size_t
      SlicePitch       () const
      { return slice_pitch_; }
      
      const cl::size_t <3>
      Region           () const
      {
        cl::size_t <3> to_ret;
        memcpy (&to_ret, region_, 3 * sizeof (size_t));
        return to_ret;
      }
      
  };

#endif	/* __OCL_ACCESS_PATTERN_HPP__ */

