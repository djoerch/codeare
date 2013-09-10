# ifndef __OCL_FUNCTION_OBJECT_HPP__



  /************
   ** makros **
   ************/
  # define __OCL_FUNCTION_OBJECT_HPP__



  /**************
   ** includes **
   **************/
  
  // ocl
  # include "oclSettings.hpp"
  # include "oclDataObject.hpp"
  
  

  
  /******************************
   ** class: oclFunctionObject **
   **   (abstract)             **
   ******************************/
  class oclFunctionObject
  {
  
  
    public:
    

      /**
       * @brief           default constructor
       */
      oclFunctionObject   ( oclDataObject * const * const pp_args,
                            int                           num_args )
                         : mpp_args    (pp_args),
                           m_num_args (num_args)
      {
        
        print_optional ("Ctor: \"oclFunctionObject\"", VERB_HIGH);
        
        /* TODO */
        
      }

      
      /**
       * @brief           virtual destructor
       */
      virtual
      ~oclFunctionObject  ()
      {
      
        print_optional ("Dtor: \"oclFunctionObject\"", VERB_HIGH);
        
        /* TODO */
        
      }


      virtual
      void
      run (const LaunchInformation &) = 0;
      
      
      virtual
      const ProfilingInformation
      getProfilingInformation (const int)
      const = 0;

  
    protected:


      /**
       * member variables
       */
       
      /* function arguments */
      oclDataObject     * const * mpp_args;
      int                         m_num_args;
  
 
  }; // class oclFunctionObject
  
  
  
# endif // __OCL_FUNCTION_OBJECT_HPP__
