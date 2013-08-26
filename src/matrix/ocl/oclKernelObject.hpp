# ifndef __OCL_KERNEL_OBJECT_HPP__




  /************
   ** makros **
   ************/
  # define __OCL_KERNEL_OBJECT_HPP__




  /**************
   ** includes **
   **************/
  
  // C++ std lib
  # include <string>

  // ocl
  # include "oclSettings.hpp"
  # include "oclDataObject.hpp"
  # include "oclFunctionObject.hpp"

    
  
    
  /****************************
   ** class: oclKernelObject **
   **   (derived)            **
   ****************************/
  class oclKernelObject : public oclFunctionObject
  {
  
    
    
    protected:
    
    
      /**********************
       ** member variables **
       **********************/
   
      std::vector <std::string> m_kernel_names;
      std::vector <cl::Event>   m_events;
    
    
   
    public:
   
    
      /**
       * @name              Constructors and destructors.
       */
      //@{
      
      
      /**
       * @brief             default constructor
       */
      oclKernelObject       ( const   std::string &               kernel_name,
                                    oclDataObject * const * const     pp_args,
                                              int                    num_args )
                           : oclFunctionObject (pp_args, num_args),
                             m_kernel_names    (1, kernel_name),
                             m_events          ()
                            
      {
      
        print_optional ("Ctor: \"oclKernelObject\"", v_level);
        
        /* TODO */
        
      }
    
                           
                           
      /**
       * @brief             default constructor
       */
      oclKernelObject       ( const std::vector   <std::string> &               kernel_names,
                                    oclDataObject               * const * const      pp_args,
                                              int                                   num_args )
                           : oclFunctionObject (pp_args, num_args),
                             m_kernel_names    (kernel_names),
                             m_events          ()
                            
      {
      
        print_optional ("Ctor: \"oclKernelObject\"", v_level);
        
        /* TODO */
        
      }
      
      
                           
      /**
       * @brief             virtual destructor
       */
      virtual
      ~oclKernelObject      ()
      {
      
        print_optional ("Dtor: \"oclKernelObject\"", v_level);

        /* TODO */
        
      }

    
      //@}


      /**
       * @brief             execute kernel
       *
       * @see               defined in oclFunctionObject
       */
      virtual
      void
      run                   ();
      
      
      /**
       * @brief             Retrieve profiling information on kernel execution.
       */
      virtual
      const ProfilingInformation
      getProfilingInformation (const int i)
      const;
      
      

    private:
    
      /* private member for verbosity level of class */
      static const VerbosityLevel v_level;
      
  
  
  }; // class oclKernelObject
  
  
  
  /*************************************
   ** initialize static class members **
   *************************************/
  const VerbosityLevel oclKernelObject :: v_level = global_verbosity [OCL_KERNEL_OBJECT];
  
  
  
  /**************************
   ** function definitions **
   **************************/


  
  /**
   * @brief                 prepare arguments, run kernel, finish arguments
   */
  void
  oclKernelObject ::
  run                       ()
  {
  
    // oclConnection for reuse in this function
    oclConnection * oclCon = oclConnection :: Instance ();
    
    for (std::vector<std::string>::const_iterator it_kernel_name = m_kernel_names.begin (); it_kernel_name != m_kernel_names.end (); it_kernel_name++)
    {

      print_optional("oclKernelObject :: run ( \"", it_kernel_name->c_str(), "\" )", v_level);

      // activate kernel
      oclCon -> activateKernel(*it_kernel_name);

      // prepare kernel arguments (load to gpu)
      for (int i = 0; i < m_num_args; i++)
      {

        // prepare argument
        mpp_args [i] -> prepare();

        // register argument at kernel
        oclCon -> setKernelArg(i, mpp_args [i]);

      }

      // run kernel
      cl::NDRange global_dims(128, 512);
      cl::NDRange local_dims(128, 1);
      cl::Event event = oclCon -> runKernel(global_dims, local_dims);

      m_events.push_back (event);
      
    }
    
    for (int i = 0; i < m_events.size (); i++)
    {
      m_events [i].wait ();
    }
    
    // perhaps get data
    for (int i = 0; i < m_num_args; i++)
    {
      mpp_args [i] -> finish ();
    }
        
  }
  
  
  const ProfilingInformation
  oclKernelObject ::
  getProfilingInformation     (const int i)
  const
  {
    return oclConnection :: Instance () -> getProfilingInformation (m_events [i]);
  }
  
  
  
  
# endif /* __OCL_KERNEL_OBJECT_HPP__ */
