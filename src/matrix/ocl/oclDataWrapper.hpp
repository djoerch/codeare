# ifndef __OCL_DATA_WRAPPER_HPP__



  /************
   ** makros **
   ************/
  # define __OCL_DATA_WRAPPER_HPP__
  
  
  
  /**************
   ** includes **
   **************/
  
  // ocl
  # include "oclSettings.hpp"
  # include "oclDataObject.hpp"
  
  // std C++ libraries
  # include <sstream>
  
  
  
  /***************************
   ** class: oclDataWrapper **
   **   (derived)           **
   ***************************/
  template <class T>
  class oclDataWrapper : public oclDataObject
  {
    
    
    
    public:


      /**
       * @name              constructors and destructors
       */
      //@{
      
      
      /**
       * @brief             construtor
       *
       * @param             cpu_data    data in linear memory in RAM
       * @param             size        ... in bytes
       *
       */
      oclDataWrapper        (        T * const   cpu_data,
                             const int          num_elems)
                          : oclDataObject (num_elems, num_elems * sizeof (T)),
                            mp_cpu_data   (cpu_data),
                            mp_ev_mem_up (NULL), mp_ev_mem_down (NULL), mp_ev_compute (NULL),
                            m_vec_ev_list ()
      {
      
        print_optional ("Ctor: \"oclDataWrapper\"", VERB_HIGH);
                        
      }
      
      
      /**
       * @brief             "copy state" constructor
       *
       */
      oclDataWrapper        (                              T     * const  cpu_data,
                             const                       int             num_elems,
                                              oclDataWrapper <T> &             obj,
                                   oclDataObject :: CopyMode             copy_mode = NO_BUFFER)
                          : oclDataObject   (obj, copy_mode),
                            mp_cpu_data     (cpu_data)
      {
      
        print_optional ("Ctor: \"oclDataWrapper\" ... copied state", VERB_HIGH);

        /* check if sizes match (copying allowed) */
        if (num_elems != obj.getNumElems())
        {
        
          throw " *!* Error: Num_elems don't match! *!*";
        
        }
        
        /* copy buffer of obj on GPU */
        if (copy_mode == COPY_BUFFER && obj.bufferCopyable ())
        {
         
          // create buffer object
          oclConnection :: Instance () -> createBuffer (NULL, oclDataObject :: getSize (), oclDataObject :: getID ());
    
          // update memory state
          oclDataObject :: setLoaded ();

        }
              
      }
      
      
      /**
       * @brief             virtual destructor
       */
      virtual
      ~oclDataWrapper       ()
      {
      
        print_optional ("Dtor: \"oclDataWrapper\"", VERB_HIGH);
                
      }
        
      
      //@}
    
    
//      /**
//       * @brief             getter to cpu_data
//       */
//      virtual
//      double
//      getData               ();
      
      
      /**
       * @brief             print object's state to command line
       */
      virtual
      void
      print                 ();
      
      
      
    protected:

      
      /**
       * @brief             synchronize data on GPU with CPU
       *
       * @see               oclDataObject :: loadToGPU ()
       */
      virtual
      double
      loadToGPU             ();
      
      
      /**
       * @brief             synchronize data on CPU with GPU
       *
       * @see               oclDataObject :: loadToCPU ()
       */
      virtual
      double
      loadToCPU             ();
      
      
      /**********************
       ** member variables **
       **   (protected)    **
       **********************/
      T * const mp_cpu_data;            // pointer to cpu memory
      
      cl_event * mp_ev_mem_up, mp_ev_mem_down, mp_ev_compute;
      
      std::vector <cl::Event> m_vec_ev_list;
      
      
    
  }; // class oclDataWrapper
  
  
  
  /**************************
   ** function definitions **
   **************************/


  /**
   * @brief                 -- refer to class definition --
   */
  template <class T>
  double
  oclDataWrapper <T> ::
  loadToGPU                 ()
  {
      
    print_optional ("oclDataWrapper :: loadToGPU ()", VERB_HIGH);

    // memory transfer time
    double mem_time = .0;
    
    // buffer on GPU exists ?
    if (! oclDataObject :: getMemState ())
    {
      
//      if (oclDataObject :: APHost ().Size () != oclDataObject :: getSize ())
//        oclDataObject :: m_size = oclDataObject :: APHost ().Size ();
      
      // create buffer object
      try {
//        std::cout << " create buffer " << std::endl;
    	  oclConnection :: Instance () -> createBuffer ((T *) NULL, oclDataObject :: getSize (), oclDataObject :: getID ());
      } catch (oclError & oe) {
    	  throw oclError (oe, "oclDataWrapper :: loadToGPU ()");
      }
    
      // update memory state
      oclDataObject :: setLoaded ();
    
    }

    // precondition: buffer exists now!
    if (oclDataObject :: mp_modified [CPU])
    {
//      std::cout << " not loaded to GPU " << std::endl;
//      if (oclDataObject :: getSize () > 10)
//      {
//        std::cout << "TO GPU: *** " << std::endl;
//        std::cout << " getSize: " << oclDataObject :: getSize () << std::endl;
//        std::cout << " ap.size: " << oclDataObject :: APHost ().Size () << std::endl;
//        std::cout << " aphost: " << oclDataObject :: APHost () << std::endl;
//        std::cout << " apdevice: " << oclDataObject :: APDevice () << std::endl;
//      }
      
      // update GPU data
//      mem_time = oclConnection :: Instance () -> loadToGPU (mp_cpu_data, oclDataObject :: getSize (), oclDataObject :: getBuffer ());
            
      mem_time = oclConnection :: Instance () -> loadToGPU (mp_cpu_data, oclDataObject :: APHost (), oclDataObject :: APDevice (), oclDataObject :: getBuffer ());
      
      // update states
      oclDataObject :: setSync ();

    }
    
    return mem_time;
    
  }
  
  
  
  /**
   * @brief                 -- refer to class definition --
   */
  template <class T>
  double
  oclDataWrapper <T> ::
  loadToCPU                 ()
  {

    print_optional ("oclDataWrapper :: loadToCPU ()", VERB_HIGH);

    // memory transfer time
    double mem_time = .0;
    
    // buffer on GPU exists
    if (oclDataObject :: getMemState ())
    {
      
      if (oclDataObject :: mp_modified [oclDataObject::GPU])
      {
//      if (oclDataObject :: getSize () > 10)
//      {
//        std::cout << "TO CPU: *** " << std::endl;
//        std::cout << " getSize: " << oclDataObject :: getSize () << std::endl;
//        std::cout << " ap.size: " << oclDataObject :: APHost ().Size () << std::endl;
//        std::cout << " aphost: " << oclDataObject :: APHost () << std::endl;
//        std::cout << " apdevice: " << oclDataObject :: APDevice () << std::endl;
//      }
//        std::cout << " GPU modified -> download " << std::endl;
        // update CPU data
//        mem_time = oclConnection :: Instance () -> loadToCPU (oclDataObject :: mp_gpu_buffer, mp_cpu_data, oclDataObject :: m_size);
        mem_time = oclConnection :: Instance () -> loadToCPU (oclDataObject :: mp_gpu_buffer, mp_cpu_data, oclDataObject :: APHost (), oclDataObject :: APDevice ());
        
        // update states
        oclDataObject :: setSync ();
        
      }
      
    }
    else // ERROR if no buffer exists
    {
    
      std::stringstream tmp;
      tmp << "No buffer on GPU (id:" << oclDataObject :: getID () << ")";
      
      throw oclError (tmp.str (), "oclDataWrapper :: loadToCPU", oclError :: WARNING);
    
    }
    
    return mem_time;
    
  }
  
  
  
  /**
   * @brief             print object's state to command line
   */
  template <class T>
  void
  oclDataWrapper <T> ::
  print                 ()
  {
  
    /* call method from super class */
    oclDataObject :: print ();
    
    /* add own state */
    std::cout << " *%* -oclDataWrapper-" << std::endl;
    std::cout << " *%*  -> data: ";
    if (oclDataObject :: getNumElems () > 16)
      std::cout << " << too large >> " << std::endl;
    else
    {
      for (int i = 0; i < oclDataObject :: getNumElems (); i++)
        std::cout << mp_cpu_data [i] << " ";
      std::cout << std::endl;
    }

  }

  
  
# endif // __OCL_DATA_WRAPPER_HPP__
