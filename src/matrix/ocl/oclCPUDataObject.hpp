# ifndef __OCL_CPU_DATA_OBJECT_HPP__



  /************
   ** makros **
   ************/
  #define __OCL_CPU_DATA_OBJECT_HPP__
  
  
  
  /**************
   ** includes **
   **************/
  
  // ocl
  # include "oclSettings.hpp"
  # include "oclDataWrapper.hpp"
  
  
  
  /*****************************
   ** class: oclCPUDataObject **
   **   (derived)             **
   *****************************/
  template <class T>
  class oclCPUDataObject : public oclDataWrapper <T>
  {
  
    
    
    public:
      
      
      /**
       * @name                constructors and destructors
       */
      //@{
      
      
      /**
       * @brief               constructor
       *
       * @param               cpu_data  @see oclDataWrapper<T>
       * @param               size      @see ocldataWrapper<T>
       *
       */
      oclCPUDataObject        (           T   * const   cpu_data,
                               const    int   &        num_elems)
                            : oclDataWrapper <T> (cpu_data, num_elems),
                                    mp_pinned_mem (NULL)
      {
      
        print_optional ("Ctor: \"oclCPUDataObject\"", VERB_HIGH);
        
      }
      
      
      /**
       * @brief               "copy state" constructor
       */
/*      oclCPUDataObject        (                   T     * const    cpu_data,
                               const            int     &         num_elems,
                                     oclDataWrapper <T> &               obj,
                                               bool             keep_buffer = false)
                            : oclDataWrapper <T> (cpu_data, num_elems, obj, keep_buffer)
      {
      
        print_optional ("Ctor: \"oclCPUDataObject\" ... copied state", VERB_HIGH);
            
      }
*/      
      
      /**
       * @brief               virtual destructor
       */
      virtual
      ~oclCPUDataObject       ()
      {
      
        print_optional ("Dtor: \"oclCPUDataObject\"", VERB_MIDDLE);

        if (mp_pinned_mem != NULL)
          oclConnection :: Instance () -> unmapPointer (oclDataObject :: getID (), mp_pinned_mem);
        
      }

      
      //@}

    
      /**
       * @brief               inherited (oclDataObject)
       */
      virtual
      double
      prepare                 ();
      
      
      /**
       * @brief               inherited (oclObservableDataObject)
       */
      virtual
      double
      finish                  ();
      
      
      /**
       * @brief               inherited (oclDataWrapper)
       */
      virtual
      double
      getData                 ();
      
      
      /**
       * @brief               get pinned pointer
       */
      T *
      getPinnedPointer        ();
      
      
    private:
      
      T * mp_pinned_mem;
    
     
  }; // class oclCPUDataObject
  
  
  
  /**************************
   ** function definitions **
   **************************/
  
  template <class T>
  T *
  oclCPUDataObject <T> ::
  getPinnedPointer            ()
  {
    
    print_optional ("oclCPUDataObject :: getPinnedPointer (%d)", oclDataObject :: getID (), VERB_MIDDLE);
        
    if (mp_pinned_mem == NULL)
    {
      
      oclConnection :: Instance () -> createPinnedBuffer (mp_pinned_mem, oclDataObject :: getSize (), oclDataObject :: getID ());
      mp_pinned_mem = oclConnection :: Instance () -> getMappedPointer <T> (oclDataObject :: getID ());
    }
    
    return mp_pinned_mem;
    
  }
  
  
  /**
   * @brief                   prepare data object for use on GPU
   *                           -- load data to GPU if needed --
   *                          !! precondition: data not in use !!
   */
  template <class T>
  double
  oclCPUDataObject <T> ::
  prepare                     ()
  {

    print_optional ("oclCPUDataObject::prepare (%d)", oclDataObject :: getID (), VERB_MIDDLE);
  
    // set status: calculating (set available via finish ())
    oclDataObject :: setLocked ();
    
    // synchronize GPU data / load to GPU
//    oclDataWrapper <T> :: loadToGPU ();
    
    // notify modification of GPU data
//    oclDataObject :: setGPUModified ();

  }
  
  
  
  /**
   * @brief                   load data to CPU memory (since it's a CPU object)
   */
  template <class T>
  double
  oclCPUDataObject <T> ::
  finish                      ()
  {

    print_optional ("oclCPUDataObject::finish", VERB_HIGH);

    // update data state: available for use
    oclDataObject :: setUnlocked ();

    // copy data to CPU memory
    this -> getData ();

  }
  
  
  
  /**
   * @brief                   copy data to CPU memory
   */
  template <class T>
  double
  oclCPUDataObject <T> ::
  getData                     ()
  {
  
    print_optional ("oclCPUDataObject::getData", VERB_HIGH);
    
    // check wether data is available or used on GPU
    if (oclDataObject :: getLockState ())
    {
    
      /* throw error */
      throw oclError ("Calculating on GPU ... data not available!", "oclCPU_DataObject :: getData");
    
    }
    else
    {
    
      try
      {

        // synchronize CPU data with GPU
        oclDataWrapper <T> :: loadToCPU ();

      }
      catch (const oclError & err)
      {
      
        print_optional (oclError (err, "oclCPUDataObject :: getData"), VERB_LOW);
      
      }

    }    
    
  }
  
  
  
# endif // __OCL_CPU_DATA_OBJECT_HPP__
