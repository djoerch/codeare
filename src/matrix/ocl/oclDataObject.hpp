# ifndef __OCL_DATA_OBJECT_HPP__



  /************
   ** makros **
   ************/
  # define __OCL_DATA_OBJECT_HPP__



  /**************
   ** includes **
   **************/

  // ocl
  # include "oclObservableDataObject.hpp"
  # include "oclSettings.hpp"
  # include "oclConnection.hpp"

  // ViennaCL
  # include "/usr/include/viennacl/vector.hpp"
  # include "/usr/include/viennacl/matrix.hpp"



  /**************************
   ** class: oclDataObject **
   **  (declaration)       **
   **************************/
  class oclDataObject : public oclObservableDataObject
  {



    public:
 
 
 
       /***********
       ** enums **
       ***********/
      
      /**
       * @brief               specify desired copy mode
       */
      enum CopyMode
      {
      
        NO_BUFFER,
        KEEP_BUFFER,
        COPY_BUFFER
      
      }; 
 
 
      /***************************
       ** function declarations **
       ***************************/
  
 
      /**
       * @brief             pure virtual: prepare ()
       *
       *                    -- prepare data object for use on gpu --
       */
      virtual
      void
      prepare               () = 0;


      /**
       * @brief             print object's state to command line
       */
      virtual
      void
      print                 ();


      /**
       * @brief             -- return ViennaCl vector representation of data object --
       */
      template <class S>
      viennacl :: vector <S>
      getVCLVector          ( );


      /**
       * @brief             -- return ViennaCl matrix representation of data object --
       */
      template <class S,
                class R>
      viennacl :: matrix <S, R>
      getVCLMatrix          (int m, int n);


      /**
       * @name              getters (public)
       */
      //@{


      /**
       * @brief             return object's ID
       */
      inline
      oclObjectID
      getID                 () const;


      /**
       * @brief             return object's size
       */
      inline
      size_t
      getSize               () const;
      
     
      /**
       * @brief             return sync status (GPU <-> CPU memory)
       */
      inline
      bool
      getSyncState          () const;
      
      
      /**
       * @brief             return flag for device thats data has been modified
       */
      inline
      bool
      getCPUModified        () const;
      
      
      /**
       * @brief             is data available on GPU ?
       */
      inline
      bool
      getMemState           () const;
      
      
      /**
       * @brief             getter for buffer pointer
       */
      inline
      cl::Buffer *
      getBuffer             () const;
      
      
      /**
       * @brief             getter for number of elements
       */
      inline
      int
      getNumElems           () const;
      
      
      /**
       * @brief             is buffer of object copyable ?
       */
      inline
      bool
      bufferCopyable     () const;
      
      
      //@}
      
      
      /**
       * @name              setters (public)
       */
      //@{
      
      
      /**
       * @brief             tell object that data (CPU) was modified
       */
      inline
      void
      setCPUModified        ();
      
      
      /**
       * @brief             set a buffer object (to be used by oclConnection)
       */
      inline
      void
      setBuffer             (cl::Buffer * p_gpu_buffer);
            
      
      //@}
      
      
      /**
       * @name              constructors and destructors
       */
      //@{

      
      /**
       * @brief             default constructor
       *
       * @param             size - ... in bytes
       */
      oclDataObject         (const int num_elems, const size_t size)
                          : m_gpu_obj_id  (id_counter ++),      // init object ID
                            mp_gpu_buffer (NULL),               // init buffer (pointer: NULL)
                            m_size        (size),               // init size
                            m_num_elems   (num_elems),          // init number of elements
                            m_on_gpu      (false),              // data loaded to GPU on demand
                            mp_modified   ({true, false}),      // ... so data aren't sync
                            m_lock        (false),              // ... and there're no calculations on GPU
                            m_release_buffer (false)            // ... and there's no buffer to be released
      {
      
        print_optional ("Ctor: \"oclDataObject\" (%d)", m_gpu_obj_id, VERB_MIDDLE);
        
        // register data object at oclConnection
        oclConnection :: Instance () -> addDataObject (this);
        
      }
      
      
      /**
       * @brief             copy constructor
       *                      -> copy_mode (COPY_BUFFER): all state vars are copied
       *                      -> copy_mode (  NO_BUFFER): state "not on gpu"
       *                      -> copy_mode (KEEP_BUFFER): copy whole state,
       *                                                  notify copied object, that buffer was grabbed
       *                        !!! KEEP_BUFFER: intended for async mode !!!
       *
       * @param             pointer oclDataObject to copy
       */
      oclDataObject         (oclDataObject & obj, CopyMode copy_mode = NO_BUFFER)
                          : m_gpu_obj_id    (id_counter ++),
                            mp_gpu_buffer   (NULL),
                            m_size          (obj.m_size),
                            m_num_elems     (obj.m_num_elems),
                            m_on_gpu        (false),              // for addDataObject (...) !!!
                            m_lock          (obj.m_lock),
                            m_release_buffer(obj.m_release_buffer)
      {
      
        print_optional ("Ctor: \"oclDataObject\" (%d) ... copied state from %d", m_gpu_obj_id, obj.m_gpu_obj_id, VERB_MIDDLE);
        
        /* copy state of obj (if it's going to be copied) */
        if (copy_mode == COPY_BUFFER && obj.bufferCopyable ())
        {
        
          // deep copy of modification array
          new (mp_modified) bool [2];
          mp_modified [CPU] = obj.mp_modified [CPU];
          mp_modified [GPU] = obj.mp_modified [GPU];

        }
        else if (copy_mode == KEEP_BUFFER)
        {
          
          if (m_release_buffer != true)
          {
          
            /* grab buffer from obj */
            this -> mp_gpu_buffer = obj.mp_gpu_buffer;
            this -> m_on_gpu      = true;
            
            /* GPU data is modified (since calculations take place right now) */
            new (mp_modified) bool [2];
            mp_modified [CPU] = obj.mp_modified [CPU];
            mp_modified [GPU] = obj.mp_modified [GPU];
            
            /* notify, that obj's buffer was grabbed */
            obj.m_release_buffer = true;
            obj.mp_to_notify = this; /* TODO */

          }
          else
          {
            
            throw oclError (" oclDataObject is not copyable! ", "oclDataObject :: Ctor");
            
          }
          
        }
        else
        {

          // sync state
          new (mp_modified) bool [2];
          mp_modified [CPU] = true;
          mp_modified [GPU] = false;
          
          // lock state
          m_lock = false;

        }
        
        // register data object at oclConnection (IMPORTANT: after object state has been updated!!!)
        oclConnection :: Instance () -> addDataObject (this);
        
      }
      

      /**
       * @brief             virtual destructor
       *
       *                    important: buffer pointer is being
       *                               deleted by oclConnection if necessary
       */
      virtual
      ~oclDataObject        ()
      {
      
        print_optional ("Dtor: \"oclDataObject\" (%d)", m_gpu_obj_id, VERB_MIDDLE);
        
        // unregister data object at oclConnection
        oclConnection :: Instance () -> removeDataObject (this);
        
      }

      
      //@}
      
      

    protected:



      /**
       * @brief             synchronize data on GPU with CPU
       */
      virtual
      void
      loadToGPU             () = 0;


      /**
       * @brief             synchronize data on CPU with GPU
       */
      virtual
      void
      loadToCPU             () = 0;


      /**
       * @brief             release buffer
       */
      void
      releaseBuffer         ();


      /**
       * @name              setters (protected)
       */
      //@{
      
      
      /**
       * @brief             tell object that data is loaded to GPU
       */
      inline
      void
      setLoaded             ();
      
      
      /**
       * @brief             block data
       */
      inline
      void
      setLocked             ();
      
      
      /**
       * @brief             unblock data
       */
      inline
      void
      setUnlocked           ();
      
      
      /**
       * @brief             notify modification of data in GPU memory
       */
      inline
      void
      setGPUModified        ();
      
      
      /**
       * @brief             notify synchronicity of CPU and GPU data
       */
      inline
      void
      setSync               ();
      
      
      //@}
      
      
      /**
       * @name              getters (protected)
       */
      //@{
      
      
      /**
       * @brief             getter for m_busy
       */
public:      inline
      bool
      getLockState          () const;
protected:
      
      //@}
      
            
      /**********************
       ** member variables **
       **   (protected)    **
       **********************/
      const oclObjectID     m_gpu_obj_id;     // local id of particular object
             cl::Buffer   * mp_gpu_buffer;    // Buffer representation of data object
      
                 size_t     m_size;           // size of data (in bytes)
                    int     m_num_elems;      // number of elements
                 
                   bool     m_on_gpu;         // determines wether data is available on GPU
                   bool     mp_modified [2];  // specifies, if data on GPU and CPU are the same
                   bool     m_lock;           // determines wether data is used on GPU right now
                   
                   bool     m_release_buffer; // if set, buffer will be released at next call of finish ()
          oclDataObject   * mp_to_notify;      // object, the buffer is grabbed from
    


    private:
    
    
      /****************
       ** enum types **
       ****************/
      
      // indices of m_modified
      enum device_flag
      {CPU = 0, GPU = 1, NONE};
    
    
      static oclObjectID    id_counter;       // global counter to produce unique IDs
            


  }; // class oclDataObject



  /**
   *  initialise static class member
   */
  oclObjectID
  oclDataObject ::
  id_counter              = 0;



  /**************************
   ** function definitions **
   **************************/



  /**
   * @brief             print object's state to command line
   */
  void
  oclDataObject ::
  print                 ()
  {
  
    std::cout << " *%* oclDataObject (" << getID () << "):" << std::endl;
    std::cout << " *%*  -> Buffer: ";
    if (m_on_gpu)
      std::cout << " yes" << std::endl;
    else
      std::cout << " no" << std::endl;
    std::cout << " *%*  -> GPU: ";
    if (mp_modified [GPU])
      std::cout << " yes" << std::endl;
    else
      std::cout << " no" << std::endl;
    std::cout << " *%*  -> CPU: ";
    if (mp_modified [CPU])
      std::cout << " yes" << std::endl;
    else
      std::cout << " no" << std::endl;
    std::cout << " *%*  -> locked: ";
    if (m_lock)
      std::cout << " yes" << std::endl;
    else
      std::cout << " no" << std::endl;
    std::cout << " *%*  -> buffer grabbed: ";
    if (m_release_buffer)
      std::cout << " yes" << std::endl;
    else
      std::cout << " no" << std::endl;
    std::cout << " *%*  -> size (num_elems): " << m_size << " (" << m_num_elems << ")" << std::endl;
  
  }
  


  /**
   *                      -- refer to class definition --
   */
  template <class S>
  viennacl :: vector <S>
  oclDataObject ::
  getVCLVector            ( )
  {
  
    print_optional ("oclDataObject :: getVCLVector ()", VERB_HIGH);
    
    /* ensure data is available on GPU */
    loadToGPU ();
  
    /* create (and return) ViennaCl vector */
    return viennacl :: vector <S> ((*mp_gpu_buffer)(), m_num_elems);
      
  }
  
  
  
  /**
   *                      -- refer to class definition --
   */
  template <class S,
            class R>
  viennacl :: matrix <S, R>
  oclDataObject ::
  getVCLMatrix            (int m, int n)
  {  
    
    print_optional ("oclDataObject :: getVCLMatrix (m, n)", VERB_HIGH);
    
    /* check given sizes */
    if (m * n != m_num_elems)
    {
      stringstream msg;
      msg << " Given sizes m = " << m << ", n = " << n << " do not match number of elements!";
      throw oclError (msg.str (), "oclDataObject :: getVCLMatrix");
    }
      
    /* create (and return ViennaCl matrix */
    return viennacl :: matrix <S, R> ((*mp_gpu_buffer)(), m, n);

  }

  
  
  /**
   *                      -- refer to class definition --
   */
  void
  oclDataObject ::
  releaseBuffer           ()
  {
  
    /* clear connection to buffer in oclConnection */
    oclConnection :: Instance () -> releaseBuffer (this);

    /* clear reference to buffer */
    this -> mp_gpu_buffer = NULL;
        
    /* update state */
    m_release_buffer = false;
    m_on_gpu         = false;
    mp_modified [GPU] = false;
    mp_modified [CPU] = true;
    
    mp_to_notify -> setUnlocked ();
    mp_to_notify = NULL;
  
  }
  
  
  
  
  
        /*************
    --   ** getters **   --
         *************/
  
  
  /**
   * @brief               getter for member buffer
   */
  inline
  cl::Buffer *
  oclDataObject ::
  getBuffer               ()
  const
  {
    
    return mp_gpu_buffer;
  
  }
  
  
  /**
   * @brief               getter for object ID
   */
  inline
  oclObjectID
  oclDataObject ::
  getID                   ()
  const
  {
    
    return m_gpu_obj_id;
    
  }
  
  
  
  /**
   * @brief               getter of object size
   */
  inline
  size_t
  oclDataObject ::
  getSize                 ()
  const
  {
    
    return m_size;
    
  }
  
  
  
  /**
   * @brief               getter for memory state
   */
  inline
  bool
  oclDataObject ::
  getMemState             ()
  const
  {
  
    return m_on_gpu;
  
  }
  
  
  
  /**
   * @brief               getter for sync state
   */
  inline
  bool
  oclDataObject ::
  getSyncState            ()
  const
  {
    
    return     ! mp_modified [CPU]
            && ! mp_modified [GPU];
    
  }
  
  
  /**
   * @brief               return flag for device, that is modified
   */
  inline
  bool
  oclDataObject ::
  getCPUModified             ()
  const
  {
  
    return mp_modified [CPU];
    
  }
  
 
  
  /**
   * @brief               getter for m_lock
   */
  inline
  bool
  oclDataObject ::
  getLockState            ()
  const
  {
  
    return m_lock;
    
  }
  
  
  
  /**
   * @brief               getter for m_num_elems
   */
  inline
  int
  oclDataObject ::
  getNumElems             ()
  const
  {
  
    return m_num_elems;
  
  }
  
  
  
  /**
   * @brief             is buffer of object copyable ?
   *                    1.) no buffer exists on GPU
   *                    2.) CPU data is modified, so data on GPU is not up to date
   */
  inline
  bool
  oclDataObject ::
  bufferCopyable     ()
  const
  {
  
    return m_on_gpu  &&  ! mp_modified [CPU]  &&  ! m_release_buffer;
  
  }

  
  
  
        /*************
    --   ** setters **   --
         *************/
  
  
  
  /**
   * @brief               set CPU data modified
   */
  inline
  void
  oclDataObject ::
  setCPUModified          ()
  {
    
    mp_modified [CPU] = true;
    
  }
  
  
  
  /**
   * @brief               set GPU data modified
   */
  inline
  void
  oclDataObject ::
  setGPUModified          ()
  {
    
    mp_modified [GPU] = true;
    
  }
  
  
  
  /**
   * @brief               set m_on_gpu
   */
  inline
  void
  oclDataObject ::
  setLoaded               ()
  {
  
    print_optional (" ---------------------------> loaded (%d)", getID (), VERB_MIDDLE);
  
    m_on_gpu = true;
  
  }
  
  
  
  /**
   * @brief               lock data (no object internal access)
   */
  inline
  void
  oclDataObject ::
  setLocked               ()
  {
  
    m_lock = true;
  
  }
  
  
  
  /**
   * @brief               unlock data
   */
  inline
  void
  oclDataObject ::
  setUnlocked             ()
  {
  
    m_lock = false;
  
  }
  
  
  
  /**
   * @brief               notify synchronicity
   */
  inline
  void
  oclDataObject ::
  setSync                 ()
  {
  
      mp_modified [CPU]
    = mp_modified [GPU]
    = false;
  
  }
  
  
  
  /**
   * @brief                set buffer pointer
   */
  inline
  void
  oclDataObject ::
  setBuffer             (cl::Buffer * p_gpu_buffer)
  {
  
    if (! getMemState ())
    {
      mp_gpu_buffer = p_gpu_buffer;
    }
    else
    { /* TODO: use oclError !!! */
      std::cout << " *!* Caution: Buffer already exists! *!*" << std::endl;
      throw int (-1);
    }
  
  }
   
   
  
# endif // __OCL_DATA_OBJECT_HPP__
