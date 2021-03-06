# ifndef __OCL_TRAITS_HPP__




  /************
   ** makros **
   ************/
  # define __OCL_TRAITS_HPP__
  
  

  
  /**************
   ** includes **
   **************/
  
  // ocl
  # include "oclGPUDataObject.hpp"
  # include "oclCPUDataObject.hpp"
  # include "oclLocalMemObject.hpp"
  # include "oclSettings.hpp"
  
  // AMD BLAS
  # include <clAmdBlas.h>
      
  

  /******************************
   ** struct: elem_type_traits **
   **     (base struct)        **
   ******************************/
  template <class T>
  struct elem_type_traits
  {
    
    /* -- */
    
  }; // struct elem_type_traits <T>


  /******************************
   ** struct: elem_type_traits **
   **     (spec: float)        **
   ******************************/
  template <>
  struct elem_type_traits <float>
  {
  
    public:

      typedef float elem_type;
      typedef float value_type;
      
      static inline
      const char *
      print_elem_type       ( )
      {
        return "float";
      }
      
  
  }; // struct elem_type_traits <float>


  /******************************
   ** struct: elem_type_traits **
   **     (spec: cxfl)         **
   ******************************/
  template <>
  struct elem_type_traits <cxfl>
  {
  
    public:

      typedef cxfl elem_type;
      typedef float value_type;
      
      static inline
      const char *
      print_elem_type       ( )
      {
        return "cxfl";
      }
  
  }; // struct elem_type_traits <cxfl>


  /******************************
   ** struct: elem_type_traits **
   **     (spec: double)       **
   ******************************/
  template <>
  struct elem_type_traits <double>
  {
  
    public:

      typedef double elem_type;
      typedef double value_type;

      static inline
      const char *
      print_elem_type       ( )
      {
        return "double";
      }
  
  }; // struct elem_type_traits <double>


  /******************************
   ** struct: elem_type_traits **
   **     (spec: cxdb)         **
   ******************************/
  template <>
  struct elem_type_traits <cxdb>
  {
  
    public:

      typedef cxdb elem_type;
      typedef double value_type;
      
      static inline
      const char *
      print_elem_type       ( )
      {
        return "cxdb";
      }
  
  }; // struct elem_type_traits <cxdb>


  /******************************
   ** struct: elem_type_traits **
   **     (spec: size_t)       **
   ******************************/
  template <>
  struct elem_type_traits <size_t>
  {
  
    public:

      typedef size_t elem_type;
      typedef size_t value_type;
      
      static inline
      const char *
      print_elem_type       ( )
      {
        return "size_t";
      }
  
  }; // struct elem_type_traits <size_t>
  

  /******************************
   ** struct: elem_type_traits **
   **     (spec: bool)         **
   ******************************/  
  template <>
  struct elem_type_traits <bool>
  {
  
    public:

      typedef bool elem_type;
      typedef bool value_type;
      
      static inline
      const char *
      print_elem_type       ( )
      {
        return "bool";
      }
  
  }; // struct elem_type_traits <bool>


  /******************************
   ** struct: elem_type_traits **
   **     (spec: int)          **
   ******************************/  
  template <>
  struct elem_type_traits <int>
  {
  
    public:

      typedef int elem_type;
      typedef int value_type;
      
      static inline
      const char *
      print_elem_type       ( )
      {
        return "int";
      }
  
  }; // struct elem_type_traits <int>


  
  /***************************
   ** struct: oclOperations **
   **   (base struct)       **
   ***************************/
  template <class      T,                        class      S =                T,
            class trait1 = elem_type_traits <T>, class trait2 = elem_type_traits <S> >
  struct oclOperations
  {
    

    private:

    
      /**********************
       ** type definitions **
       **********************/
      typedef typename trait1 :: elem_type elem_type;
      typedef typename trait2 :: elem_type scalar_type;
      
      
      /*********************
       ** local variables **
       *********************/
      
      /* verbosity of operators */
      static
      const VerbosityLevel op_v_level = VERB_LOW;


      /**
       * @name                        basic operator algos
       */
      //@{
      
      
      /**
       * @brief                       run given function object
       */
      static inline
      const oclError &
      ocl_run_func_obj            (      oclFunctionObject * const func_obj,
                                   const LaunchInformation               lc = LaunchInformation (128, 1, 128, 1))
      {
      
        try
        {

          // activate precision mode for type elem_type
          oclConnection :: Instance () -> activate <elem_type, scalar_type> ();

          // execute function object
          func_obj -> run (lc);

        }
        catch (oclError & err)
        {
        
          std::stringstream msg;
          msg << "oclOperations <" << trait1 :: print_elem_type () << ", "
                                   << trait2 :: print_elem_type () << "> :: ocl_run_func_obj";
        
          throw oclError (err, msg.str ().c_str ());
        
        }
      
      }
    
    
      /**
       * @brief                       execute specified kernel with 3 arguments
       */
      static inline
      const oclError &
      ocl_basic_operator_kernel_3              ( const          char * const kernel_name,
                                                       oclDataObject * const        arg1,
                                                       oclDataObject * const        arg2,
                                                       oclDataObject * const      result,
                                                                 int           num_elems )
      {
    
        // number of kernel arguments
        const int num_args = 4;
    
        // create array of function arguments
        oclDataObject ** args = (oclDataObject **) malloc (num_args * sizeof (oclDataObject *));
        args [0] = arg1;
        args [1] = arg2;
        args [2] = result;
        args [3] = new oclGPUDataObject <int> (& num_elems, 1);

        // create function object
        oclFunctionObject * op_obj = oclConnection :: Instance ()
                                        -> makeFunctionObject <elem_type, scalar_type>
                                              (kernel_name, args, num_args, oclConnection::KERNEL, oclConnection::SYNC);
        
        // execute function object
        ocl_run_func_obj (op_obj);

        // clear memory
        delete op_obj;      
        delete args [3];
        free (args);
    
      }


      /**
       * @brief                       execute specified kernel with 3 arguments and 1 scalar
       */
      static inline
      ProfilingInformation
      ocl_basic_operator_kernel_31              ( const std::string & kernel_name,
                                                       oclDataObject * const        arg1,
                                                       oclDataObject * const        arg2,
                                                       oclDataObject * const      arg3,
                                                      int               s1,
                                                 const LaunchInformation              lc )
      {
    
        // number of kernel arguments
        const int num_args = 4;
    
        // create array of function arguments
        oclDataObject ** args = (oclDataObject **) malloc (num_args * sizeof (oclDataObject *));
        args [0] = arg1;
        args [1] = arg2;
        args [2] = arg3;
        args [3] = new oclGPUDataObject <int> (& s1, 1);

        // create function object
        oclFunctionObject * op_obj = oclConnection :: Instance ()
                                        -> makeFunctionObject <elem_type, scalar_type>
                                              (kernel_name, args, num_args, oclConnection::KERNEL, oclConnection::SYNC);
        
        // execute function object
        ocl_run_func_obj (op_obj, lc);

        // retrieve profiling information
        ProfilingInformation pi = op_obj -> getProfilingInformation(0);
        
        // clear memory
        delete op_obj;
        delete args [3];
        free (args);
        
        return pi;
    
      }
      
      
            /**
       * @brief                       execute specified kernel with 2 arguments and 1 scalar
       */
      static inline
      ProfilingInformation
      ocl_basic_operator_kernel_21              ( const std::string & kernel_name,
                                                       oclDataObject * const        arg1,
                                                       oclDataObject * const        arg2,
                                                   int                           s1,
                                                 const LaunchInformation              lc )
      {
    
        // number of kernel arguments
        const int num_args = 3;
    
        // create array of function arguments
        oclDataObject ** args = (oclDataObject **) malloc (num_args * sizeof (oclDataObject *));
        args [0] = arg1;
        args [1] = arg2;
        args [2] = new oclGPUDataObject <int> (& s1, 1);

        // create function object
        oclFunctionObject * op_obj = oclConnection :: Instance ()
                                        -> makeFunctionObject <elem_type, scalar_type>
                                              (kernel_name, args, num_args, oclConnection::KERNEL, oclConnection::SYNC);
        
        // execute function object
        ocl_run_func_obj (op_obj, lc);

        // retrieve profiling information
        ProfilingInformation pi = op_obj -> getProfilingInformation ();
        
        // clear memory
        delete op_obj;
        delete args [2];
        free (args);
        
        return pi;
    
      }
      
      
      /**
       * @brief                     execute specified kernel with 5 arguments and 5 scalars
       */
      static inline
      ProfilingInformation
      ocl_basic_operator_kernel_55  ( const   std::string               &        kernel_name,
                                            oclDataObject               * const         arg1,
                                            oclDataObject               * const         arg2,
                                            oclDataObject               * const         arg3,
                                            oclDataObject               * const       result,
                                            oclDataObject               * const      loc_mem,
                                                      int                                 s1,
                                                      int                                 s2,
                                                      int                                 s3,
                                                      int                                 s4,
                                                      int                       loc_mem_size,
                                      const LaunchInformation                             lc)
      {

        // number of kernel arguments
        const int num_args = 10;

        // create array of function arguments
        oclDataObject ** args = (oclDataObject **) malloc (num_args * sizeof (oclDataObject *));
        args [0] = arg1;
        args [1] = arg2;
        args [2] = arg3;
        args [3] = result;
        args [4] = loc_mem;
        args [5] = new oclGPUDataObject <int> (& s1, 1);
        args [6] = new oclGPUDataObject <int> (& s2, 1);
        args [7] = new oclGPUDataObject <int> (& s3, 1);
        args [8] = new oclGPUDataObject <int> (& s4, 1);
        args [9] = new oclGPUDataObject <int> (& loc_mem_size, 1);
        
        // create function object
        oclFunctionObject * op_obj = oclConnection :: Instance ()
                                        -> makeFunctionObject <elem_type, scalar_type>
                                              (kernel_name, args, num_args, oclConnection::KERNEL, oclConnection::SYNC);
        
        // execute function object
        ocl_run_func_obj (op_obj, lc);
        
        // retrieve profiling information
        ProfilingInformation pi = op_obj -> getProfilingInformation ();

        // clear memory
        delete op_obj;
        delete args [5];
        delete args [6];
        delete args [7];
        delete args [8];
        delete args [9];
        free (args);

        return pi;
        
      }
      
      
      /**
       * @brief                     execute specified kernel with 5 arguments and 6 scalars
       */
      static inline
      ProfilingInformation
      ocl_basic_operator_kernel_10 ( const   std::string               &        kernel_name,
                                           oclDataObject               * const         arg1,
                                           oclDataObject               * const         arg2,
                                           oclDataObject               * const         arg3,
                                           oclDataObject               * const       result,
                                           oclDataObject               * const      loc_mem,
                                           oclDataObject               * const         arg4,
                                           oclDataObject               * const         arg5,
                                           oclDataObject               * const         arg6,
                                           oclDataObject               * const         arg7,
                                           oclDataObject               * const         arg8,
                                     const LaunchInformation                             lc)
      {

        // number of kernel arguments
        const int num_args = 10;

        // create array of function arguments
        oclDataObject ** args = (oclDataObject **) malloc (num_args * sizeof (oclDataObject *));
        args [0] = arg1;
        args [1] = arg2;
        args [2] = arg3;
        args [3] = result;
        args [4] = loc_mem;
        args [5] = arg4;
        args [6] = arg5;
        args [7] = arg6;
        args [8] = arg7;
        args [9] = arg8;
        
        // create function object
        oclFunctionObject * op_obj = oclConnection :: Instance ()
                                        -> makeFunctionObject <elem_type, scalar_type>
                                              (kernel_name, args, num_args, oclConnection::KERNEL, oclConnection::SYNC);
        
        // execute function object
        ocl_run_func_obj (op_obj, lc);
        
        // retrieve profiling information
        ProfilingInformation pi = op_obj -> getProfilingInformation ();

        // clear memory
        delete op_obj;
        free (args);

        return pi;
        
      }


      /**
       * @brief                     execute specified kernel with 5 arguments and 6 scalars
       */
      static inline
      ProfilingInformation
      ocl_basic_operator_kernel_11 ( const   std::string               &        kernel_name,
                                           oclDataObject               * const         arg1,
                                           oclDataObject               * const         arg2,
                                           oclDataObject               * const         arg3,
                                           oclDataObject               * const       result,
                                           oclDataObject               * const      loc_mem,
                                           oclDataObject               * const         arg4,
                                           oclDataObject               * const         arg5,
                                           oclDataObject               * const         arg6,
                                           oclDataObject               * const         arg7,
                                           oclDataObject               * const         arg8,
                                           oclDataObject               * const         arg9,
                                     const LaunchInformation                             lc)
      {

        // number of kernel arguments
        const int num_args = 11;

        // create array of function arguments
        oclDataObject ** args = (oclDataObject **) malloc (num_args * sizeof (oclDataObject *));
        args [0] = arg1;
        args [1] = arg2;
        args [2] = arg3;
        args [3] = result;
        args [4] = loc_mem;
        args [5] = arg4;
        args [6] = arg5;
        args [7] = arg6;
        args [8] = arg7;
        args [9] = arg8;
        args [10] = arg9;
        
        // create function object
        oclFunctionObject * op_obj = oclConnection :: Instance ()
                                        -> makeFunctionObject <elem_type, scalar_type>
                                              (kernel_name, args, num_args, oclConnection::KERNEL, oclConnection::SYNC);
        
        // execute function object
        ocl_run_func_obj (op_obj, lc);
        
        // retrieve profiling information
        ProfilingInformation pi = op_obj -> getProfilingInformation ();

        // clear memory
        delete op_obj;
        free (args);

        return pi;
        
      }




      /**
       * @brief                     execute specified kernel with 5 arguments and 6 scalars
       */
      static inline
      ProfilingInformation
      ocl_basic_operator_kernel_56  ( const   std::string               &        kernel_name,
                                            oclDataObject               * const         arg1,
                                            oclDataObject               * const         arg2,
                                            oclDataObject               * const         arg3,
                                            oclDataObject               * const       result,
                                            oclDataObject               * const      loc_mem,
                                                      int                                 s1,
                                                      int                                 s2,
                                                      int                                 s3,
                                                      int                                 s4,
                                                      int                                 s5,
                                                      int                       loc_mem_size,
                                      const LaunchInformation                             lc)
      {

        // number of kernel arguments
        const int num_args = 11;

        // create array of function arguments
        oclDataObject ** args = (oclDataObject **) malloc (num_args * sizeof (oclDataObject *));
        args [0] = arg1;
        args [1] = arg2;
        args [2] = arg3;
        args [3] = result;
        args [4] = loc_mem;
        args [5] = new oclGPUDataObject <int> (& s1, 1);
        args [6] = new oclGPUDataObject <int> (& s2, 1);
        args [7] = new oclGPUDataObject <int> (& s3, 1);
        args [8] = new oclGPUDataObject <int> (& s4, 1);
        args [9] = new oclGPUDataObject <int> (& s5, 1);
        args[10] = new oclGPUDataObject <int> (& loc_mem_size, 1);
        
        // create function object
        oclFunctionObject * op_obj = oclConnection :: Instance ()
                                        -> makeFunctionObject <elem_type, scalar_type>
                                              (kernel_name, args, num_args, oclConnection::KERNEL, oclConnection::SYNC);
        
        // execute function object
        ocl_run_func_obj (op_obj, lc);
        
        // retrieve profiling information
        ProfilingInformation pi = op_obj -> getProfilingInformation ();

        // clear memory
        delete op_obj;
        delete args [5];
        delete args [6];
        delete args [7];
        delete args [8];
        delete args [9];
        delete args [10];
        free (args);

        return pi;
        
      }
      
      
      /**
       * @brief                     execute specified kernel with 5 arguments and 7 scalars
       */
      static inline
      ProfilingInformation
      ocl_basic_operator_kernel_57  ( const   std::string               &        kernel_name,
                                            oclDataObject               * const         arg1,
                                            oclDataObject               * const         arg2,
                                            oclDataObject               * const         arg3,
                                            oclDataObject               * const       result,
                                            oclDataObject               * const      loc_mem,
                                                      int                                 s1,
                                                      int                                 s2,
                                                      int                                 s3,
                                                      int                                 s4,
                                                      int                                 s5,
                                                      int                                 s6,
                                                      int                       loc_mem_size,
                                      const LaunchInformation                             lc)
      {

        // number of kernel arguments
        const int num_args = 12;

        // create array of function arguments
        oclDataObject ** args = (oclDataObject **) malloc (num_args * sizeof (oclDataObject *));
        args [0] = arg1;
        args [1] = arg2;
        args [2] = arg3;
        args [3] = result;
        args [4] = loc_mem;
        args [5] = new oclGPUDataObject <int> (& s1, 1);
        args [6] = new oclGPUDataObject <int> (& s2, 1);
        args [7] = new oclGPUDataObject <int> (& s3, 1);
        args [8] = new oclGPUDataObject <int> (& s4, 1);
        args [9] = new oclGPUDataObject <int> (& s5, 1);
        args[10] = new oclGPUDataObject <int> (& s6, 1);
        args[11] = new oclGPUDataObject <int> (& loc_mem_size, 1);
        
        // create function object
        oclFunctionObject * op_obj = oclConnection :: Instance ()
                                        -> makeFunctionObject <elem_type, scalar_type>
                                              (kernel_name, args, num_args, oclConnection::KERNEL, oclConnection::SYNC);
        
        // execute function object
        ocl_run_func_obj (op_obj, lc);
        
        // retrieve profiling information
        ProfilingInformation pi = op_obj -> getProfilingInformation ();

        // clear memory
        delete op_obj;
        delete args [5];
        delete args [6];
        delete args [7];
        delete args [8];
        delete args [9];
        delete args [10];
        delete args [11];
        free (args);

        return pi;
        
      }
      
            /**
       * @brief                     execute specified kernel with 5 arguments and 7 scalars
       */
      static inline
      ProfilingInformation
      ocl_basic_operator_kernel_58  ( const   std::string               &        kernel_name,
                                            oclDataObject               * const         arg1,
                                            oclDataObject               * const         arg2,
                                            oclDataObject               * const         arg3,
                                            oclDataObject               * const       result,
                                            oclDataObject               * const      loc_mem,
                                                      int                                 s1,
                                                      int                                 s2,
                                                      int                                 s3,
                                                      int                                 s4,
                                                      int                                 s5,
                                                      int                                 s6,
                                                      int                                 s7,
                                                      int                                 s8,
                                      const LaunchInformation                             lc)
      {

        // number of kernel arguments
        const int num_args = 13;

        // create array of function arguments
        oclDataObject ** args = (oclDataObject **) malloc (num_args * sizeof (oclDataObject *));
        args [0] = arg1;
        args [1] = arg2;
        args [2] = arg3;
        args [3] = result;
        args [4] = loc_mem;
        args [5] = new oclGPUDataObject <int> (& s1, 1);
        args [6] = new oclGPUDataObject <int> (& s2, 1);
        args [7] = new oclGPUDataObject <int> (& s3, 1);
        args [8] = new oclGPUDataObject <int> (& s4, 1);
        args [9] = new oclGPUDataObject <int> (& s5, 1);
        args[10] = new oclGPUDataObject <int> (& s6, 1);
        args[11] = new oclGPUDataObject <int> (& s7, 1);
        args[12] = new oclGPUDataObject <int> (& s8, 1);
        
        // create function object
        oclFunctionObject * op_obj = oclConnection :: Instance ()
                                        -> makeFunctionObject <elem_type, scalar_type>
                                              (kernel_name, args, num_args, oclConnection::KERNEL, oclConnection::SYNC);
        
        // execute function object
        ocl_run_func_obj (op_obj, lc);
        
        // retrieve profiling information
        ProfilingInformation pi = op_obj -> getProfilingInformation ();

        // clear memory
        delete op_obj;
        delete args [5];
        delete args [6];
        delete args [7];
        delete args [8];
        delete args [9];
        delete args [10];
        delete args [11];
        delete args [12];
        free (args);

        return pi;
        
      }
      
      
      /**
       * @brief                     execute specified kernel with 6 arguments and 5 scalars
       */
      static inline
      std::vector <ProfilingInformation>
      ocl_basic_operator_kernel_65  ( const   std::vector <std::string> &       kernel_names,
                                            oclDataObject               * const         arg1,
                                            oclDataObject               * const         arg2,
                                            oclDataObject               * const         arg3,
                                            oclDataObject               * const         arg4,
                                            oclDataObject               * const       result,
                                            oclDataObject               * const      loc_mem,
                                                      int                                 s1,
                                                      int                                 s2,
                                                      int                                 s3,
                                                      int                                 s4,
                                                      int                       loc_mem_size,
                                      const LaunchInformation                             lc)
      {

        // number of kernel arguments
        const int num_args = 11;

        // create array of function arguments
        oclDataObject ** args = (oclDataObject **) malloc (num_args * sizeof (oclDataObject *));
        args [0] = arg1;
        args [1] = arg2;
        args [2] = arg3;
        args [3] = arg4;
        args [4] = result;
        args [5] = loc_mem;
        args [6] = new oclGPUDataObject <int> (& s1, 1);
        args [7] = new oclGPUDataObject <int> (& s2, 1);
        args [8] = new oclGPUDataObject <int> (& s3, 1);
        args [9] = new oclGPUDataObject <int> (& s4, 1);
        args [10] = new oclGPUDataObject <int> (& loc_mem_size, 1);
        
        // create function object
        oclFunctionObject * op_obj = oclConnection :: Instance ()
                                        -> makeFunctionObject <elem_type, scalar_type>
                                              (kernel_names, args, num_args, oclConnection::KERNEL, oclConnection::SYNC);
        
        // execute function object
        ocl_run_func_obj (op_obj, lc);
        
        // retrieve profiling information
        std::vector <ProfilingInformation> vec_pi;
        for (int i = 0; i < kernel_names.size (); i++)
          vec_pi.push_back (op_obj -> getProfilingInformation(i));

        // clear memory
        delete op_obj;
        delete args [6];
        delete args [7];
        delete args [8];
        delete args [9];
        delete args [10];
        free (args);

        return vec_pi;
        
      }

      
      /**
       * @brief                       execute specified kernel with 2 arguments and 5 scalars
       */
      static inline
      ProfilingInformation
      ocl_basic_operator_kernel_25             ( const          char * const kernel_name,
                                                       oclDataObject * const        arg1,
                                                       oclDataObject * const        arg2,
                                                                 int                  s1,
                                                                 int                  s2,
                                                                 int                  s3,
                                                                 int                  s4,
                                                                 int                  s5,
                                                LaunchInformation                     lc)
      {
    
        // number of kernel arguments
        const int num_args = 7;
    
        // create array of function arguments
        oclDataObject ** args = (oclDataObject **) malloc (num_args * sizeof (oclDataObject *));
        args [0] = arg1;
        args [1] = arg2;
        args [2] = new oclGPUDataObject <int> (& s1, 1);
        args [3] = new oclGPUDataObject <int> (& s2, 1);
        args [4] = new oclGPUDataObject <int> (& s3, 1);
        args [5] = new oclGPUDataObject <int> (& s4, 1);
        args [6] = new oclGPUDataObject <int> (& s5, 1);

        // create function object
        oclFunctionObject * op_obj = oclConnection :: Instance ()
                                        -> makeFunctionObject <elem_type, scalar_type>
                                              (kernel_name, args, num_args, oclConnection::KERNEL, oclConnection::SYNC);
        

        // execute function object
        ocl_run_func_obj (op_obj, lc);
        
        // retrieve profiling information
        ProfilingInformation pi = op_obj -> getProfilingInformation(0);

        // clear memory
        delete op_obj;
        delete args [2];
        delete args [3];
        delete args [4];
        delete args [5];
        delete args [6];
        free (args);

        return pi;
        
      }


      /**
       * @brief                       execute specified kernel with 2 arguments and 6 scalars
       */
      static inline
      std::vector <ProfilingInformation>
      ocl_basic_operator_kernel_26             ( const          char * const kernel_name,
                                                       oclDataObject * const        arg1,
                                                       oclDataObject * const        arg2,
                                                                 int                  s1,
                                                                 int                  s2,
                                                                 int                  s3,
                                                                 int                  s4,
                                                                 int                  s5,
                                                                 int                  s6,
                                                LaunchInformation                     lc)
      {
    
        // number of kernel arguments
        const int num_args = 8;
    
        // create array of function arguments
        oclDataObject ** args = (oclDataObject **) malloc (num_args * sizeof (oclDataObject *));
        args [0] = arg1;
        args [1] = arg2;
        args [2] = new oclGPUDataObject <int> (& s1, 1);
        args [3] = new oclGPUDataObject <int> (& s2, 1);
        args [4] = new oclGPUDataObject <int> (& s3, 1);
        args [5] = new oclGPUDataObject <int> (& s4, 1);
        args [6] = new oclGPUDataObject <int> (& s5, 1);
        args [7] = new oclGPUDataObject <int> (& s6, 1);

        // create function object
        oclFunctionObject * op_obj = oclConnection :: Instance ()
                                        -> makeFunctionObject <elem_type, scalar_type>
                                              (kernel_name, args, num_args, oclConnection::KERNEL, oclConnection::SYNC);
        

        // execute function object
        ocl_run_func_obj (op_obj, lc);
        
        // retrieve profiling information
        std::vector <ProfilingInformation> vec_pi;
        vec_pi.push_back (op_obj -> getProfilingInformation(0));

        // clear memory
        delete op_obj;
        delete args [2];
        delete args [3];
        delete args [4];
        delete args [5];
        delete args [6];
        delete args [7];
        free (args);

        return vec_pi;
        
      }
      
      
      
            /**
       * @brief                       execute specified kernel with 2 arguments and 4 scalars
       */
      static inline
      ProfilingInformation
      ocl_basic_operator_kernel_24             ( const          char * const kernel_name,
                                                       oclDataObject * const        arg1,
                                                       oclDataObject * const        arg2,
                                                                 int                  s1,
                                                                 int                  s2,
                                                                 int                  s3,
                                                                 int                  s4,
                                                LaunchInformation                     lc)
      {
    
        // number of kernel arguments
        const int num_args = 6;
    
        // create array of function arguments
        oclDataObject ** args = (oclDataObject **) malloc (num_args * sizeof (oclDataObject *));
        args [0] = arg1;
        args [1] = arg2;
        args [2] = new oclGPUDataObject <int> (& s1, 1);
        args [3] = new oclGPUDataObject <int> (& s2, 1);
        args [4] = new oclGPUDataObject <int> (& s3, 1);
        args [5] = new oclGPUDataObject <int> (& s4, 1);

        // create function object
        oclFunctionObject * op_obj = oclConnection :: Instance ()
                                        -> makeFunctionObject <elem_type, scalar_type>
                                              (kernel_name, args, num_args, oclConnection::KERNEL, oclConnection::SYNC);
        

        // execute function object
        ocl_run_func_obj (op_obj, lc);
        
        // retrieve profiling information
        ProfilingInformation pi;
        pi = op_obj -> getProfilingInformation(0);

        // clear memory
        delete op_obj;
        delete args [2];
        delete args [3];
        delete args [4];
        delete args [5];
        free (args);

        return pi;
        
      }
      

      /**
       * @brief                       execute specified kernel with 3 arguments and 5 scalars
       */
      static inline
      const oclError &
      ocl_basic_operator_kernel_35             ( const          char * const kernel_name,
                                                       oclDataObject * const        arg1,
                                                       oclDataObject * const        arg2,
                                                       oclDataObject * const      result,
                                                                 int                  s1,
                                                                 int                  s2,
                                                                 int                  s3,
                                                                 int                  s4,
                                                                 int                  s5 )
      {
    
        // number of kernel arguments
        const int num_args = 8;
    
        // create array of function arguments
        oclDataObject ** args = (oclDataObject **) malloc (num_args * sizeof (oclDataObject *));
        args [0] = arg1;
        args [1] = arg2;
        args [2] = result;
        args [3] = new oclGPUDataObject <int> (& s1, 1);
        args [4] = new oclGPUDataObject <int> (& s2, 1);
        args [5] = new oclGPUDataObject <int> (& s3, 1);
        args [6] = new oclGPUDataObject <int> (& s4, 1);
        args [7] = new oclGPUDataObject <int> (& s5, 1);

        // create function object
        oclFunctionObject * op_obj = oclConnection :: Instance ()
                                        -> makeFunctionObject <elem_type, scalar_type>
                                              (kernel_name, args, num_args, oclConnection::KERNEL, oclConnection::SYNC);
        
        // execute function object
        ocl_run_func_obj (op_obj);

        // clear memory
        delete op_obj;
        delete args [3];
        delete args [4];
        delete args [5];
        delete args [6];
        delete args [7];
        free (args);
    
      }
    
    
      /**
       * @brief                      execute specified kernel with 1 arguments and 1 scalar
       */
      static inline
      const oclError &
      ocl_basic_operator_kernel_11              ( const          char * const kernel_name,
                                                        oclDataObject * const        arg1,
                                                          scalar_type                arg2,
                                                                  int           num_elems )
      {
    
        // number of kernel arguments
        const int num_args = 3;
    
        // create array of function arguments
        oclDataObject ** args = (oclDataObject **) malloc (num_args * sizeof (oclDataObject *));
        args [0] = arg1;
        args [1] = new oclGPUDataObject <scalar_type> (&      arg2, 1);
        args [2] = new oclGPUDataObject         <int> (& num_elems, 1);

        // create function object
        oclFunctionObject * op_obj = oclConnection :: Instance ()
                                        -> makeFunctionObject <elem_type, scalar_type>
                                              (kernel_name, args, num_args, oclConnection::KERNEL, oclConnection::SYNC);
        
        // execute function object
        ocl_run_func_obj (op_obj);

        // clear memory      
        delete op_obj;
        delete args [1];
        delete args [2];
        free (args);
    
      }
    
    
      /**
       * @brief                      execute specified kernel with 2 arguments and 1 scalar
       */
      static inline
      const oclError &
      ocl_basic_operator_kernel_21              ( const          char * const kernel_name,
                                                        oclDataObject * const        arg1,
                                                          scalar_type                arg2,
                                                        oclDataObject * const        arg3,
                                                                  int           num_elems )
      {
    
        // number of kernel arguments
        const int num_args = 4;
    
        // create array of function arguments
        oclDataObject ** args = (oclDataObject **) malloc (num_args * sizeof (oclDataObject *));
        args [0] = arg1;
        args [1] = new oclGPUDataObject <scalar_type> (&      arg2, 1);
        args [2] = arg3;
        args [3] = new oclGPUDataObject         <int> (& num_elems, 1);

        // create function object
        oclFunctionObject * op_obj = oclConnection :: Instance ()
                                        -> makeFunctionObject <elem_type, scalar_type>
                                              (kernel_name, args, num_args, oclConnection::KERNEL, oclConnection::SYNC);
        
        // execute function object
        ocl_run_func_obj (op_obj);

        // clear memory      
        delete op_obj;      
        delete args [1];
        delete args [3];
        free (args);
    
      }    
    
    
      /**
       * @brief                      execute specified kernel with 2 arguments
       */
      static inline
      const oclError &
      ocl_basic_operator_kernel_2              ( const          char * const kernel_name,
                                                       oclDataObject * const        arg1,
                                                       oclDataObject * const        arg2,
                                                                 int           num_elems )
      {
    
        // number of kernel arguments
        const int num_args = 3;
    
        // create array of function arguments
        oclDataObject ** args = (oclDataObject **) malloc (num_args * sizeof (oclDataObject *));
        args [0] = arg1;
        args [1] = arg2;
        args [2] = new oclGPUDataObject <int> (& num_elems, 1);

        // create function object
        oclFunctionObject * op_obj = oclConnection :: Instance ()
                                        -> makeFunctionObject <elem_type, scalar_type>
                                              (kernel_name, args, num_args, oclConnection::KERNEL, oclConnection::SYNC);
      
        // execute function object
        ocl_run_func_obj (op_obj);

        // clear memory      
        delete op_obj;      
        delete args [2];
        free (args);
    
      }
    
    
# ifdef __USE_VIENNA_CL__
      /**
       * @brief                       execute specified ViennaCl algorithm with 3 arguments
       */
      static inline
      const oclError &
      ocl_basic_operator_vclAlgo_3             ( const   vclAlgoType            vcl_algo,
                                                       oclDataObject * const        arg1,
                                                       oclDataObject * const        arg2,
                                                       oclDataObject * const      result,
                                                                 int           num_elems )
      {
    
        // number of kernel arguments
        const int num_args = 4;
    
        // create array of function arguments
        oclDataObject ** args = (oclDataObject **) malloc (num_args * sizeof (oclDataObject *));
        args [0] = arg1;
        args [1] = arg2;
        args [2] = result;
        args [3] = new oclGPUDataObject <int> (& num_elems, 1);

        // create function object
        oclFunctionObject * op_obj = oclConnection :: Instance ()
                                        -> makeFunctionObject <elem_type, scalar_type>
                                              (vcl_algo, args, num_args, oclConnection::VCL, oclConnection::SYNC);
      
        // execute function object
        ocl_run_func_obj (op_obj);

        // clear memory      
        delete op_obj;      
        delete args [3];
        free (args);
    
      }


      /**
       * @brief                       execute specified ViennaCl algorithm with 2 arguments + 2 scalars
       */
      static inline
      const oclError &
      ocl_basic_operator_vclAlgo_22            ( const   vclAlgoType            vcl_algo,
                                                       oclDataObject * const        arg1,
                                                       oclDataObject * const      result,
                                                                 int                  s1,
                                                                 int                  s2 )
      {
    
        // number of kernel arguments
        const int num_args = 2;
        const int num_scalars = 2;
    
        // create array of function arguments
        oclDataObject ** args = (oclDataObject **) malloc (num_args * sizeof (oclDataObject *));
        args [0] = arg1;
        args [1] = result;

        // create array of scalars
        int * scalars = (int *) malloc (num_scalars * sizeof (int));
        scalars [0] = s1;
        scalars [1] = s2;

        // create function object
        oclFunctionObject * op_obj = oclConnection :: Instance ()
                                       -> makeFunctionObject <elem_type, scalar_type>
                                           (vcl_algo, args, num_args, oclConnection::VCL, oclConnection::SYNC, num_scalars, scalars);

        // execute function object
        ocl_run_func_obj (op_obj);

        // clear memory      
        delete op_obj;
        free (scalars);
        free (args);
    
      }
    
      
      /**
       * @brief                       execute specified ViennaCl algorithm with 3 arguments + 5 scalars
       */
      static inline
      const oclError &
      ocl_basic_operator_vclAlgo_35            ( const   vclAlgoType            vcl_algo,
                                                       oclDataObject * const        arg1,
                                                       oclDataObject * const        arg2,
                                                       oclDataObject * const      result,
                                                                 int                  s1,
                                                                 int                  s2,
                                                                 int                  s3,
                                                                 int                  s4,
                                                                 int                  s5 )
      {
    
        // number of kernel arguments
        const int num_args = 3;
        const int num_scalars = 5;
    
        // create array of function arguments
        oclDataObject ** args = (oclDataObject **) malloc (num_args * sizeof (oclDataObject *));
        args [0] = arg1;
        args [1] = arg2;
        args [2] = result;

        // create array of scalars
        int * scalars = (int *) malloc (num_scalars * sizeof (int));
        scalars [0] = s1;
        scalars [1] = s2;
        scalars [2] = s3;
        scalars [3] = s4;
        scalars [4] = s5;

        // create function object
        oclFunctionObject * op_obj = oclConnection :: Instance ()
                                       -> makeFunctionObject <elem_type, scalar_type>
                                           (vcl_algo, args, num_args, oclConnection::VCL, oclConnection::SYNC, num_scalars, scalars);

        // execute function object
        ocl_run_func_obj (op_obj);

        // clear memory      
        delete op_obj;
        free (scalars);
        free (args);
    
      }
# endif
      
      
      /**
       * @brief                       execute specified AMD BLAS algorithm
       */
      static inline
      const oclError &
      ocl_basic_operator_amdblas_35            ( const oclAMDBlasType            amd_algo,
                                                       oclDataObject * const        arg1,
                                                       oclDataObject * const        arg2,
                                                       oclDataObject * const      result,
                                                                 int                  s1,
                                                                 int                  s2,
                                                                 int                  s3,
                                                                 int                  s4,
                                                                 int                  s5 )
      {
    
        // number of kernel arguments
        const int num_args = 3;
        const int num_scalars = 5;
    
        // create array of function arguments
        oclDataObject ** args = (oclDataObject **) malloc (num_args * sizeof (oclDataObject *));
        args [0] = arg1;
        args [1] = arg2;
        args [2] = result;

        // create array of scalars
        int * scalars = (int *) malloc (num_scalars * sizeof (int));
        scalars [0] = s1;
        scalars [1] = s2;
        scalars [2] = s3;
        scalars [3] = s4;
        scalars [4] = s5;

        // create function object
        oclFunctionObject * op_obj = oclConnection :: Instance ()
                                       -> makeFunctionObject <elem_type, scalar_type>
                                           (amd_algo, args, num_args, oclConnection::AMD, oclConnection::SYNC, num_scalars, scalars);

        // execute function object
        ocl_run_func_obj (op_obj);

        // clear memory      
        delete op_obj;
        free (scalars);
        free (args);
    
      }
      
      
      /**
       * @brief                       execute specified AMD BLAS algorithm
       */
      static inline
      const oclError &
      ocl_basic_operator_amdblas_34            ( const oclAMDBlasType            amd_algo,
                                                       oclDataObject * const        arg1,
                                                       oclDataObject * const        arg2,
                                                       oclDataObject * const      result,
                                                                 int                  s1,
                                                                 int                  s2,
                                                                 int                  s3,
                                                                 int                  s4 )
      {
    
        // number of kernel arguments
        const int num_args = 3;
        const int num_scalars = 4;
    
        // create array of function arguments
        oclDataObject ** args = (oclDataObject **) malloc (num_args * sizeof (oclDataObject *));
        args [0] = arg1;
        args [1] = arg2;
        args [2] = result;

        // create array of scalars
        int * scalars = (int *) malloc (num_scalars * sizeof (int));
        scalars [0] = s1;
        scalars [1] = s2;
        scalars [2] = s3;
        scalars [3] = s4;

        // create function object
        oclFunctionObject * op_obj = oclConnection :: Instance ()
                                       -> makeFunctionObject <elem_type, scalar_type>
                                           (amd_algo, args, num_args, oclConnection::AMD, oclConnection::SYNC, num_scalars, scalars);

        // execute function object
        ocl_run_func_obj (op_obj);

        // clear memory      
        delete op_obj;
        free (scalars);
        free (args);
    
      }
    
    
      //@}


    public:
    
    
      /**
       * @name                        memory management
       */
      //@{
    
      
      /**
       * @brief                       Create oclCPUDataObject (pinned).
       */
      static inline
      oclDataWrapper <elem_type> *
      make_CPU_Obj                    (      elem_type * const   cpu_arg,
                                       const    size_t &       num_elems)
      {
        
        
      	print_optional ("make_CPU_Obj <", trait1 :: print_elem_type (),
                                          trait2 :: print_elem_type (), "> (create new)", VERB_HIGH);
    
        return new oclCPUDataObject <elem_type> (cpu_arg, num_elems);
        
      }
      
    
      /**
       * @brief                       Create oclGPUDataObject.
       */
      static inline
      oclDataWrapper <elem_type> *
      make_GPU_Obj                    (      elem_type * const   cpu_arg,
                                       const    size_t &       num_elems)
      {

    	print_optional ("make_GPU_Obj <", trait1 :: print_elem_type (),
                                          trait2 :: print_elem_type (), "> (create new)", VERB_HIGH);
    
        return new oclGPUDataObject <elem_type> (cpu_arg, num_elems);

      }
    
    
      /**
       * @brief                      Create oclGPUDataObject
       *                             with same state as given oclDataWrapper.
       */
      static inline
      oclDataWrapper <elem_type> *
      make_GPU_Obj                   (                      elem_type             *  const   cpu_arg,
                                      const                    size_t             &        num_elems,
                                                       oclDataWrapper <elem_type> &              obj,
                                            oclDataObject :: CopyMode                      copy_mode = oclDataObject :: NO_BUFFER)
      {
      
        print_optional ("make_GPU_Obj <", trait1 :: print_elem_type (),
                                          trait2 :: print_elem_type (), "> (copy obj's state)", VERB_HIGH);
      
        // check if sizes fit (for !some! more control (or safety))
        if (num_elems != obj.getNumElems ())
        {
      
          return NULL;
      
        }
        else
        {
      
          // create oclDataObject to return
          oclDataWrapper <elem_type> * cp_obj = new oclGPUDataObject <elem_type> (cpu_arg, num_elems, obj, copy_mode);
        
          /* update GPU buffer with data if possible */
          if (copy_mode == oclDataObject :: COPY_BUFFER  &&  obj.bufferCopyable ())
          {
  
            ocl_basic_operator_kernel_2 ("copy_buffer", cp_obj, & obj, num_elems);
  
          }
        
          return cp_obj;
      
        }
    
      }
    
    
      //@}
      
      
      static inline
      void
      addKernelSource                 (const std::string & filename)
      {
        oclConnection :: Instance () -> rebuildWithSource <T, S> (filename);
      }


      static inline
      void
      addKernelSource                 (const std::string               & filename,
                                       const std::vector <std::string> & makros)
      {
        oclConnection :: Instance () -> rebuildWithSource <T, S> (filename, makros);
      }

      
      static inline
      void
      addKernelSources                (const std::vector <std::string> & filenames)
      {
        oclConnection :: Instance () -> rebuildWithSources <T, S> (filenames);
      }

      
      static inline
      void
      addKernelSources                (const std::vector <std::string> & filenames,
                                       const std::vector <std::string> & makros)
      {
        oclConnection :: Instance () -> rebuildWithSources <T, S> (filenames, makros);
      }

      
    
      /**
       * @name                        operators
       */
      //@{



      static inline
      std::vector <PerformanceInformation>
      ocl_operator_perf_dwt           ( oclDataObject * const arg1,
                                                  int            n,
                                                  int            m,
                                                  int            k,
                                        oclDataObject * const lpf,
                                        oclDataObject * const hpf,
                                        oclDataObject * const arg2,
                                        const int line_length,
                                        const int fl )
      {
        
          std::vector <PerformanceInformation> vec_perf;
        
          print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                             trait2 :: print_elem_type (), "> :: ocl_operator_perf_dwt", op_v_level);

          // kernel launch configuration
//          LaunchInformation lc (group_size_x, group_size_y, global_x, global_y);
          const LaunchInformation & lc = oclConnection :: Instance () -> getThreadConfig (std::string ("dwt2"));
          
          // dynamically allocate local memory
          const int num_loc_mem_elems = (n / (lc.global_x/lc.local_x) + fl) * (n / (lc.global_y/lc.local_y) + fl) + (n / (lc.global_x/lc.local_x)) * (n / (lc.global_y/lc.local_y) + fl);
          oclDataObject * loc_mem = new oclLocalMemObject <elem_type> (num_loc_mem_elems);
          
          ////////////////////
          // Global to Local
          ////////////////////
          std::string kernel_name ("perf_dwtGlobalToLocal");
          ProfilingInformation pi = ocl_basic_operator_kernel_21 (kernel_name, loc_mem, arg1, line_length, lc);
          double time_seconds = pi.time_end - pi.time_start;
          double effective_bw = ((double) ((line_length/(lc.global_x/(double)lc.local_x)+fl-2)/(double)lc.local_x * (line_length/(lc.global_y/(double)lc.local_y)+fl-2)/(double)lc.local_y) * lc.global_x * lc.global_y * sizeof (elem_type) ) * 1.0e-9f / time_seconds;
          vec_perf.push_back (PerformanceInformation (kernel_name, lc, std::string () + " Effective bandwidth (GB/s)", time_seconds, pi.time_mem_up, pi.time_mem_down, effective_bw));
                    
          ////////////////////
          // Local to Global
          ////////////////////
          kernel_name = std::string ("perf_dwtLocalToGlobal");
          pi = ocl_basic_operator_kernel_21 (kernel_name, loc_mem, arg2, line_length, lc);
          time_seconds = pi.time_end - pi.time_start;
          effective_bw = ((double) pow (line_length,2) * sizeof (elem_type) ) * 1.0e-9f / time_seconds;
          vec_perf.push_back (PerformanceInformation (kernel_name, lc, std::string () + " Effective bandwidth (GB/s)", time_seconds, pi.time_mem_up, pi.time_mem_down, effective_bw));
          
          ////////////////////
          // Convolution
          ////////////////////
          kernel_name = std::string ("perf_dwtFilter");
          pi = ocl_basic_operator_kernel_31 (kernel_name, loc_mem, lpf, hpf, line_length, lc);
          time_seconds = pi.time_end - pi.time_start;
          float effective_flops = ((float) (7 + (line_length/(lc.global_x/lc.local_x)+fl) / lc.local_x * line_length/lc.global_x * (9 + 4 * fl + 2)) + (7 + pow(line_length/lc.global_x,2)) * (9 + 4*fl + 2)) * lc.global_x * lc.global_y / time_seconds * 1.0e-9f;
          vec_perf.push_back (PerformanceInformation (kernel_name, lc, std::string () + " Effective Flop/s (GFlop/s)", time_seconds, pi.time_mem_up, pi.time_mem_down, effective_flops));
                    
          delete loc_mem;
          
          return vec_perf;
        
      }


      /**
       * @brief                       3D Discrete Wavelet Transform.
       *
       * @param  arg1                 Address of signal (n x m x k).
       * @param  arg2                 Address of resulting DWT (n x m x k).
       * @param  n                    First dimension.
       * @param  m                    Second dimension.
       * @param  k                    Third dimension.
       * @param  filter               Convolution kernel.
       * @param  fl                   Length of convolution kernel.
       * ...
       */
      static inline
      std::vector <PerformanceInformation>
      ocl_operator_dwt2                ( oclDataObject * const arg1,
                                                  int            n,
                                                  int            m,
                                                  int            k,
                                        oclDataObject * const  lpf,
                                        oclDataObject * const  hpf,
                                                  int           fl,
                                            const int         levels,
                                        oclDataObject * const arg2)
      {
        
          print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                             trait2 :: print_elem_type (), "> :: ocl_operator_dwt2", op_v_level);

          LaunchInformation lc = oclConnection :: Instance () -> getThreadConfig (std::string ("dwt2"));
          
          // dynamically allocate local memory
//          const int num_loc_mem_elems = (2*lc.local_x + fl) * (2*lc.local_y + fl) * 2;
          const int num_loc_mem_elems = (n / (lc.global_x/lc.local_x) + fl) * (n / (lc.global_y/lc.local_y) + fl) + (n / (lc.global_x/lc.local_x)) * (n / (lc.global_y/lc.local_y) + fl);
          oclDataObject * loc_mem = new oclLocalMemObject <elem_type> (num_loc_mem_elems);
          
          std::cout << " loc_mem (dwt2): " << num_loc_mem_elems * sizeof (elem_type) << " Bytes " << std::endl;
          
          std::vector <ProfilingInformation> vec_pi;
          std::vector <ProfilingInformation> vec_pi2;
          
          // launch kernels
          oclDataObject * tmp1 = arg1;
          oclDataObject * tmp2 = arg2;
          for (int i = 0; i < levels; i++)
          {
            const int line_length = n / pow (2, i);
            ProfilingInformation pi = ocl_basic_operator_kernel_56 ("dwt2", tmp1, lpf, hpf, tmp2, loc_mem, n, m, k, line_length, 1, num_loc_mem_elems, lc);
            
            vec_pi.push_back (pi);
            
            oclDataObject * tmp = tmp1;
            tmp1 = tmp2;
            tmp2 = tmp;
          }
          
          ProfilingInformation tmp_pi = ocl_basic_operator_kernel_25 ("dwt2_final", arg1, arg2, n, m, k, n / pow (2, levels-1), levels, lc);
          
          // load data back to CPU
          tmp_pi.time_mem_down += arg2 -> getData ();
          
          delete loc_mem;
          
          ///////////////
          // performance
          ///////////////

          std::vector <PerformanceInformation> vec_perf;
          
# ifdef __PERFORMANCE_INFO__
          
          const int num_groups_0 = lc.global_x / lc.local_x;
          const int num_groups_1 = lc.global_y / lc.local_y;
          
          // data amount for dwt2 over all levels
          int data_size_1 = 0;
          for (int i = 0; i < levels; i++)
          {
            const int sl_0 = n / pow (2, i);
            const int sl_1 = m / pow (2, i);
            const int sl_2 = k / pow (2, i);
            const float block_size_0 = sl_0 / num_groups_0;
            const float block_size_1 = sl_1 / num_groups_1;
            const int offset = fl - 2;
            // global -> local
            data_size_1 += (block_size_0 + offset) * (block_size_1 + offset) * num_groups_0 * num_groups_1 * sl_2;
            // local -> global
            data_size_1 += sl_0 * sl_1 * sl_2;
          }
                   
          // dwt2
          float time_seconds_1 = 0,
                time_mem_up_1 = 0,
                time_mem_down_1 = 0;
          for (std::vector <ProfilingInformation> :: const_iterator it_pi = vec_pi.begin (); it_pi != vec_pi.end (); ++it_pi)
          {
            time_seconds_1 += it_pi -> time_end - it_pi -> time_start;
            time_mem_up_1   += it_pi -> time_mem_up;
            time_mem_down_1 += it_pi -> time_mem_down;
          }
          float effective_bw_1 = (data_size_1 * sizeof (elem_type) * 1.0e-9f) / time_seconds_1;

          // data amount for dwt2_final
          int data_size_2 = 0;
          for (int i = 1; i <= (levels-1)/2; i++)
            data_size_2 += pow (n/pow (2,2*i), 2);
          data_size_2 *= 3;
          if (levels&1)
            data_size_2 += pow (n/pow (2,2*(levels-1)), 2);
          
          float time_seconds_2 = tmp_pi.time_end - tmp_pi.time_start;
          float effective_bw_2 = (data_size_2 * sizeof (elem_type) * 1.0e-9f) / time_seconds_2;

          float effective_bw = ((data_size_1 + data_size_2) * sizeof (elem_type) * 1.0e-9f) / (time_seconds_1 + time_seconds_2);
          vec_perf.push_back (PerformanceInformation ("dwt2 (all levels, final)", lc, " Effective bandwidth (GB/s)", time_seconds_1 + time_seconds_2, time_mem_up_1, time_mem_down_1, effective_bw));
          vec_perf.push_back (PerformanceInformation ("dwt2 (all levels)", lc, " Effective bandwidth (GB/s)", time_seconds_1, time_mem_up_1, time_mem_down_1, effective_bw_1));
          vec_perf.push_back (PerformanceInformation ("dwt2_final", lc, " Effective bandwidth (GB/s)", time_seconds_2, tmp_pi.time_mem_up, tmp_pi.time_mem_down, effective_bw_2));
# endif
          
          return vec_perf;
          
      }


      static inline
      std::vector <PerformanceInformation>
      ocl_operator_dwt3                ( oclDataObject * const arg1,
                                                  int            n,
                                                  int            m,
                                                  int            k,
                                        oclDataObject * const  lpf,
                                        oclDataObject * const  hpf,
                                                  int           fl,
                                            const int         levels,
                                        oclDataObject * const arg2,
                                        oclDataObject * const tmp,
                                            const int         chunk_size)
      {
        
          print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                             trait2 :: print_elem_type (), "> :: ocl_operator_dwt3", op_v_level);

          LaunchInformation lc = oclConnection :: Instance () -> getThreadConfig (std::string ("dwt2"));
          LaunchInformation lc2 = oclConnection :: Instance () -> getThreadConfig (std::string ("dwt3"));
          
          ////////////////
          // dynamically allocate local memory
          ////////////////
          
          // dwt2
          const int num_loc_mem_elems = ((n / (lc.global_x/lc.local_x) + fl) * (n / (lc.global_y/lc.local_y) + fl) + (n / (lc.global_x/lc.local_x)) * (n / (lc.global_y/lc.local_y) + fl)) * lc.local_z;
          oclDataObject * loc_mem = new oclLocalMemObject <elem_type> (num_loc_mem_elems);
          
          // dwt3
          const int num_loc_mem_elems2 = (2 * k + 8) * lc2.local_x * lc2.local_y;
          oclDataObject * loc_mem2 = new oclLocalMemObject <elem_type> (num_loc_mem_elems2);
          
          std::cout << " loc_mem (dwt2): " << num_loc_mem_elems * sizeof (elem_type) << " Bytes " << std::endl;
          std::cout << " loc_mem2 (dwt3): " << num_loc_mem_elems2 * sizeof (elem_type) << " Bytes " << std::endl;
          std::cout << " chunk_size: " << chunk_size << std::endl;
          
          //////////
          // collect profiling information for each level
          //////////
          std::vector <ProfilingInformation> vec_pi;
          std::vector <ProfilingInformation> vec_pi2;
          
          //////////
          // prepare access patterns for CPU-GPU-transfers
          //////////
          arg1 -> APDevice () = oclAccessPattern (0, 0, 0,
                  n * sizeof (elem_type), n * m * sizeof (elem_type),
                  n * sizeof (elem_type), m, k);
          arg1 -> APHost () = arg1 -> APDevice ();
          arg2 -> APDevice () = arg2 -> APHost () = arg1 -> APHost ();
          tmp -> APDevice () = tmp -> APHost () = arg1 -> APHost ();
          
          ////////////
          // temporaries for buffer objects (avoid writing to src memory)
          ////////////
          oclDataObject * tmp1 = arg1;
          oclDataObject * tmp2 = arg2;
          
          // launch kernels
          for (int i = 0; i < levels; i++)
          {

            const int line_length = n / pow (2, i);
            const int num_slices = line_length;
            const int chunk_size_dwt2 = min (num_slices, chunk_size);
            
            // run kernel "dwt2" on slices
            ProfilingInformation pi = {0, 0, 0, 0};
            tmp1 -> APHost ().Region (0) = tmp1 -> APDevice ().Region (0) = line_length * sizeof (elem_type);
            tmp2 -> APHost ().Region (0) = tmp2 -> APDevice ().Region (0) = line_length * sizeof (elem_type);
            tmp1 -> APHost ().Region (1) = tmp1 -> APDevice ().Region (1) = line_length;
            tmp2 -> APHost ().Region (1) = tmp2 -> APDevice ().Region (1) = line_length;
            tmp1 -> APHost ().Region (2) = tmp1 -> APDevice ().Region (2) = chunk_size_dwt2;
            tmp2 -> APHost ().Region (2) = tmp2 -> APDevice ().Region (2) = chunk_size_dwt2;
            std::cout << "\n ************ \n DWT2: level " << i << std::endl;
            for (int l = 0; l < num_slices; l += chunk_size_dwt2)
            {
              tmp1 -> APHost ().Origin (2) = tmp2 -> APHost ().Origin (2) = l;
              tmp2 -> setSync(); // no need to upload tmp2 here
              pi += ocl_basic_operator_kernel_56 ("dwt2", tmp1, lpf, hpf, tmp2, loc_mem, n, m, k, line_length, chunk_size_dwt2, num_loc_mem_elems, lc);
              pi.time_mem_down += tmp2 -> getData ();
            }  
            vec_pi.push_back (pi);
            tmp1 -> APHost ().Origin (2) = tmp2 -> APHost ().Origin (2) = 0; // reset
            tmp1 -> APDevice ().Region (2) = tmp1 -> APHost ().Region (2) = line_length; // reset
            tmp2 -> APDevice ().Region (2) = tmp2 -> APHost ().Region (2) = line_length; // reset
            
            // switch to temporary memory for writing (not to src image)
            if (i == 0)
              tmp1 = tmp;
            
            // run kernel "dwt3" on beams
            ProfilingInformation pi2 = {0, 0, 0, 0};
            const int chunk_size_dwt3 = min (line_length, chunk_size);
            const int chunk_size_dwt3_dim0 = line_length;
            lc2.global_z = lc2.local_z = lc2.local_z > line_length ? line_length : lc2.local_z;
            const oclAccessPattern tmp_ap_array [4] = {tmp1 -> APHost (), tmp1 -> APDevice (),
                                                       tmp2 -> APHost (), tmp2 -> APDevice ()}; // save current state !!!
            tmp1 -> APHost ().Region (0) = tmp1 -> APDevice ().Region (0) = chunk_size_dwt3_dim0 * sizeof (elem_type);
            tmp2 -> APHost ().Region (0) = tmp2 -> APDevice ().Region (0) = chunk_size_dwt3_dim0 * sizeof (elem_type);
            tmp1 -> APHost ().Region (1) = tmp1 -> APDevice ().Region (1) = chunk_size_dwt3;
            tmp2 -> APHost ().Region (1) = tmp2 -> APDevice ().Region (1) = chunk_size_dwt3;
            tmp1 -> APHost ().Region (2) = tmp1 -> APDevice ().Region (2) = line_length;
            tmp2 -> APHost ().Region (2) = tmp2 -> APDevice ().Region (2) = line_length;
            tmp1 -> APDevice ().RowPitch () = tmp2 -> APDevice ().RowPitch () = chunk_size_dwt3_dim0 * sizeof (elem_type);
            tmp1 -> APDevice ().SlicePitch () = tmp2 -> APDevice ().SlicePitch () = chunk_size_dwt3_dim0 * chunk_size_dwt3 * sizeof (elem_type);
            std::cout << "\n ********** \n DWT3: level " << i << std::endl;
            for (int l = 0; l < line_length; l += chunk_size_dwt3_dim0)
            {
              tmp1 -> APHost ().Origin (0) = tmp2 -> APHost ().Origin (0) = l * sizeof (elem_type);
              for (int ll = 0; ll < line_length; ll += chunk_size_dwt3)
              {
                tmp1 -> APHost ().Origin (1) = tmp2 -> APHost ().Origin (1) = ll;
                tmp1 -> setSync (); // no need to upload tmp1 here
                pi2 += ocl_basic_operator_kernel_57 ("dwt3", tmp2, lpf, hpf, tmp1, loc_mem2, n, m, k, line_length, chunk_size_dwt3_dim0, chunk_size_dwt3, num_loc_mem_elems2, lc2);
                pi2.time_mem_down += tmp1 -> getData ();
              }
            }
            vec_pi2.push_back (pi2);
            tmp1 -> APHost () = tmp_ap_array [0]; tmp1 -> APDevice () = tmp_ap_array [1]; // reset
            tmp2 -> APHost () = tmp_ap_array [2]; tmp2 -> APDevice () = tmp_ap_array [3]; // reset
                
          }
          
          delete loc_mem;
          delete loc_mem2;
          
          ///////////////
          // performance
          ///////////////

          std::vector <PerformanceInformation> vec_perf;
          
# ifdef __PERFORMANCE_INFO__
          
          const int num_groups_0 = lc.global_x / lc.local_x;
          const int num_groups_1 = lc.global_y / lc.local_y;
          
          // data amount for dwt2 over all levels
          int data_size_1 = 0;
          int data_size_3 = 0;
          for (int i = 0; i < levels; i++)
          {
            const int sl_0 = n / pow (2, i);
            const int sl_1 = m / pow (2, i);
            const int sl_2 = k / pow (2, i);
            const float block_size_0 = sl_0 / num_groups_0;
            const float block_size_1 = sl_1 / num_groups_1;
            const int offset = fl - 2;
            // global -> local
            data_size_1 += (block_size_0 + offset) * (block_size_1 + offset) * num_groups_0 * num_groups_1 * sl_2;
            data_size_3 += (sl_2 + offset) * sl_0 * sl_1;
            // local -> global
            data_size_1 += sl_0 * sl_1 * sl_2;
            data_size_3 += sl_2 * sl_0 * sl_1;
          }
         
          // dwt2
          float time_seconds_1 = 0,
                time_mem_up_1 = 0,
                time_mem_down_1 = 0;
          for (std::vector <ProfilingInformation> :: const_iterator it_pi = vec_pi.begin (); it_pi != vec_pi.end (); ++it_pi)
          {
            time_seconds_1 += it_pi -> time_end - it_pi -> time_start;
            time_mem_up_1   += it_pi -> time_mem_up;
            time_mem_down_1 += it_pi -> time_mem_down;
          }
          float effective_bw_1 = (data_size_1 * sizeof (elem_type) * 1.0e-9f) / time_seconds_1;

          // dwt3
          float time_seconds_3 = 0,
                time_mem_up_3 = 0,
                time_mem_down_3 = 0;
          for (std::vector <ProfilingInformation> :: const_iterator it_pi = vec_pi2.begin (); it_pi != vec_pi2.end (); ++it_pi)
          {
            time_seconds_3 += it_pi -> time_end - it_pi -> time_start;
            time_mem_up_3   += it_pi -> time_mem_up;
            time_mem_down_3 += it_pi -> time_mem_down;
          }
          float effective_bw_3 = (data_size_3 * sizeof (elem_type) * 1.0e-9f) / time_seconds_3;
          

          float effective_bw = ((data_size_1 + data_size_3) * sizeof (elem_type) * 1.0e-9f) / (time_seconds_1 + time_seconds_3);
          vec_perf.push_back (PerformanceInformation ("dwt2+3 (all levels)", lc, " Effective bandwidth (GB/s)", time_seconds_1 + time_seconds_3, time_mem_up_1 + time_mem_up_3, time_mem_down_1 + time_mem_down_3, effective_bw));
          vec_perf.push_back (PerformanceInformation ("dwt2 (all levels)", lc, " Effective bandwidth (GB/s)", time_seconds_1, time_mem_up_1, time_mem_down_1, effective_bw_1));
          vec_perf.push_back (PerformanceInformation ("dwt3 (all levels)", lc2, " Effective bandwidth (GB/s)", time_seconds_3, time_mem_up_3, time_mem_down_3, effective_bw_3));

# endif
          
          return vec_perf;
          
      }
      
      static
      int roundUp(const int numToRound, const int multiple) 
      { 
       if(multiple == 0) 
       { 
        return numToRound; 
       } 

       int remainder = numToRound % multiple;
       if (remainder == 0)
        return numToRound;
       return numToRound + multiple - remainder;
      } 
      
      
      static inline
      std::vector <PerformanceInformation>
      ocl_operator_dwt3_alt           ( oclDataObject * const arg1,
                                                  int            n,
                                                  int            m,
                                                  int            k,
                                        oclDataObject * const  lpf,
                                        oclDataObject * const  hpf,
                                                  int           fl,
                                                  int       levels,
                                        oclDataObject * const arg2,
                                        oclDataObject * const tmp,
                                            const int         chunk_size)
      {
        
          print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                             trait2 :: print_elem_type (), "> :: ocl_operator_dwt3_alt", op_v_level);

          LaunchInformation lc1 = oclConnection :: Instance () -> getThreadConfig (std::string ("dwt_1_alt"));
          LaunchInformation lc2 = oclConnection :: Instance () -> getThreadConfig (std::string ("dwt_2_alt"));
          LaunchInformation lc3 = oclConnection :: Instance () -> getThreadConfig (std::string ("dwt_3_alt"));
          
          ////////////////
          // dynamically allocate local memory
          ////////////////
                    
          const int loc_mem_line_in = 64 + fl-2 + 1;
          const int loc_mem_line_out = 64 + 1;
          
          // dwt_1
          const int num_loc_mem_elems1 = (loc_mem_line_in + loc_mem_line_out) * max (lc1.local_x, lc1.local_y) * lc1.local_z;
          oclDataObject * loc_mem1 = new oclLocalMemObject <elem_type> (num_loc_mem_elems1);
          
          // dwt_2
          const int num_loc_mem_elems2 = (loc_mem_line_in + loc_mem_line_out) * lc2.local_x * lc2.local_z;
          oclDataObject * loc_mem2 = new oclLocalMemObject <elem_type> (num_loc_mem_elems2);
          
          // dwt_3
          const int num_loc_mem_elems3 = (loc_mem_line_in + loc_mem_line_out) * lc3.local_x * lc3.local_y;
          oclDataObject * loc_mem3 = new oclLocalMemObject <elem_type> (num_loc_mem_elems3);
          
          std::cout << " loc_mem1 (dwt_1_alt): " << num_loc_mem_elems1 * sizeof (elem_type) << " Bytes " << std::endl;
          std::cout << " loc_mem2 (dwt_2_alt): " << num_loc_mem_elems2 * sizeof (elem_type) << " Bytes " << std::endl;
          std::cout << " loc_mem3 (dwt_3_alt): " << num_loc_mem_elems3 * sizeof (elem_type) << " Bytes " << std::endl;
          std::cout << " chunk_size: " << chunk_size << std::endl;
          
          //////////
          // collect profiling information for each level
          //////////
          std::vector <ProfilingInformation> vec_pi;
          std::vector <ProfilingInformation> vec_pi2;
          std::vector <ProfilingInformation> vec_pi3;
          
          ///////////
          // allocate kernel arguments
          ///////////
          oclDataObject * gpu_n = new oclGPUDataObject <int> (&n, 1);
          oclDataObject * gpu_m = new oclGPUDataObject <int> (&m, 1);
          oclDataObject * gpu_k = new oclGPUDataObject <int> (&k, 1);
          int line_length = 0;
          oclDataObject * gpu_line_length = new oclGPUDataObject <int> (&line_length, 1);
          
          //////////
          // prepare access patterns for CPU-GPU-transfers
          //////////
          const int roundTo = 1;
          arg1 -> APDevice () = oclAccessPattern (0, 0, 0,
                  n * sizeof (elem_type), n * m * sizeof (elem_type),
                   n * sizeof (elem_type), m, k);
          arg1 -> APHost () = oclAccessPattern (0, 0, 0,
                   n * sizeof (elem_type), n * m * sizeof (elem_type),
                   n * sizeof (elem_type), m, k);
          arg2 -> APDevice () = arg1 -> APDevice ();
          tmp -> APDevice () = arg2 -> APHost () = tmp -> APHost () = arg1 -> APHost ();
          
          ////////////
          // temporaries for buffer objects (avoid writing to src memory)
          ////////////
          oclDataObject * tmp1 = arg1;
          oclDataObject * tmp2 = arg2;
                    
          ProfilingInformation pi_final = {0, 0, 0, 0};
          
          // loop over levels
          for (int i = 0; i < levels; i++)
          {
            
            line_length = n / pow (2, i);
            gpu_line_length -> setCPUModified ();
        
            // run kernels "dwt_2" and "dwt_1" on slices
            ProfilingInformation pi = {0, 0, 0, 0};
            ProfilingInformation pi2 = {0, 0, 0, 0};
            
            if (i == 0)
              tmp1 -> setCPUModified (); // upload src
            else
              tmp1 -> setSync ();
            tmp2 -> setSync (); // do NOT upload dest
            pi += ocl_basic_operator_kernel_10 ("dwt_1_alt", tmp1, lpf, hpf, tmp2, loc_mem1, gpu_n, gpu_m, gpu_k, gpu_line_length, gpu_line_length, lc1);
                
            // do not write to src image
            if (i == 0)
              tmp1 = tmp;
              
            tmp1 -> setSync (); // use results on GPU
            tmp2 -> setSync (); // use results on GPU
            pi2 += ocl_basic_operator_kernel_10 ("dwt_2_alt", tmp2, lpf, hpf, tmp1, loc_mem2, gpu_n, gpu_m, gpu_k, gpu_line_length, gpu_line_length, lc2);
            vec_pi.push_back (pi);
            vec_pi2.push_back (pi2);
                        
            // run kernel "dwt3" on beams
            ProfilingInformation pi3 = {0, 0, 0, 0};

            tmp1 -> setSync ();
            tmp2 -> setSync (); // do NOT upload dest
            pi3 += ocl_basic_operator_kernel_11 ("dwt_3_alt", tmp1, lpf, hpf, tmp2, loc_mem3, gpu_n, gpu_m, gpu_k, gpu_line_length, gpu_line_length, gpu_line_length, lc3);
                
//            if (i != 0)
//            {
//              pi_final += ocl_basic_operator_kernel_56 ("dwt_final_alt", tmp2, lpf, hpf, tmp1, loc_mem1, n, m, k, line_length, line_length, num_loc_mem_elems1, lc1);
//            }
//            else
//            {
              oclDataObject * cpy = tmp1;
              tmp1 = tmp2;
              tmp2 = cpy;
//            }
            vec_pi3.push_back (pi3);
            
          }
          
          oclDataObject * gpu_levels = new oclGPUDataObject <int> (&levels, 1);
          line_length = n/pow(2,levels-1);
          gpu_line_length -> setCPUModified ();
          
          pi_final += ocl_basic_operator_kernel_10 ("dwt_final_alt", tmp, lpf, hpf, arg2, loc_mem1, gpu_n, gpu_m, gpu_k, gpu_line_length, gpu_levels, lc1);
          
          pi_final.time_mem_down += arg2 -> getData ();
          
          delete loc_mem1;
          delete loc_mem2;
          delete loc_mem3;
          
          delete gpu_n;
          delete gpu_m;
          delete gpu_k;
          delete gpu_line_length;
          delete gpu_levels;
          
          ///////////////
          // performance
          ///////////////
          
# ifdef __PERFORMANCE_INFO__
          return dwt_performance (vec_pi, vec_pi2, vec_pi3, pi_final, lc1, lc2, lc3, levels, n, m, k, fl);
# else
          return std::vector <PerformanceInformation> (); 
# endif
          
      }
      
      
      static inline
      std::vector <PerformanceInformation>
      ocl_operator_dwt3_alt_cs        ( oclDataObject * const arg1,
                                                  int            n,
                                                  int            m,
                                                  int            k,
                                        oclDataObject * const  lpf,
                                        oclDataObject * const  hpf,
                                                  int           fl,
                                            const int         levels,
                                        oclDataObject * const arg2,
                                        oclDataObject * const tmp,
                                            const int         chunk_size)
      {
        
          print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                             trait2 :: print_elem_type (), "> :: ocl_operator_dwt3_alt_cs", op_v_level);

          LaunchInformation lc1 = oclConnection :: Instance () -> getThreadConfig (std::string ("dwt_1_alt"));
          LaunchInformation lc2 = oclConnection :: Instance () -> getThreadConfig (std::string ("dwt_2_alt"));
          LaunchInformation lc3 = oclConnection :: Instance () -> getThreadConfig (std::string ("dwt_3_alt"));
          
          ////////////////
          // dynamically allocate local memory
          ////////////////
                    
          const int loc_mem_line_in = 64 + fl-2 + 1;
          const int loc_mem_line_out = 64 + 1;
          
          // dwt_1
          const int num_loc_mem_elems1 = (loc_mem_line_in + loc_mem_line_out) * max (lc1.local_x, lc1.local_y) * lc1.local_z;
          oclDataObject * loc_mem1 = new oclLocalMemObject <elem_type> (num_loc_mem_elems1);
          
          // dwt_2
          const int num_loc_mem_elems2 = (loc_mem_line_in + loc_mem_line_out) * lc2.local_x * lc2.local_z;
          oclDataObject * loc_mem2 = new oclLocalMemObject <elem_type> (num_loc_mem_elems2);
          
          // dwt_3
          const int num_loc_mem_elems3 = (loc_mem_line_in + loc_mem_line_out) * lc3.local_x * lc3.local_y;
          oclDataObject * loc_mem3 = new oclLocalMemObject <elem_type> (num_loc_mem_elems3);
          
          std::cout << " loc_mem1 (dwt_1_alt): " << num_loc_mem_elems1 * sizeof (elem_type) << " Bytes " << std::endl;
          std::cout << " loc_mem2 (dwt_2_alt): " << num_loc_mem_elems2 * sizeof (elem_type) << " Bytes " << std::endl;
          std::cout << " loc_mem3 (dwt_3_alt): " << num_loc_mem_elems3 * sizeof (elem_type) << " Bytes " << std::endl;
          std::cout << " chunk_size: " << chunk_size << std::endl;
          
          //////////
          // collect profiling information for each level
          //////////
          std::vector <ProfilingInformation> vec_pi;
          std::vector <ProfilingInformation> vec_pi2;
          std::vector <ProfilingInformation> vec_pi3;
          
          //////////
          // kernel arguments
          //////////
          int line_length = 0;
          oclDataObject * gpu_line_length = new oclGPUDataObject <int> (&line_length, 1);
          int chunk_size_dwt12 = 0;
          oclDataObject * gpu_chunk_size_dwt12 = new oclGPUDataObject <int> (&chunk_size_dwt12, 1);
          int chunk_size_dwt3 = 0;
          oclDataObject * gpu_chunk_size_dwt3 = new oclGPUDataObject <int> (&chunk_size_dwt3, 1);
          int chunk_size_dwt3_dim0 = 0;
          oclDataObject * gpu_chunk_size_dwt3_dim0 = new oclGPUDataObject <int> (&chunk_size_dwt3_dim0, 1);
          
          //////////
          // prepare access patterns for CPU-GPU-transfers
          //////////
          const int padding = 0;//fl - 2;
          const int roundTo = 1;
          arg1 -> APDevice () = oclAccessPattern (padding * sizeof (elem_type), 0, 0,
                  roundUp (n + padding, roundTo) * sizeof (elem_type), roundUp (n + padding, roundTo) * m * sizeof (elem_type),
                   n * sizeof (elem_type), m, k);
          arg1 -> APHost () = oclAccessPattern (0, 0, 0,
                   n * sizeof (elem_type), n * m * sizeof (elem_type),
                   n * sizeof (elem_type), m, k);
          arg2 -> APDevice () = arg1 -> APDevice ();
          tmp -> APDevice () = arg2 -> APHost () = tmp -> APHost () = arg1 -> APHost ();
          
          ////////////
          // temporaries for buffer objects (avoid writing to src memory)
          ////////////
          oclDataObject * tmp1 = arg1;
          oclDataObject * tmp2 = arg2;
                    
          ProfilingInformation pi_final = {0, 0, 0, 0};
          
          // loop over levels
          for (int i = 0; i < levels; i++)
          {
            
            line_length = n / pow (2, i);
            const int pad_line_length = roundUp (line_length + padding, roundTo);
            const int num_slices = line_length;
            chunk_size_dwt12 = min (num_slices, chunk_size);
            const int ld_12 = (chunk_size < line_length ? line_length : n);
                    
            gpu_line_length -> setCPUModified ();
            gpu_chunk_size_dwt12 -> setCPUModified ();
                    
            // run kernels "dwt_2" and "dwt_1" on slices
            ProfilingInformation pi = {0, 0, 0, 0};
            ProfilingInformation pi2 = {0, 0, 0, 0};
            
            // set memory region to current chunk_size
            tmp1 -> APHost ().Region (2) = tmp1 -> APDevice ().Region (2) = chunk_size_dwt12;
            tmp2 -> APHost ().Region (2) = tmp2 -> APDevice ().Region (2) = chunk_size_dwt12;
            tmp -> APHost ().Region (2) = tmp -> APDevice ().Region (2) = chunk_size_dwt12;
            
            // loop over chunks of slices
            for (int l = 0; l < num_slices; l += chunk_size_dwt12)
            {
                            
              // set origin of current chunk on host
              tmp1 -> APHost ().Origin (2) = tmp2 -> APHost ().Origin (2)
                                           = tmp  -> APHost ().Origin (2) = l;
              
              // set padding for tmp1 (READ)
              tmp1 -> APDevice ().Origin (0) = padding * sizeof (elem_type);
              tmp1 -> APDevice ().RowPitch () = pad_line_length * sizeof (elem_type);
              tmp1 -> APDevice ().SlicePitch () = pad_line_length * line_length * sizeof(elem_type);
              
              tmp1 -> setCPUModified (); // upload src
              tmp2 -> setSync (); // do NOT upload dest
              
              pi += ocl_basic_operator_kernel_10 ("dwt_1_alt", tmp1, lpf, hpf, tmp2, loc_mem1, gpu_line_length, gpu_line_length, gpu_chunk_size_dwt12, gpu_line_length, gpu_chunk_size_dwt12, lc1);
              
              // do not write to src image
              if (i == 0)
                tmp1 = tmp;
              
              // set padding for tmp1 (WRITE)
              tmp1 -> APDevice ().Origin (0) = 0;
              tmp1 -> APDevice ().RowPitch () = (line_length) * sizeof (elem_type);
              tmp1 -> APDevice ().SlicePitch () = (line_length) * line_length * sizeof(elem_type);
              
              tmp1 -> setSync (); // use results on GPU
              tmp2 -> setSync (); // use results on GPU
              
              pi2 += ocl_basic_operator_kernel_10 ("dwt_2_alt", tmp2, lpf, hpf, tmp1, loc_mem2, gpu_line_length, gpu_line_length, gpu_chunk_size_dwt12, gpu_line_length, gpu_chunk_size_dwt12, lc2);

              // download results
              pi2.time_mem_down += tmp1 -> getData ();
              
              // do not write to src image
              if (i == 0)
                tmp1 = arg1;
            }
            
            vec_pi.push_back (pi);
            vec_pi2.push_back (pi2);
            
            tmp1 -> APHost ().Origin (2) = tmp2 -> APHost ().Origin (2) = tmp -> APHost ().Origin (2) = 0; // reset
            tmp1 -> APDevice ().Region (2) = tmp1 -> APHost ().Region (2) = line_length; // reset
            tmp2 -> APDevice ().Region (2) = tmp2 -> APHost ().Region (2) = line_length; // reset
            tmp -> APDevice ().Region (2) = tmp -> APHost ().Region (2) = line_length; // reset
            
            // switch to temporary memory for writing (not to src image)
            if (i == 0)
              tmp1 = tmp;
            
            // run kernel "dwt3" on beams
            ProfilingInformation pi3 = {0, 0, 0, 0};
            chunk_size_dwt3 = min (line_length, chunk_size);
            chunk_size_dwt3_dim0 = line_length;
            const int pad_chunk_size_dwt3 = roundUp (chunk_size_dwt3 + padding, roundTo);
            const int pad_chunk_size_dwt3_dim0 = roundUp (chunk_size_dwt3_dim0 + padding, roundTo);
            const int ld_3 = (chunk_size < line_length ? chunk_size_dwt3 : n);
            const int ld_3_dim0 = (chunk_size < line_length ? chunk_size_dwt3_dim0 : n);
            
            gpu_chunk_size_dwt3 -> setCPUModified ();
            gpu_chunk_size_dwt3_dim0 -> setCPUModified ();
            
            const oclAccessPattern tmp_ap_array [4] = {tmp1 -> APHost (), tmp1 -> APDevice (),
                                                       tmp2 -> APHost (), tmp2 -> APDevice ()}; // save current state !!!
            
            // set memory region: shrink dim0 & dim1
            tmp1 -> APHost ().Region (0) = tmp1 -> APDevice ().Region (0) = chunk_size_dwt3_dim0 * sizeof (elem_type);
            tmp2 -> APHost ().Region (0) = tmp2 -> APDevice ().Region (0) = chunk_size_dwt3_dim0 * sizeof (elem_type);
            tmp1 -> APHost ().Region (1) = tmp1 -> APDevice ().Region (1) = chunk_size_dwt3;
            tmp2 -> APHost ().Region (1) = tmp2 -> APDevice ().Region (1) = chunk_size_dwt3;
            
            // sizes of rows and slices change accordingly
            // tmp1: input -> padding
            // tmp2: output -> without padding
            tmp1 -> APDevice ().Origin (0) = padding * sizeof (elem_type);
            tmp1 -> APDevice ().RowPitch () = pad_chunk_size_dwt3_dim0 * sizeof (elem_type);
            tmp1 -> APDevice ().SlicePitch () = pad_chunk_size_dwt3_dim0 * chunk_size_dwt3 * sizeof (elem_type);
            tmp2 -> APDevice ().Origin (0) = 0;
            tmp2 -> APDevice ().RowPitch () = chunk_size_dwt3_dim0 * sizeof (elem_type);
            tmp2 -> APDevice ().SlicePitch () = chunk_size_dwt3_dim0 * chunk_size_dwt3 * sizeof (elem_type);

            // loop over chunks of beams along dim0
            for (int l = 0; l < line_length; l += chunk_size_dwt3_dim0)
            {
              
              // adjust origin in host memory for dim0
              tmp1 -> APHost ().Origin (0) = tmp2 -> APHost ().Origin (0) = l * sizeof (elem_type);
              
              // loop over chunks of beams along dim1
              for (int ll = 0; ll < line_length; ll += chunk_size_dwt3)
              {
                
                // adjust origin in host memory for dim1
                tmp1 -> APHost ().Origin (1) = tmp2 -> APHost ().Origin (1) = ll;
                
                tmp1 -> setCPUModified (); // upload src
                tmp2 -> setSync (); // do NOT upload dest
                
                pi3 += ocl_basic_operator_kernel_11 ("dwt_3_alt", tmp1, lpf, hpf, tmp2, loc_mem3, gpu_chunk_size_dwt3_dim0, gpu_chunk_size_dwt3, gpu_line_length, gpu_line_length, gpu_chunk_size_dwt3_dim0, gpu_chunk_size_dwt3, lc3);
                
                // download results from GPU
                pi3.time_mem_down += tmp2 -> getData ();
                
              }
              
            }
                          
            vec_pi3.push_back (pi3);
            
            tmp1 -> APHost () = tmp_ap_array [0]; tmp1 -> APDevice () = tmp_ap_array [1]; // reset
            tmp2 -> APHost () = tmp_ap_array [2]; tmp2 -> APDevice () = tmp_ap_array [3]; // reset
            
            // shrink memory region for NEXT LEVEL
            tmp1 -> APHost ().Region (0) = tmp1 -> APDevice ().Region (0) = line_length/2 * sizeof (elem_type);
            tmp2 -> APHost ().Region (0) = tmp2 -> APDevice ().Region (0) = line_length/2 * sizeof (elem_type);
            tmp1 -> APHost ().Region (1) = tmp1 -> APDevice ().Region (1) = line_length/2;
            tmp2 -> APHost ().Region (1) = tmp2 -> APDevice ().Region (1) = line_length/2;
            tmp1 -> APHost ().Region (2) = tmp1 -> APDevice ().Region (2) = line_length/2;
            tmp2 -> APHost ().Region (2) = tmp2 -> APDevice ().Region (2) = line_length/2;
            tmp1 -> APDevice ().RowPitch () = tmp2 -> APDevice ().RowPitch () = (roundUp (line_length/2 + padding, roundTo)) * sizeof (elem_type);
            tmp1 -> APDevice ().SlicePitch () = tmp2 -> APDevice ().SlicePitch () = (roundUp (line_length/2 + padding, roundTo)) * line_length/2 * sizeof (elem_type);
                
          }
          
            
          delete loc_mem1;
          delete loc_mem2;
          delete loc_mem3;
          
          delete gpu_line_length;
          delete gpu_chunk_size_dwt12;
          delete gpu_chunk_size_dwt3;
          delete gpu_chunk_size_dwt3_dim0;
          
          ///////////////
          // performance
          ///////////////
          
# ifdef __PERFORMANCE_INFO__
          return dwt_performance (vec_pi, vec_pi2, vec_pi3, pi_final, lc1, lc2, lc3, levels, n, m, k, fl);
# else
          return std::vector <PerformanceInformation> (); 
# endif
          
      }
      
      
      static inline
      std::vector <PerformanceInformation>
      dwt_performance (const std::vector <ProfilingInformation> & vec_pi1,
                       const std::vector <ProfilingInformation> & vec_pi2,
                       const std::vector <ProfilingInformation> & vec_pi3,
                       const ProfilingInformation & pi_final,
                       const LaunchInformation & lc1,
                       const LaunchInformation & lc2,
                       const LaunchInformation & lc3,
                       const int levels,
                       const int n, const int m, const int k, const int fl)
      {
        
          // data amount for dwt1/2/3 over all levels
          int data_size_1 = 0;
          int data_size_2 = 0;
          int data_size_3 = 0;
          int data_size_final = 0;
          const int ng_0 = lc1.global_x / lc1.local_x;
          const int ng_1 = lc2.global_y / lc2.local_y;
          const int ng_2 = lc3.global_z / lc3.local_z;
          for (int i = 0; i < levels; i++)
          {
            const int sl_0 = n / pow (2, i);
            const int sl_1 = m / pow (2, i);
            const int sl_2 = k / pow (2, i);
            const int offset = fl - 2;
            // global -> local
            data_size_1 += (sl_0 + ng_0 * offset) * sl_1 * sl_2;
            data_size_2 += (sl_1 + ng_1 * offset) * sl_0 * sl_2;
            data_size_3 += (sl_2 + ng_2 * offset) * sl_0 * sl_1;
            if ((i&1))
              data_size_final += sl_0 * sl_1 * sl_2 * 0.875;
            // local -> global
            data_size_1 += sl_0 * sl_1 * sl_2;
            data_size_2 += sl_1 * sl_0 * sl_2;
            data_size_3 += sl_2 * sl_0 * sl_1;
            if ((i&1))
              data_size_final += sl_0 * sl_1 * sl_2 * 0.875;
          }
         
          std::vector <PerformanceInformation> vec_perf;
          
          // dwt1
          float time_seconds_1 = 0,
                time_mem_up_1 = 0,
                time_mem_down_1 = 0;
          for (std::vector <ProfilingInformation> :: const_iterator it_pi = vec_pi1.begin (); it_pi != vec_pi1.end (); ++it_pi)
          {
            time_seconds_1 += it_pi -> time_end - it_pi -> time_start;
            time_mem_up_1   += it_pi -> time_mem_up;
            time_mem_down_1 += it_pi -> time_mem_down;
          }
          float effective_bw_1 = (data_size_1 * sizeof (elem_type) * 1.0e-9f) / time_seconds_1;

          
          // dwt2
          float time_seconds_2 = 0,
                time_mem_up_2 = 0,
                time_mem_down_2 = 0;
          for (std::vector <ProfilingInformation> :: const_iterator it_pi = vec_pi2.begin (); it_pi != vec_pi2.end (); ++it_pi)
          {
            time_seconds_2 += it_pi -> time_end - it_pi -> time_start;
            time_mem_up_2   += it_pi -> time_mem_up;
            time_mem_down_2 += it_pi -> time_mem_down;
          }
          float effective_bw_2 = (data_size_2 * sizeof (elem_type) * 1.0e-9f) / time_seconds_2;

          // dwt3
          float time_seconds_3 = 0,
                time_mem_up_3 = 0,
                time_mem_down_3 = 0;
          for (std::vector <ProfilingInformation> :: const_iterator it_pi = vec_pi3.begin (); it_pi != vec_pi3.end (); ++it_pi)
          {
            time_seconds_3 += it_pi -> time_end - it_pi -> time_start;
            time_mem_up_3   += it_pi -> time_mem_up;
            time_mem_down_3 += it_pi -> time_mem_down;
          }
          float effective_bw_3 = (data_size_3 * sizeof (elem_type) * 1.0e-9f) / time_seconds_3;
          
          // dwt_final
          float time_seconds_final = pi_final.time_end - pi_final.time_start;
          float effective_bw_final = (data_size_final * sizeof (elem_type) * 1.0e-9f) / time_seconds_final;
          if (time_seconds_final < 1.0e-6)
          {
            data_size_final = 0;
            effective_bw_final = 0;
          }

          float effective_bw = ((data_size_1 + data_size_2 + data_size_3 + data_size_final) * sizeof (elem_type) * 1.0e-9f) / (time_seconds_1 + time_seconds_2 + time_seconds_3 + time_seconds_final);
          float time_seconds = time_seconds_1 + time_seconds_2 + time_seconds_3 + time_seconds_final;
          float time_mem_up = time_mem_up_1 + time_mem_up_2 + time_mem_up_3 + pi_final.time_mem_up;
          float time_mem_down = time_mem_down_1 + time_mem_down_2 + time_mem_down_3 + pi_final.time_mem_down;
          vec_perf.push_back (PerformanceInformation ("dwt1+2+3 (all levels)", lc1, " Effective bandwidth (GB/s)", time_seconds, time_mem_up, time_mem_down, effective_bw));
          vec_perf.push_back (PerformanceInformation ("dwt_1_alt (all levels)", lc1, " Effective bandwidth (GB/s)", time_seconds_1, time_mem_up_1, time_mem_down_1, effective_bw_1));
          vec_perf.push_back (PerformanceInformation ("dwt_2_alt (all levels)", lc2, " Effective bandwidth (GB/s)", time_seconds_2, time_mem_up_2, time_mem_down_2, effective_bw_2));
          vec_perf.push_back (PerformanceInformation ("dwt_3_alt (all levels)", lc3, " Effective bandwidth (GB/s)", time_seconds_3, time_mem_up_3, time_mem_down_3, effective_bw_3));
          vec_perf.push_back (PerformanceInformation ("dwt_final_alt (all levels)", lc1, " Effective bandwidth (GB/s)", time_seconds_final, pi_final.time_mem_up, pi_final.time_mem_down, effective_bw_final));

          return vec_perf;
          
      }
      
      
      
      
      /**
       * @brief                       3D Discrete Wavelet Transform.
       *
       * @param  arg1                 Address of signal (n x m x k).
       * @param  arg2                 Address of resulting DWT (n x m x k).
       * @param  n                    First dimension.
       * @param  m                    Second dimension.
       * @param  k                    Third dimension.
       * @param  filter               Convolution kernel.
       * @param  fl                   Length of convolution kernel.
       * ...
       */
      static inline
      std::vector <PerformanceInformation>
      ocl_operator_idwt2               ( oclDataObject * const arg1,
                                                  int            n,
                                                  int            m,
                                                  int            k,
                                        oclDataObject * const  lpf,
                                        oclDataObject * const  hpf,
                                                  int           fl,
                                            const int         levels,
                                        oclDataObject * const arg2)
      {
        
          std::vector <PerformanceInformation> vec_perf;
        
          print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                             trait2 :: print_elem_type (), "> :: ocl_operator_idwt2", op_v_level);

          
          // create launch configuration
          const LaunchInformation & lc = oclConnection :: Instance () -> getThreadConfig (std::string ("idwt2"));
          
          // dynamically allocate local memory
          const int num_loc_mem_elems = (n / (lc.global_x/lc.local_x) + 2*(fl-1)) * (n / (lc.global_y/lc.local_y) + 2*(fl-1)) + (n / (lc.global_x/lc.local_x) + 2*(fl-1)) * (n / (lc.global_y/lc.local_y));
          oclDataObject * loc_mem = new oclLocalMemObject <elem_type> (num_loc_mem_elems);
          std::cout << " local_mem (idwt2): " << num_loc_mem_elems * sizeof (elem_type) << " Bytes " << std::endl;
          
          std::vector <ProfilingInformation> vec_pi_1, vec_pi_2;
          
          const int line_length = n/pow (2,levels-1);
          ProfilingInformation pi = ocl_basic_operator_kernel_56 ("idwt2", arg1, lpf, hpf, arg2, loc_mem, n, m, k, line_length, 1, num_loc_mem_elems, lc);
          vec_pi_1.push_back (pi);
          
          for (int i = levels-2; i >= 0; i--)
          {
            const int line_length2 = n / pow (2, i);
            ProfilingInformation pi_tmp2 = ocl_basic_operator_kernel_25 ("idwt2_prepare", arg1, arg2, n, m, k, line_length2/2, 1, lc);
            ProfilingInformation pi_tmp1 = ocl_basic_operator_kernel_56 ("idwt2", arg1, lpf, hpf, arg2, loc_mem, n, m, k, line_length2, 1, num_loc_mem_elems, lc);
            vec_pi_2.push_back (pi_tmp2);
            vec_pi_1.push_back (pi_tmp1);
          }
          
          arg2 -> getData ();
          
# ifdef __PERFORMANCE_INFO__
          
          const int num_groups_0 = lc.global_x / lc.local_x;
          const int num_groups_1 = lc.global_y / lc.local_y;
          
          // data amount for idwt2 over all levels
          int data_size_1 = 0;
          for (int i = levels-1; i >= 0; i--)
          {
            const int sl_0 = n / pow (2, i);
            const int sl_1 = m / pow (2, i);
            const float block_size_0 = sl_0 / num_groups_0;
            const float block_size_1 = sl_1 / num_groups_1;
            const int offset = fl - 1;
            // global -> local
            data_size_1 += (block_size_0 + 2 * offset) * (block_size_1 + 2 * offset) * num_groups_0 * num_groups_1;
            // local -> global
            data_size_1 += sl_0 * sl_1;
          }
          
          float time_seconds_1 = 0,
                time_mem_up_1 = 0,
                time_mem_down_1 = 0;
          for (std::vector <ProfilingInformation> :: const_iterator it = vec_pi_1.begin (); it != vec_pi_1.end (); ++it)
          {
            time_seconds_1 += it -> time_end - it -> time_start;
            time_mem_up_1 += it -> time_mem_up;
            time_mem_down_1 += it -> time_mem_down;
          }
          float effective_bw_1 = ((float) (data_size_1 * sizeof (elem_type)) * 1.0e-9f) / time_seconds_1;
          
          // data amount for idwt2_prepare
          int data_size_2 = 0;
          for (int l = 2; l <= levels; l++)
          {
            const int sl_0 = n / pow (2, l-1);
            const int sl_1 = m / pow (2, l-1);
            data_size_2 += sl_0 * sl_1;
          }
          
          float time_seconds_2 = 0,
                time_mem_up_2 = 0,
                time_mem_down_2 = 0;
          for (std::vector <ProfilingInformation> :: const_iterator it = vec_pi_2.begin (); it != vec_pi_2.end (); ++it)
          {
            time_seconds_2 += it -> time_end - it -> time_start;
            time_mem_up_2 += it -> time_mem_up;
            time_mem_down_2 += it -> time_mem_down;
          }
          float effective_bw_2 = ((float)(data_size_2 * sizeof (elem_type)) * 1.0e-9f) / time_seconds_2;
          
          // overall bandwidth
          float effective_bw = ((float)((data_size_1 + data_size_2) * sizeof (elem_type)) * 1.0e-9f) / (time_seconds_1 + time_seconds_2);
          
            vec_perf.push_back (PerformanceInformation ("idwt2 (+prepare)", lc, " Effective bandwidth (GB/s)", time_seconds_1 + time_seconds_2, time_mem_up_1 + time_mem_up_2, time_mem_down_1 + time_mem_down_2, effective_bw));
            vec_perf.push_back (PerformanceInformation ("idwt2 (kernel)", lc, " Effective bandwidth (GB/s)", time_seconds_1, time_mem_up_1, time_mem_down_1, effective_bw_1));
            vec_perf.push_back (PerformanceInformation ("idwt2_prepare (kernel)", lc, " Effective bandwidth (GB/s)", time_seconds_2, time_mem_up_2, time_mem_down_2, effective_bw_2));
                        
# endif
            
          delete loc_mem;
          
          return vec_perf;
          
      }
      

      
      static inline
      std::vector <PerformanceInformation>
      ocl_operator_idwt3               ( oclDataObject * const arg1,
                                                  int            n,
                                                  int            m,
                                                  int            k,
                                        oclDataObject * const  lpf,
                                        oclDataObject * const  hpf,
                                                  int           fl,
                                            const int         levels,
                                        oclDataObject * const arg2,
                                            const int         chunk_size)
      {
        
          std::vector <PerformanceInformation> vec_perf;
        
          print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                             trait2 :: print_elem_type (), "> :: ocl_operator_idwt3", op_v_level);

          
          // create launch configuration
          LaunchInformation lc = oclConnection :: Instance () -> getThreadConfig (std::string ("idwt2"));
          LaunchInformation lc2 = oclConnection :: Instance () -> getThreadConfig (std::string ("idwt3"));
          
          /////////////
          // dynamically allocate local memory
          /////////////
          
          
          lc2.global_z = lc2.local_z = 16;
          
          // idwt2
          const int num_loc_mem_elems = (n / (lc.global_x/lc.local_x) + 2*(fl-1)) * (n / (lc.global_y/lc.local_y) + 2*(fl-1)) + (n / (lc.global_x/lc.local_x) + 2*(fl-1)) * (n / (lc.global_y/lc.local_y)) * lc.local_z;
          oclDataObject * loc_mem = new oclLocalMemObject <elem_type> (num_loc_mem_elems);
          
          // idwt3
          const int num_loc_mem_elems2 = (2 * (k + fl-1)) * lc2.local_x * lc2.local_y;
          oclDataObject * loc_mem2 = new oclLocalMemObject <elem_type> (num_loc_mem_elems2);
          
          std::cout << " local_mem (idwt3): " << num_loc_mem_elems * sizeof (elem_type) << " Bytes " << std::endl;
          std::cout << " local_mem2 (idwt3): " << num_loc_mem_elems2 * sizeof (elem_type) << " Bytes " << std::endl;
          std::cout << " chunk_size: " << chunk_size << std::endl;

          /////////////
          // collect profiling information for each level
          /////////////
          std::vector <ProfilingInformation> vec_pi_1, vec_pi_2;

          //////////
          // prepare access patterns for CPU-GPU-transfers
          //////////
          arg1 -> APDevice () = oclAccessPattern (0, 0, 0,
                  n * sizeof (elem_type), n * m * sizeof (elem_type),
                  n * sizeof (elem_type), m, k);
          arg1 -> APHost () = arg1 -> APDevice ();
          arg2 -> APDevice () = arg2 -> APHost () = arg1 -> APHost ();
          
          ///////////
          // temporaries (avoid writing to source memory)
          ///////////
          oclDataObject * tmp1 = arg1;
          oclDataObject * tmp2 = arg2;
          
//          lc2.global_z = lc2.local_z = 16;
          
          for (int i = levels-1; i >= 0; i--)
          {
            const int line_length = n / pow (2, i);
            
            const int chunk_size_idwt3 = min (line_length, chunk_size);
            const int chunk_size_idwt3_dim0 = line_length;
//            lc2.global_z = lc2.local_z = (line_length/4 < lc2.local_z ? line_length/4 : lc2.local_z);
            lc2.global_x /= (chunk_size_idwt3_dim0 < lc2.global_x ? 2 : 1);
            lc2.global_y /= (chunk_size_idwt3 < lc2.global_y ? 2 : 1);
            
            // run kernel "idwt3"
            ProfilingInformation pi2 = {0, 0, 0, 0};
            const oclAccessPattern tmp_ap_array [4] = {tmp1 -> APHost (), tmp1 -> APDevice (),
                                                       tmp2 -> APHost (), tmp2 -> APDevice ()}; // save current state !!!
            tmp1 -> APHost ().Region (0) = tmp1 -> APDevice ().Region (0) = chunk_size_idwt3_dim0 * sizeof (elem_type);
            tmp2 -> APHost ().Region (0) = tmp2 -> APDevice ().Region (0) = chunk_size_idwt3_dim0 * sizeof (elem_type);
            tmp1 -> APHost ().Region (1) = tmp1 -> APDevice ().Region (1) = chunk_size_idwt3;
            tmp2 -> APHost ().Region (1) = tmp2 -> APDevice ().Region (1) = chunk_size_idwt3;
            tmp1 -> APHost ().Region (2) = tmp1 -> APDevice ().Region (2) = line_length;
            tmp2 -> APHost ().Region (2) = tmp2 -> APDevice ().Region (2) = line_length;
            tmp1 -> APDevice ().RowPitch () = tmp2 -> APDevice ().RowPitch () = chunk_size_idwt3_dim0 * sizeof (elem_type);
            tmp1 -> APDevice ().SlicePitch () = tmp2 -> APDevice ().SlicePitch () = chunk_size_idwt3_dim0 * chunk_size_idwt3 * sizeof (elem_type);
            for (int l = 0; l < line_length; l += chunk_size_idwt3_dim0)
            {
              tmp1 -> APHost ().Origin (0) = tmp2 -> APHost ().Origin (0) = l * sizeof (elem_type);
              for (int ll = 0; ll < line_length; ll += chunk_size_idwt3)
              {
                tmp1 -> APHost ().Origin (1) = tmp2 -> APHost ().Origin (1) = ll;
                tmp1 -> setCPUModified (); // upload src
                tmp2 -> setSync (); // do not upload data to result buffer
                pi2 += ocl_basic_operator_kernel_57 ("idwt3", tmp1, lpf, hpf, tmp2, loc_mem2, n, m, k, line_length, chunk_size_idwt3_dim0, chunk_size_idwt3, num_loc_mem_elems2, lc2);
                pi2.time_mem_down += tmp2 -> getData ();
              }
            }
            vec_pi_2.push_back (pi2);
            
            tmp1 -> APHost () = tmp_ap_array [0]; tmp1 -> APDevice () = tmp_ap_array [1]; // reset
            tmp2 -> APHost () = tmp_ap_array [2]; tmp2 -> APDevice () = tmp_ap_array [3]; // reset
            
            // run kernel "idwt2" on slices
            ProfilingInformation pi1 = {0, 0, 0, 0};
            const int chunk_size_idwt2 = min (line_length, chunk_size);
            lc.local_z = (chunk_size_idwt2 < lc.local_z ? chunk_size_idwt2 : lc.local_z);
            lc.global_z = (chunk_size_idwt2 < lc.global_z ? chunk_size_idwt2 : lc.global_z);
            lc.global_x /= (line_length < lc.global_x ? 2 : 1);
            lc.global_y /= (line_length < lc.global_y ? 2 : 1);
            tmp1 -> APHost ().Region (0) = tmp1 -> APDevice ().Region (0) = line_length * sizeof (elem_type);
            tmp2 -> APHost ().Region (0) = tmp2 -> APDevice ().Region (0) = line_length * sizeof (elem_type);
            tmp1 -> APHost ().Region (1) = tmp1 -> APDevice ().Region (1) = line_length;
            tmp2 -> APHost ().Region (1) = tmp2 -> APDevice ().Region (1) = line_length;
            tmp1 -> APHost ().Region (2) = tmp1 -> APDevice ().Region (2) = chunk_size_idwt2;
            tmp2 -> APHost ().Region (2) = tmp2 -> APDevice ().Region (2) = chunk_size_idwt2;
            for (int l = 0; l < line_length; l += chunk_size_idwt2)
            {
              tmp1 -> APHost ().Origin (2) = tmp2 -> APHost ().Origin (2) = l;
              tmp2 -> setCPUModified (); // upload src
              tmp1 -> setSync (); // do not upload data to result buffer
              pi1 += ocl_basic_operator_kernel_56 ("idwt2", tmp2, lpf, hpf, tmp1, loc_mem, n, m, k, line_length, chunk_size_idwt2, num_loc_mem_elems, lc);
              pi1.time_mem_down += tmp1 -> getData ();
            }
            vec_pi_1.push_back (pi1);
            
            tmp1 -> APHost ().Region (2) = tmp1 -> APDevice ().Region (2) = k; // reset
            tmp2 -> APHost ().Region (2) = tmp2 -> APDevice ().Region (2) = k; // reset
            tmp1 -> APHost ().Origin (2) = tmp2 -> APHost ().Origin (2) = 0; // reset
            
          }
          
# ifdef __PERFORMANCE_INFO__
          
          const int num_groups_0 = lc.global_x / lc.local_x;
          const int num_groups_1 = lc.global_y / lc.local_y;
          
          // data amount for idwt2 over all levels
          int data_size_1 = 0;
          int data_size_2 = 0;
          for (int i = levels-1; i >= 0; i--)
          {
            const int sl_0 = n / pow (2, i);
            const int sl_1 = m / pow (2, i);
            const int sl_2 = k / pow (2, i);
            const float block_size_0 = sl_0 / num_groups_0;
            const float block_size_1 = sl_1 / num_groups_1;
            const int offset = fl - 1;
            // global -> local
            data_size_1 += (block_size_0 + 2 * offset) * (block_size_1 + 2 * offset) * num_groups_0 * num_groups_1 * sl_2;
            data_size_2 += sl_0 * sl_1 * (sl_2 + fl-1) * 2;
            // local -> global
            data_size_1 += sl_0 * sl_1 * sl_2;
            data_size_2 += sl_0 * sl_1 * sl_2;
          }
          
          float time_seconds_1 = 0,
                time_mem_up_1 = 0,
                time_mem_down_1 = 0;
          for (std::vector <ProfilingInformation> :: const_iterator it = vec_pi_1.begin (); it != vec_pi_1.end (); ++it)
          {
            time_seconds_1 += it -> time_end - it -> time_start;
            time_mem_up_1 += it -> time_mem_up;
            time_mem_down_1 += it -> time_mem_down;
          }
          float effective_bw_1 = ((float) (data_size_1 * sizeof (elem_type)) * 1.0e-9f) / time_seconds_1;
          
          float time_seconds_2 = 0,
                time_mem_up_2 = 0,
                time_mem_down_2 = 0;
          for (std::vector <ProfilingInformation> :: const_iterator it = vec_pi_2.begin (); it != vec_pi_2.end (); ++it)
          {
            time_seconds_2 += it -> time_end - it -> time_start;
            time_mem_up_2 += it -> time_mem_up;
            time_mem_down_2 += it -> time_mem_down;
          }
          float effective_bw_2 = ((float)(data_size_2 * sizeof (elem_type)) * 1.0e-9f) / time_seconds_2;
          
          // overall bandwidth
          float effective_bw = ((float)((data_size_1 + data_size_2) * sizeof (elem_type)) * 1.0e-9f) / (time_seconds_1 + time_seconds_2);
          
            vec_perf.push_back (PerformanceInformation ("idwt2 (+idwt3)", lc, " Effective bandwidth (GB/s)", time_seconds_1 + time_seconds_2, time_mem_up_1 + time_mem_up_2, time_mem_down_1 + time_mem_down_2, effective_bw));
            vec_perf.push_back (PerformanceInformation ("idwt2 (kernel)", lc, " Effective bandwidth (GB/s)", time_seconds_1, time_mem_up_1, time_mem_down_1, effective_bw_1));
            vec_perf.push_back (PerformanceInformation ("idwt3 (kernel)", lc, " Effective bandwidth (GB/s)", time_seconds_2, time_mem_up_2, time_mem_down_2, effective_bw_2));
                        
# endif
            
          delete loc_mem;
          delete loc_mem2;
          
          return vec_perf;
          
      }
  
      
            static inline
      std::vector <PerformanceInformation>
      ocl_operator_idwt3_alt           ( oclDataObject * const arg1,
                                                  int            n,
                                                  int            m,
                                                  int            k,
                                        oclDataObject * const  lpf,
                                        oclDataObject * const  hpf,
                                                  int           fl,
                                            const int         levels,
                                        oclDataObject * const arg2,
                                            const int         chunk_size)
      {
                
          print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                             trait2 :: print_elem_type (), "> :: ocl_operator_idwt3", op_v_level);

          
          // create launch configuration
          LaunchInformation lc1 = oclConnection :: Instance () -> getThreadConfig (std::string ("idwt_1_alt"));
          LaunchInformation lc2 = oclConnection :: Instance () -> getThreadConfig (std::string ("idwt_2_alt"));
          LaunchInformation lc3 = oclConnection :: Instance () -> getThreadConfig (std::string ("idwt_3_alt"));
          
          /////////////
          // dynamically allocate local memory
          /////////////
          
          const int loc_mem_line_in = 64 + 2 * (fl-1) + 1;
          const int loc_mem_line_out = 64 + 1; 
          
          // idwt1
          const int num_loc_mem_elems1 = (loc_mem_line_in + loc_mem_line_out) * lc1.local_y * lc1.local_z;
          oclDataObject * loc_mem1 = new oclLocalMemObject <elem_type> (num_loc_mem_elems1);

          // idwt2
          const int num_loc_mem_elems2 = (loc_mem_line_in + loc_mem_line_out) * lc2.local_x * lc2.local_z;
          oclDataObject * loc_mem2 = new oclLocalMemObject <elem_type> (num_loc_mem_elems2);
          
          // idwt3
          const int num_loc_mem_elems3 = (loc_mem_line_in + loc_mem_line_out) * lc3.local_x * lc3.local_y;
          oclDataObject * loc_mem3 = new oclLocalMemObject <elem_type> (num_loc_mem_elems3);
          
          std::cout << " local_mem (idwt_1_alt): " << num_loc_mem_elems1 * sizeof (elem_type) << " Bytes " << std::endl;
          std::cout << " local_mem2 (idwt_2_alt): " << num_loc_mem_elems2 * sizeof (elem_type) << " Bytes " << std::endl;
          std::cout << " local_mem3 (idwt_3_alt): " << num_loc_mem_elems3 * sizeof (elem_type) << " Bytes " << std::endl;
          std::cout << " chunk_size: " << chunk_size << std::endl;

          /////////////
          // collect profiling information for each level
          /////////////
          std::vector <ProfilingInformation> vec_pi_1, vec_pi_2, vec_pi_3;

          ///////////
          // kernel arguments
          ///////////
          oclDataObject * gpu_n = new oclGPUDataObject <int> (&n, 1);
          oclDataObject * gpu_m = new oclGPUDataObject <int> (&m, 1);
          oclDataObject * gpu_k = new oclGPUDataObject <int> (&k, 1);
          int line_length = 0;
          oclDataObject * gpu_line_length = new oclGPUDataObject <int> (&line_length, 1);

          //////////
          // prepare access patterns for CPU-GPU-transfers
          //////////
          arg1 -> APDevice () = oclAccessPattern (0, 0, 0,
                  n * sizeof (elem_type), n * m * sizeof (elem_type),
                  n * sizeof (elem_type), m, k);
          arg1 -> APHost () = arg1 -> APDevice ();
          arg2 -> APDevice () = arg2 -> APHost () = arg1 -> APHost ();
          
          ///////////
          // temporaries (avoid writing to source memory)
          ///////////
          oclDataObject * tmp1 = arg1;
          oclDataObject * tmp2 = arg2;
          
          ProfilingInformation pi_final = {0,0,0,0};
          
          for (int i = levels-1; i >= 0; i--)
          {
            line_length = n / pow (2, i);
            gpu_line_length -> setCPUModified ();
            
            // run kernel "idwt3"
            ProfilingInformation pi3 = {0, 0, 0, 0};
            if (i == levels-1)
              tmp2 -> setCPUModified (); // upload src
            else
              tmp2 -> setSync ();
            tmp1 -> setSync (); // do not upload data to result buffer
            pi3 += ocl_basic_operator_kernel_11 ("idwt_3_alt", tmp2, lpf, hpf, tmp1, loc_mem3, gpu_n, gpu_m, gpu_k, gpu_line_length, gpu_line_length, gpu_line_length, lc3);
            vec_pi_3.push_back (pi3);
            
            // run kernel "idwt_1" and "idwt_2" on slices
            ProfilingInformation pi1 = {0, 0, 0, 0};
            ProfilingInformation pi2 = {0, 0, 0, 0};
            
            tmp1 -> setSync (); // upload src
            tmp2 -> setSync (); // do not upload data to result buffer
            pi2 += ocl_basic_operator_kernel_10 ("idwt_2_alt", tmp1, lpf, hpf, tmp2, loc_mem2, gpu_n, gpu_m, gpu_k, gpu_line_length, gpu_line_length, lc2);
            tmp1 -> setSync (); // do not upload src
            tmp2 -> setSync (); // do not upload data to dest
            pi1 += ocl_basic_operator_kernel_10 ("idwt_1_alt", tmp2, lpf, hpf, tmp1, loc_mem1, gpu_n, gpu_m, gpu_k, gpu_line_length, gpu_line_length, lc1);
            vec_pi_1.push_back (pi1);
            vec_pi_2.push_back (pi2);
            
            if (i != 0)
            {
              pi_final += ocl_basic_operator_kernel_10 ("idwt_final_alt", tmp1, lpf, hpf, tmp2, loc_mem1, gpu_n, gpu_m, gpu_k, gpu_line_length, gpu_line_length, lc1);
              
            }
            
          }
          
          pi_final.time_mem_down += tmp1 -> getData ();
          
          delete loc_mem1;
          delete loc_mem2;
          delete loc_mem3;
          
          delete gpu_n;
          delete gpu_m;
          delete gpu_k;
          delete gpu_line_length;
          
# ifdef __PERFORMANCE_INFO__
          return idwt_performance (vec_pi_1, vec_pi_2, vec_pi_3, pi_final, lc1, lc2, lc3, levels, n, m, k, fl);
# else
          return std::vector <PerformanceInformation> ();
# endif
          
      }
            
            
      static inline
      std::vector <PerformanceInformation>
      idwt_performance (const std::vector <ProfilingInformation> & vec_pi1,
                        const std::vector <ProfilingInformation> & vec_pi2,
                        const std::vector <ProfilingInformation> & vec_pi3,
                        const ProfilingInformation & pi_final,
                        const LaunchInformation & lc1,
                        const LaunchInformation & lc2,
                        const LaunchInformation & lc3,
                        const int levels,
                        const int n, const int m, const int k, const int fl)
      {
        
          std::vector <PerformanceInformation> vec_perf;
        
          // data amount for idwt2 over all levels
          int data_size_1 = 0;
          int data_size_2 = 0;
          int data_size_3 = 0;
          int data_size_final = 0;
          const int ng_0 = lc1.global_x / lc1.local_x;
          const int ng_1 = lc2.global_y / lc2.local_y;
          const int ng_2 = lc3.global_z / lc3.local_z;
          for (int i = levels-1; i >= 0; i--)
          {
            const int sl_0 = n / pow (2, i);
            const int sl_1 = m / pow (2, i);
            const int sl_2 = k / pow (2, i);
            const int offset = fl - 1;
            // global -> local
            data_size_1 += sl_1 * sl_2 * (sl_0 + ng_0 * 2 * (fl-1));
            data_size_2 += sl_0 * sl_2 * (sl_1 + ng_1 * 2 * (fl-1));
            data_size_3 += sl_0 * sl_1 * (sl_2 + ng_2 * 2 * (fl-1));
            if (i != 0)
              data_size_final += sl_0 * sl_1 * sl_2;
            // local -> global
            data_size_1 += sl_0 * sl_1 * sl_2;
            data_size_2 += sl_0 * sl_1 * sl_2;
            data_size_3 += sl_0 * sl_1 * sl_2;
            if (i != 0)
              data_size_final += sl_0 * sl_1 * sl_2;
          }
          
          float time_seconds_1 = 0,
                time_mem_up_1 = 0,
                time_mem_down_1 = 0;
          for (std::vector <ProfilingInformation> :: const_iterator it = vec_pi1.begin (); it != vec_pi1.end (); ++it)
          {
            time_seconds_1 += it -> time_end - it -> time_start;
            time_mem_up_1 += it -> time_mem_up;
            time_mem_down_1 += it -> time_mem_down;
          }
          float effective_bw_1 = ((float) (data_size_1 * sizeof (elem_type)) * 1.0e-9f) / time_seconds_1;

          float time_seconds_2 = 0,
                time_mem_up_2 = 0,
                time_mem_down_2 = 0;
          for (std::vector <ProfilingInformation> :: const_iterator it = vec_pi2.begin (); it != vec_pi2.end (); ++it)
          {
            time_seconds_2 += it -> time_end - it -> time_start;
            time_mem_up_2 += it -> time_mem_up;
            time_mem_down_2 += it -> time_mem_down;
          }
          float effective_bw_2 = ((float) (data_size_2 * sizeof (elem_type)) * 1.0e-9f) / time_seconds_2;
                    
          float time_seconds_3 = 0,
                time_mem_up_3 = 0,
                time_mem_down_3 = 0;
          for (std::vector <ProfilingInformation> :: const_iterator it = vec_pi3.begin (); it != vec_pi3.end (); ++it)
          {
            time_seconds_3 += it -> time_end - it -> time_start;
            time_mem_up_3 += it -> time_mem_up;
            time_mem_down_3 += it -> time_mem_down;
          }
          float effective_bw_3 = ((float)(data_size_3 * sizeof (elem_type)) * 1.0e-9f) / time_seconds_3;
          
          float time_seconds_final = pi_final.time_end - pi_final.time_start;
          float effective_bw_final = ((float) (data_size_final * sizeof(elem_type))*1.0e-9f) / time_seconds_final;
          if (time_seconds_final < 1.0e-6)
          {
            data_size_final = 0;
            effective_bw_final = 0;
          }
          
          // overall bandwidth
          float time_seconds = time_seconds_1 + time_seconds_2 + time_seconds_3 + time_seconds_final;
          float data_size = data_size_1 + data_size_2 + data_size_3 + data_size_final;
          float effective_bw = ((float)((data_size) * sizeof (elem_type)) * 1.0e-9f) / (time_seconds);
          
            vec_perf.push_back (PerformanceInformation ("idwt1+2+3", lc1, " Effective bandwidth (GB/s)", time_seconds, time_mem_up_1 + time_mem_up_2 + time_mem_up_3 + pi_final.time_mem_up, time_mem_down_1 + time_mem_down_2 + time_mem_down_3 + pi_final.time_mem_down, effective_bw));
            vec_perf.push_back (PerformanceInformation ("idwt_1_alt (kernel)", lc1, " Effective bandwidth (GB/s)", time_seconds_1, time_mem_up_1, time_mem_down_1, effective_bw_1));
            vec_perf.push_back (PerformanceInformation ("idwt_2_alt (kernel)", lc2, " Effective bandwidth (GB/s)", time_seconds_2, time_mem_up_2, time_mem_down_2, effective_bw_2));
            vec_perf.push_back (PerformanceInformation ("idwt_3_alt (kernel)", lc3, " Effective bandwidth (GB/s)", time_seconds_3, time_mem_up_3, time_mem_down_3, effective_bw_3));
            vec_perf.push_back (PerformanceInformation ("idwt_final_alt (kernel)", lc1, " Effective bandwidth (GB/s)", time_seconds_final, pi_final.time_mem_up, pi_final.time_mem_down, effective_bw_final));

            return vec_perf;
            
      }
      
      
      
      static inline
      std::vector <PerformanceInformation>
      ocl_operator_idwt3_alt_cs       ( oclDataObject * const arg1,
                                                  int            n,
                                                  int            m,
                                                  int            k,
                                        oclDataObject * const  lpf,
                                        oclDataObject * const  hpf,
                                                  int           fl,
                                            const int         levels,
                                        oclDataObject * const arg2,
                                            const int         chunk_size)
      {
        
          print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                             trait2 :: print_elem_type (), "> :: ocl_operator_idwt3_alt_cs", op_v_level);

          
          // create launch configuration
          LaunchInformation lc1 = oclConnection :: Instance () -> getThreadConfig (std::string ("idwt_1_alt"));
          LaunchInformation lc2 = oclConnection :: Instance () -> getThreadConfig (std::string ("idwt_2_alt"));
          LaunchInformation lc3 = oclConnection :: Instance () -> getThreadConfig (std::string ("idwt_3_alt"));
          
          /////////////
          // dynamically allocate local memory
          /////////////
          
          const int loc_mem_line_in = 64 + 2 * (fl-1) + 1;
          const int loc_mem_line_out = 64 + 1;
          
          // idwt1
          const int num_loc_mem_elems1 = (loc_mem_line_in + loc_mem_line_out) * lc1.local_y * lc1.local_z;
          oclDataObject * loc_mem1 = new oclLocalMemObject <elem_type> (num_loc_mem_elems1);

          // idwt2
          const int num_loc_mem_elems2 = (loc_mem_line_in + loc_mem_line_out) * lc2.local_x * lc2.local_z;
          oclDataObject * loc_mem2 = new oclLocalMemObject <elem_type> (num_loc_mem_elems2);
          
          // idwt3
          const int num_loc_mem_elems3 = (loc_mem_line_in + loc_mem_line_out) * lc3.local_x * lc3.local_y;
          oclDataObject * loc_mem3 = new oclLocalMemObject <elem_type> (num_loc_mem_elems3);
          
          std::cout << " local_mem (idwt_1_alt): " << num_loc_mem_elems1 * sizeof (elem_type) << " Bytes " << std::endl;
          std::cout << " local_mem2 (idwt_2_alt): " << num_loc_mem_elems2 * sizeof (elem_type) << " Bytes " << std::endl;
          std::cout << " local_mem3 (idwt_3_alt): " << num_loc_mem_elems3 * sizeof (elem_type) << " Bytes " << std::endl;
          std::cout << " chunk_size: " << chunk_size << std::endl;

          /////////////
          // collect profiling information for each level
          /////////////
          std::vector <ProfilingInformation> vec_pi_1, vec_pi_2, vec_pi_3;

          /////////////
          // kernel arguments
          /////////////
          int line_length = 0;
          oclDataObject * gpu_line_length = new oclGPUDataObject <int> (&line_length, 1);
          int chunk_size_idwt3 = 0;
          oclDataObject * gpu_chunk_size_idwt3 = new oclGPUDataObject <int> (&chunk_size_idwt3, 1);
          int chunk_size_idwt3_dim0 = 0;
          oclDataObject * gpu_chunk_size_idwt3_dim0 = new oclGPUDataObject <int> (&chunk_size_idwt3_dim0, 1);
          int chunk_size_idwt12 = 0;
          oclDataObject * gpu_chunk_size_idwt12 = new oclGPUDataObject <int> (&chunk_size_idwt12, 1);

          //////////
          // prepare access patterns for CPU-GPU-transfers
          //////////
          arg1 -> APDevice () = oclAccessPattern (0, 0, 0,
                  n * sizeof (elem_type), n * m * sizeof (elem_type),
                  n * sizeof (elem_type), m, k);
          arg1 -> APHost () = arg1 -> APDevice ();
          arg2 -> APDevice () = arg2 -> APHost () = arg1 -> APHost ();
          
          ///////////
          // temporaries (avoid writing to source memory)
          ///////////
          oclDataObject * tmp1 = arg1;
          oclDataObject * tmp2 = arg2;
          
          for (int i = levels-1; i >= 0; i--)
          {
            line_length = n / pow (2, i);
            
            chunk_size_idwt3 = min (line_length, chunk_size);
            chunk_size_idwt3_dim0 = line_length;
            
            gpu_line_length -> setCPUModified ();
            gpu_chunk_size_idwt3 -> setCPUModified ();
            gpu_chunk_size_idwt3_dim0 -> setCPUModified ();
            
            // run kernel "idwt3"
            ProfilingInformation pi3 = {0, 0, 0, 0};
            const oclAccessPattern tmp_ap_array [4] = {tmp1 -> APHost (), tmp1 -> APDevice (),
                                                       tmp2 -> APHost (), tmp2 -> APDevice ()}; // save current state !!!
            tmp1 -> APHost ().Region (0) = tmp1 -> APDevice ().Region (0) = chunk_size_idwt3_dim0 * sizeof (elem_type);
            tmp2 -> APHost ().Region (0) = tmp2 -> APDevice ().Region (0) = chunk_size_idwt3_dim0 * sizeof (elem_type);
            tmp1 -> APHost ().Region (1) = tmp1 -> APDevice ().Region (1) = chunk_size_idwt3;
            tmp2 -> APHost ().Region (1) = tmp2 -> APDevice ().Region (1) = chunk_size_idwt3;
            tmp1 -> APHost ().Region (2) = tmp1 -> APDevice ().Region (2) = line_length;
            tmp2 -> APHost ().Region (2) = tmp2 -> APDevice ().Region (2) = line_length;
            tmp1 -> APDevice ().RowPitch () = tmp2 -> APDevice ().RowPitch () = chunk_size_idwt3_dim0 * sizeof (elem_type);
            tmp1 -> APDevice ().SlicePitch () = tmp2 -> APDevice ().SlicePitch () = chunk_size_idwt3_dim0 * chunk_size_idwt3 * sizeof (elem_type);
            for (int l = 0; l < line_length; l += chunk_size_idwt3_dim0)
            {
              tmp1 -> APHost ().Origin (0) = tmp2 -> APHost ().Origin (0) = l * sizeof (elem_type);
              for (int ll = 0; ll < line_length; ll += chunk_size_idwt3)
              {
                tmp1 -> APHost ().Origin (1) = tmp2 -> APHost ().Origin (1) = ll;
                tmp1 -> setCPUModified (); // upload src
                tmp2 -> setSync (); // do not upload data to result buffer
                pi3 += ocl_basic_operator_kernel_11 ("idwt_3_alt", tmp1, lpf, hpf, tmp2, loc_mem3, gpu_chunk_size_idwt3_dim0, gpu_chunk_size_idwt3, gpu_line_length, gpu_line_length, gpu_chunk_size_idwt3_dim0, gpu_chunk_size_idwt3, lc3);
                pi3.time_mem_down += tmp2 -> getData ();
              }
            }
            vec_pi_3.push_back (pi3);
            
            tmp1 -> APHost () = tmp_ap_array [0]; tmp1 -> APDevice () = tmp_ap_array [1]; // reset
            tmp2 -> APHost () = tmp_ap_array [2]; tmp2 -> APDevice () = tmp_ap_array [3]; // reset
            
            // run kernel "idwt_1" and "idwt_2" on slices
            ProfilingInformation pi1 = {0, 0, 0, 0};
            ProfilingInformation pi2 = {0, 0, 0, 0};
            
            chunk_size_idwt12 = min (line_length, chunk_size);
            gpu_chunk_size_idwt12 -> setCPUModified ();
            
            tmp1 -> APHost ().Region (0) = tmp1 -> APDevice ().Region (0) = line_length * sizeof (elem_type);
            tmp2 -> APHost ().Region (0) = tmp2 -> APDevice ().Region (0) = line_length * sizeof (elem_type);
            tmp1 -> APHost ().Region (1) = tmp1 -> APDevice ().Region (1) = line_length;
            tmp2 -> APHost ().Region (1) = tmp2 -> APDevice ().Region (1) = line_length;
            tmp1 -> APHost ().Region (2) = tmp1 -> APDevice ().Region (2) = chunk_size_idwt12;
            tmp2 -> APHost ().Region (2) = tmp2 -> APDevice ().Region (2) = chunk_size_idwt12;
            tmp1 -> APDevice ().RowPitch () = tmp2 -> APDevice ().RowPitch () = line_length * sizeof (elem_type);
            tmp1 -> APDevice ().SlicePitch () = tmp2 -> APDevice ().SlicePitch () = line_length * line_length * sizeof (elem_type);
            
            for (int l = 0; l < line_length; l += chunk_size_idwt12)
            {
              
              tmp1 -> APHost ().Origin (2) = tmp2 -> APHost ().Origin (2) = l;
              
              tmp1 -> setCPUModified (); // upload src
              tmp2 -> setSync (); // do not upload data to result buffer
              
              pi2 += ocl_basic_operator_kernel_10 ("idwt_2_alt", tmp1, lpf, hpf, tmp2, loc_mem2, gpu_line_length, gpu_line_length, gpu_chunk_size_idwt12, gpu_line_length, gpu_chunk_size_idwt12, lc2);
              
              tmp1 -> setSync (); // do not upload src
              tmp2 -> setSync (); // do not upload data to dest
              
              pi1 += ocl_basic_operator_kernel_10 ("idwt_1_alt", tmp2, lpf, hpf, tmp1, loc_mem1, gpu_line_length, gpu_line_length, gpu_chunk_size_idwt12, gpu_line_length, gpu_chunk_size_idwt12, lc1);
              
              pi1.time_mem_down += tmp1 -> getData ();
              
            }
            vec_pi_1.push_back (pi1);
            vec_pi_2.push_back (pi2);
            
            tmp1 -> APHost ().Region (2) = tmp1 -> APDevice ().Region (2) = k; // reset
            tmp2 -> APHost ().Region (2) = tmp2 -> APDevice ().Region (2) = k; // reset
            tmp1 -> APHost ().Origin (2) = tmp2 -> APHost ().Origin (2) = 0; // reset
                        
          }

          delete loc_mem1;
          delete loc_mem2;
          delete loc_mem3;

          delete gpu_line_length;
          delete gpu_chunk_size_idwt3;
          delete gpu_chunk_size_idwt3_dim0;
          delete gpu_chunk_size_idwt12;
          
# ifdef __PERFORMANCE_INFO__
          return idwt_performance (vec_pi_1, vec_pi_2, vec_pi_3, {0,0,0,0}, lc1, lc2, lc3, levels, n, m, k, fl);
# else
          return std::vector <PerformanceInformation> ();
# endif
            
      }
      
      


      /**
       * @brief                       Matrix product.
       *
       * @param  arg1                 Address of first factor   (m x k matrix).
       * @param  arg2                 Address of second factor  (k x n matrix).
       * @param  prod                 Address of product        (m x n matrix).
       * @param  m                    First dimension of product.
       * @param  k                    Inner dimension.
       * @param  n                    Second dimension of product.
       * @param  trans1               1 -> Transpose first matrix,  2 -> Complex conjugate and transpose first matrix.
       * @param  trans2               1 -> Transpose second matrix, 2 -> Complex conjugate and transpose second matrix.
       */
      static inline
      const oclError &
      ocl_operator_matprod            ( oclDataObject * const     arg1,
                                        oclDataObject * const     arg2,
                                        oclDataObject * const     prod,
                                                  int                m,
                                                  int                k,
                                                  int                n,
                                                  int           transA,
                                                  int           transB )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_matprod", op_v_level);
        
        if (n == 1)
          ocl_basic_operator_amdblas_34 (amdblasGEMV, arg1, arg2, prod, m, k,    transA, transB);
        else
          ocl_basic_operator_amdblas_35 (amdblasGEMM, arg1, arg2, prod, m, n, k, transA, transB);
      
      }
    
    
      /**
       * @brief                       Elementwise addition of two vectors.
       *
       * @param  arg1                 Address of first ocl data object.
       * @param  arg2                 Address of second ocl data object.
       * @param  sum                  Address of resulting ocl data object.
       * @param  num_elems            Number of elements.
       */
      static inline
      const oclError &
      ocl_operator_add                ( oclDataObject * const      arg1,
                                        oclDataObject * const      arg2,
                                        oclDataObject * const       sum,
                                                  int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_add", op_v_level);
        ocl_basic_operator_kernel_3 ("vector_add", arg1, arg2, sum, num_elems);
        
      }           
        
    
      /**
       * @brief                       Elementwise subtraction of two vectors.
       *
       * @param  arg1                 Address of first ocl data object.
       * @param  arg2                 Address of second ocl data object.
       * @param  diff                 Address of resulting ocl data object.
       * @param  num_elems            Number of elements.
       */
      static inline
      const oclError &
      ocl_operator_subtract           (       oclDataObject * const      arg1,
                                              oclDataObject * const      arg2,
                                              oclDataObject * const      diff,
                                                        int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_subtract", op_v_level);
//        ocl_basic_operator_vclAlgo_3 (vclSUBTRACT, arg1, arg2, diff, num_elems);
        ocl_basic_operator_kernel_3 ("vector_sub", arg1, arg2, diff, num_elems);
      
      }
      
      
      /**
       * @brief                       Elementwise increment of vector.
       *
       * @param  arg1                 Address of vector's data object to be incremented.
       * @param  inc                  Scalar (increment).
       * @param  num_elems            Number of vector's elements.
       */
      static inline
      const oclError &
      ocl_operator_inc                ( oclDataObject * const      arg1,
                                          scalar_type               inc,
                                                  int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_inc", op_v_level);
        ocl_basic_operator_kernel_11 ("inc", arg1, inc, num_elems);
      
      }


      /**
       * @brief                       Elementwise decrement of vector.
       *
       * @param  arg1                 Address of vector's data object to be decremented.
       * @param  dec                  Scalar (decrement).
       * @param  num_elems            Number of vector's elements.
       */
      static inline
      const oclError &
      ocl_operator_dec                ( oclDataObject * const      arg1,
                                          scalar_type               dec,
                                                  int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_dec", op_v_level);
        
        // use matching type (for std::complex!) //
        typename trait2 :: value_type factor = -1;
        
        /* use increment kernel with inverse decrement */
        ocl_basic_operator_kernel_11 ("inc", arg1, factor * dec, num_elems);
      
      }


      /**
       * @brief                       Elementwise raise to higher power.
       *
       * @param  arg1                 Address of ocl data object.
       * @param  p                    Power.
       * @param  result               Address of resulting ocl data object.
       * @param  num_elems            Number of elements.
       */
      static inline
      const oclError &
      ocl_operator_pow                (       oclDataObject * const      arg1,
                                                  elem_type                 p,
                                              oclDataObject * const    result,
                                                        int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_pow", op_v_level);
        ocl_basic_operator_kernel_21 ("vector_pow", arg1, p, result, num_elems);
      
      }
      
      
      /**
       * @brief                       Elementwise multiplication with scalar.
       *
       * @param  arg1                 Address of ocl data object.
       * @param  scalar               Factor scalar.
       * @param  result               Address of resulting ocl data object.
       * @param  num_elems            Number of elements.
       */
      static inline
      const oclError &
      ocl_operator_mult_scalar        ( oclDataObject * const      arg1,
                                          scalar_type            scalar,
                                        oclDataObject * const    result,
                                                  int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_mult_scalar", op_v_level);
        ocl_basic_operator_kernel_21 ("scalar_mult", arg1, scalar, result, num_elems);
      
      }


      /**
       * @brief                       Elementwise multiplication with vector.
       *
       * @param  arg1                 Address of first vector's ocl data object.
       * @param  arg2                 Address of second vector's ocl data object.
       * @param  result               Address of resulting ocl data object.
       * @param  num_elems            Number of elements.
       */
      static inline
      const oclError &
      ocl_operator_mult_vector        ( oclDataObject * const      arg1,
                                        oclDataObject * const      arg2,
                                        oclDataObject * const    result,
                                                  int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_mult_vector", op_v_level);
        ocl_basic_operator_kernel_3 ("vector_mult", arg1, arg2, result, num_elems);
      
      }


      /**
       * @brief                       Elementwise division by scalar.
       *
       * @param  arg1                 Address of ocl data object.
       * @param  scalar               Divisor scalar.
       * @param  result               Address of resulting ocl data object.
       * @param  num_elems            Number of elements.
       */
      static inline
      const oclError &
      ocl_operator_div_scalar         ( oclDataObject * const      arg1,
                                          scalar_type            scalar,
                                        oclDataObject * const    result,
                                                  int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_div_scalar", op_v_level);
        ocl_basic_operator_kernel_21 ("scalar_div", arg1, scalar, result, num_elems);
      
      }


      /**
       * @brief                       Elementwise division by vector.
       *
       * @param  arg1                 Address of first vector's ocl data object.
       * @param  arg2                 Address of second vector's ocl data object.
       * @param  result               Address of resulting ocl data object.
       * @param  num_elems            Number of elements.
       */
      static inline
      const oclError &
      ocl_operator_div_vector         ( oclDataObject * const      arg1,
                                        oclDataObject * const      arg2,
                                        oclDataObject * const    result,
                                                  int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_div_vector", op_v_level);
        ocl_basic_operator_kernel_3 ("vector_div", arg1, arg2, result, num_elems);
      
      }
      
      
      /**
       * @brief                       Scalar equality.
       *
       * @param  arg1                 Address of vector's data object.
       * @param  scalar               Scalar.
       * @param  result               Address of result vector's data object.
       * @param  num_elems            Number of vectors' elements.
       */
      static inline
      const oclError &
      ocl_operator_equal              ( oclDataObject * const      arg1,
                                          scalar_type            scalar,
                                        oclDataObject * const    result,
                                                  int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_equal (scalar)", op_v_level);
        ocl_basic_operator_kernel_21 ("scalar_equal", arg1, scalar, result, num_elems);
      
      }
      
      
      /**
       * @brief                       Scalar inequality.
       *
       * @param  arg1                 Address of vector's data object.
       * @param  scalar               Scalar.
       * @param  result               Address of result vector's data object.
       * @param  num_elems            Number of vectors' elements.
       */
      static inline
      const oclError &
      ocl_operator_inequal              ( oclDataObject * const      arg1,
                                            scalar_type            scalar,
                                          oclDataObject * const    result,
                                                    int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_inequal (scalar)", op_v_level);
        ocl_basic_operator_kernel_21 ("scalar_inequal", arg1, scalar, result, num_elems);
      
      }
      
      
      /**
       * @brief                       Scalar greater comparison.
       *
       * @param  arg1                 Address of vector's data object.
       * @param  scalar               Scalar.
       * @param  result               Address of result vector's data object.
       * @param  num_elems            Number of vectors' elements.
       */
      static inline
      const oclError &
      ocl_operator_greater              ( oclDataObject * const      arg1,
                                            scalar_type            scalar,
                                          oclDataObject * const    result,
                                                    int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_greater (scalar)", op_v_level);
        ocl_basic_operator_kernel_21 ("scalar_greater", arg1, scalar, result, num_elems);
      
      }


      /**
       * @brief                       Scalar greater or equal comparison.
       *
       * @param  arg1                 Address of vector's data object.
       * @param  scalar               Scalar.
       * @param  result               Address of result vector's data object.
       * @param  num_elems            Number of vectors' elements.
       */
      static inline
      const oclError &
      ocl_operator_greater_equal              ( oclDataObject * const      arg1,
                                                  scalar_type            scalar,
                                                oclDataObject * const    result,
                                                          int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_greater_equal (scalar)", op_v_level);
        ocl_basic_operator_kernel_21 ("scalar_greater_equal", arg1, scalar, result, num_elems);
      
      }


      /**
       * @brief                       Scalar less comparison.
       *
       * @param  arg1                 Address of vector's data object.
       * @param  scalar               Scalar.
       * @param  result               Address of result vector's data object.
       * @param  num_elems            Number of vectors' elements.
       */
      static inline
      const oclError &
      ocl_operator_less              ( oclDataObject * const      arg1,
                                         scalar_type            scalar,
                                       oclDataObject * const    result,
                                                 int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_less (scalar)", op_v_level);
        ocl_basic_operator_kernel_21 ("scalar_less", arg1, scalar, result, num_elems);
      
      }


      /**
       * @brief                       Scalar less or equal comparison.
       *
       * @param  arg1                 Address of vector's data object.
       * @param  scalar               Scalar.
       * @param  result               Address of result vector's data object.
       * @param  num_elems            Number of vectors' elements.
       */
      static inline
      const oclError &
      ocl_operator_less_equal              ( oclDataObject * const      arg1,
                                               scalar_type            scalar,
                                             oclDataObject * const    result,
                                                       int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_less_equal (scalar)", op_v_level);
        ocl_basic_operator_kernel_21 ("scalar_less_equal", arg1, scalar, result, num_elems);
      
      }


      /**
       * @brief                       Elementwise equality of vectors.
       *
       * @param  arg1                 Address of first vector's data object.
       * @param  arg2                 Address of second vector's data object.
       * @param  result               Address of result vector's data object.
       * @param  num_elems            Number of vectors' elements.
       */
      static inline
      const oclError &
      ocl_operator_equal              ( oclDataObject * const      arg1,
                                        oclDataObject * const      arg2,
                                        oclDataObject * const    result,
                                                  int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_equal (vector)", op_v_level);
        ocl_basic_operator_kernel_3 ("vector_equal", arg1, arg2, result, num_elems);
      
      }
      
      
      /**
       * @brief                       Elementwise inequality of vectors.
       *
       * @param  arg1                 Address of first vector's data object.
       * @param  arg2                 Address of second vector's data object.
       * @param  result               Address of result vector's data object.
       * @param  num_elems            Number of vectors' elements.
       */
      static inline
      const oclError &
      ocl_operator_inequal              ( oclDataObject * const      arg1,
                                          oclDataObject * const      arg2,
                                          oclDataObject * const    result,
                                                    int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_inequal (vector)", op_v_level);
        ocl_basic_operator_kernel_3 ("vector_inequal", arg1, arg2, result, num_elems);
      
      }


      /**
       * @brief                       Elementwise greater comparison of vectors.
       *
       * @param  arg1                 Address of first vector's data object.
       * @param  arg2                 Address of second vector's data object.
       * @param  result               Address of result vector's data object.
       * @param  num_elems            Number of vectors' elements.
       */
      static inline
      const oclError &
      ocl_operator_greater              ( oclDataObject * const      arg1,
                                          oclDataObject * const      arg2,
                                          oclDataObject * const    result,
                                                    int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_greater (vector)", op_v_level);
        ocl_basic_operator_kernel_3 ("vector_greater", arg1, arg2, result, num_elems);
      
      }


      /**
       * @brief                       Elementwise greater or equal comparison of vectors.
       *
       * @param  arg1                 Address of first vector's data object.
       * @param  arg2                 Address of second vector's data object.
       * @param  result               Address of result vector's data object.
       * @param  num_elems            Number of vectors' elements.
       */
      static inline
      const oclError &
      ocl_operator_greater_equal        ( oclDataObject * const      arg1,
                                          oclDataObject * const      arg2,
                                          oclDataObject * const    result,
                                                    int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_greater_equal (vector)", op_v_level);
        ocl_basic_operator_kernel_3 ("vector_greater_equal", arg1, arg2, result, num_elems);
      
      }


      /**
       * @brief                       Elementwise less comparison of vectors.
       *
       * @param  arg1                 Address of first vector's data object.
       * @param  arg2                 Address of second vector's data object.
       * @param  result               Address of result vector's data object.
       * @param  num_elems            Number of vectors' elements.
       */
      static inline
      const oclError &
      ocl_operator_less                 ( oclDataObject * const      arg1,
                                          oclDataObject * const      arg2,
                                          oclDataObject * const    result,
                                                    int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_less (vector)", op_v_level);
        ocl_basic_operator_kernel_3 ("vector_less", arg1, arg2, result, num_elems);
      
      }


      /**
       * @brief                       Elementwise less or equal comparison of vectors.
       *
       * @param  arg1                 Address of first vector's data object.
       * @param  arg2                 Address of second vector's data object.
       * @param  result               Address of result vector's data object.
       * @param  num_elems            Number of vectors' elements.
       */
      static inline
      const oclError &
      ocl_operator_less_equal         ( oclDataObject * const      arg1,
                                        oclDataObject * const      arg2,
                                        oclDataObject * const    result,
                                                  int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_less_equal (vector)", op_v_level);
        ocl_basic_operator_kernel_3 ("vector_less_equal", arg1, arg2, result, num_elems);
      
      }


      /**
       * @brief                       Bitwise AND operation (mask).
       *
       * @param  arg1                 Vector to be masked.
       * @param  mask                 Masking vector.
       * @param  result               Cross-Section or zero.
       * @param  num_elems            Number of vectors' elements.
       */
      static inline
      const oclError &
      ocl_operator_bitw_and           ( oclDataObject * const      arg1,
                                        oclDataObject * const      mask,
                                        oclDataObject * const    result,
                                                  int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_bitw_and", op_v_level);
        ocl_basic_operator_kernel_3 ("bitw_and", arg1, mask, result, num_elems);
      
      }
      
      
      /**
       * @brief                       Elementwise AND operation of vectors.
       *
       * @param  arg1                 Address of first vector's data object.
       * @param  arg2                 Address of second vector's data object.
       * @param  result               Address of result vector's data object.
       * @param  num_elems            Number of vectors' elements.
       */
      static inline
      const oclError &
      ocl_operator_and                 ( oclDataObject * const     arg1,
                                         oclDataObject * const     arg2,
                                         oclDataObject * const   result,
                                                  int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_and", op_v_level);
        ocl_basic_operator_kernel_3 ("vector_and", arg1, arg2, result, num_elems);
      
      }


      /**
       * @brief                       Elementwise AND operation of vectors.
       *
       * @param  arg1                 Address of first vector's data object.
       * @param  arg2                 Address of second vector's data object.
       * @param  result               Address of result vector's data object.
       * @param  num_elems            Number of vectors' elements.
       */
      static inline
      const oclError &
      ocl_operator_or                 ( oclDataObject * const     arg1,
                                        oclDataObject * const     arg2,
                                        oclDataObject * const   result,
                                                 int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_or", op_v_level);
        ocl_basic_operator_kernel_3 ("vector_or", arg1, arg2, result, num_elems);
      
      }      

      
      /**
       * @brief                       Elementwise assignment of scalar to vector.
       *
       * @param  arg1                 Address of vector's data object.
       * @param  scalar               Scalar (to be assigned).
       * @param  num_elems            Number of vector' elements.
       */
      static inline
      const oclError &
      ocl_operator_assign             ( oclDataObject * const      arg1,
                                          scalar_type            scalar,
                                                  int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_assign", op_v_level);
        ocl_basic_operator_kernel_11 ("assign", arg1, scalar, num_elems);
      
      } 
          
    
      //@}
      
      
      /**
       * @brief                       Deep copy of arg1 to arg2.
       *
       * @param  dest                 Address of destination.
       * @param  src                  Address of source.
       * @param  num_elems            Number of elements.
       */
      static inline
      const oclError &
      ocl_operator_copy               ( oclDataObject * const      dest,
                                        oclDataObject * const       src,
                                                  int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_copy", op_v_level);
        ocl_basic_operator_kernel_2 ("copy_buffer", dest, src, num_elems);
      
      }
      
      
      
      /**
       * @brief                       Cast elements of arg1 and store them in arg2.
       *
       * @param  arg1                 Vector to be casted.
       * @param  arg2                 Vector containing casted elements of arg1.
       * @param  num_elems            Number of elements.
       */
      static inline
      const oclError &
      ocl_operator_cast               ( oclDataObject * const      arg1,
                                        oclDataObject * const      arg2,
                                                  int         num_elems )
      {
      
        print_optional ("oclOperations <", trait1 :: print_elem_type (), ", ",
                                           trait2 :: print_elem_type (), "> :: ocl_operator_cast", op_v_level);
        ocl_basic_operator_kernel_2 ("cast", arg1, arg2, num_elems);
      
      }
    
    
    
  };
  
  
  
  
# endif // __OCL_TRAITS_HPP__
