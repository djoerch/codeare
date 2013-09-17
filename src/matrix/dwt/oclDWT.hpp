/*
 *  codeare Copyright (C) 2013 Daniel Joergens
 *                             Forschungszentrum Juelich, Germany
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 *  02110-1301  USA
 *
 */


# ifndef __OCL_DWT_HPP__

# define __OCL_DWT_HPP__

/**
 * @brief OMP related makros
 */
# define NUM_THREADS_DWT 4
# define OMP_SCHEDULE guided

/**
 * @brief Default wavelet parameters
 */
# define WL_FAM WL_DAUBECHIES
# define WL_MEM 4
# define WL_SCALE 4

/**
 * @brief OpenCL related makros
 */
# define OPENCL_GROUP_SIZE 16
# define OPENCL_GLOBAL_SIZE 256



# include "Matrix.hpp"
# include "Wavelet.hpp"

# include "ocl/oclConnection.hpp"
# include "ocl/oclDataWrapper.hpp"
# include "ocl/oclTraits.hpp"


// OpenCL
# include <CL/cl.hpp>


/**
 * @brief   Discrete wavelet transform (periodic boundaries) for 2d and 3D case for Matrix template.
 */
template<class T>
class oclDWT {


    private:

      inline void
      setupOpenCL           (const int fl,
                             const int side_length)
      {
        std::stringstream ss;
            std::vector <std::string> makros;
            makros.push_back ((ss << "GROUP_SIZE_0 " << _group_size_x, ss.str ())); ss.str ("");
            makros.push_back ((ss << "GROUP_SIZE_1 " << _group_size_y, ss.str ())); ss.str ("");
            makros.push_back ((ss << "NUM_GROUPS_0 " << (_global_size_x/_group_size_x), ss.str ())); ss.str ("");
            makros.push_back ((ss << "NUM_GROUPS_1 " << (_global_size_y/_group_size_y), ss.str ())); ss.str ("");
            makros.push_back ((ss << "FL " << fl, ss.str ())); ss.str ("");
            makros.push_back ((ss << "LDA " << side_length, ss.str ())); ss.str ("");
            std::vector <std::string> filenames;
            filenames.push_back (base_kernel_path + "src/matrix/dwt/dwt2.cl");
//            filenames.push_back (base_kernel_path + "src/matrix/dwt/dwt.cl");
            oclOperations <T> :: addKernelSources (filenames, makros);
      }
      
  
    public:


        /**
         * @brief Construct oclDWT object for images of given side lengths and column major memory scheme.
         *
         * @param  sl1          Side length along first dimension.
         * @param  sl2          Side length along second dimension.
         * @param  sl3          Side length along third dimension.
         * @param  wl_fam       Wavelet family.
         * @param  wl_mem       Member of wavelet family.
         * @param  wl_scale     Decomposition until side length equals 2^wl_scale.
         */
        oclDWT (const size_t sl1, const size_t sl2, const size_t sl3,
             const wlfamily wl_fam = WL_FAM, const int wl_mem = WL_MEM, const int wl_scale = WL_SCALE, const int group_size_x = OPENCL_GROUP_SIZE, const int group_size_y = OPENCL_GROUP_SIZE, const int global_size_x = OPENCL_GLOBAL_SIZE, const int global_size_y = OPENCL_GLOBAL_SIZE)
            : _sl1 (sl1),
              _sl2 (sl2),
              _sl3 (sl3),
              _dim (_sl3 == 1 ? 2 : 3),
              _min_sl (_dim == 2 ? MIN (_sl1, _sl2) : MIN (MIN (_sl1, _sl2),_sl3)),
              _min_level (wl_scale),
              _max_level (MaxLevel ()),
              _wl_fam(wl_fam),
              _fl (wl_mem),
              _group_size_x (group_size_x),
              _group_size_y (group_size_y),
          _global_size_x (global_size_x),
          _global_size_y (global_size_y)
        {
            setupWlFilters <T> (wl_fam, wl_mem, _lpf_d, _lpf_r, _hpf_d, _hpf_r);
            setupOpenCL (wl_mem, _min_sl);
        }


        /**
         * @brief       Construct 2D oclDWT object.
         *
         * @param  sl1          Side length along first dimension.
         * @param  sl2          Side length along second dimension.
         * @param  wl_fam       Wavelet family.
         * @param  wl_mem       Member of wavelet family.
         * @param  wl_scale     Decomposition until side length equals 2^wl_scale.
         */
        oclDWT (const size_t sl1, const size_t sl2,
             const wlfamily wl_fam = WL_FAM, const int wl_mem = WL_MEM, const int wl_scale = WL_SCALE,
             const int num_threads = NUM_THREADS_DWT, const int group_size_x = OPENCL_GROUP_SIZE, const int group_size_y = OPENCL_GROUP_SIZE, const int global_size_x = OPENCL_GLOBAL_SIZE, const int global_size_y = OPENCL_GLOBAL_SIZE)
        : _sl1 (sl1),
          _sl2 (sl2),
          _sl3 (1),
          _dim (_sl3 == 1 ? 2 : 3),
          _min_sl (_dim == 2 ? MIN (_sl1, _sl2) : MIN (MIN (_sl1, _sl2),_sl3)),
          _min_level (wl_scale),
          _max_level (MaxLevel ()),
          _wl_fam(wl_fam),
          _fl (wl_mem),
          _group_size_x (group_size_x),
              _group_size_y (group_size_y),
          _global_size_x (global_size_x),
          _global_size_y (global_size_y)
        {
            setupWlFilters <T> (wl_fam, wl_mem, _lpf_d, _lpf_r, _hpf_d, _hpf_r);
            setupOpenCL (wl_mem, _min_sl);
        }


        /**
         * @brief       Construct 2D oclDWT object for square matrices.
         *
         * @param  sl1          Side length along first dimension.
         * @param  wl_fam       Wavelet family.
         * @param  wl_mem       Member of wavelet family.
         * @param  wl_scale     Decomposition until side length equals 2^wl_scale.
         */
        oclDWT (const size_t sl1,
             const wlfamily wl_fam = WL_FAM, const int wl_mem = WL_MEM, const int wl_scale = WL_SCALE, const int group_size_x = OPENCL_GROUP_SIZE, const int group_size_y = OPENCL_GROUP_SIZE, const int global_size_x = OPENCL_GLOBAL_SIZE, const int global_size_y = OPENCL_GLOBAL_SIZE)
        : _sl1 (sl1),
          _sl2 (_sl1),
          _sl3 (1),
          _dim (_sl3 == 1 ? 2 : 3),
          _min_sl (_dim == 2 ? MIN (_sl1, _sl2) : MIN (MIN (_sl1, _sl2),_sl3)),
          _min_level (wl_scale),
          _max_level (MaxLevel ()),
          _wl_fam(wl_fam),
          _fl (wl_mem),
          _group_size_x (group_size_x),
              _group_size_y (group_size_y),
          _global_size_x (global_size_x),
          _global_size_y (global_size_y)
        {
            setupWlFilters <T> (wl_fam, wl_mem, _lpf_d, _lpf_r, _hpf_d, _hpf_r);
            setupOpenCL (wl_mem, _min_sl);
        }


        virtual
        ~oclDWT ()
        { }


        /**
         * @brief    Forward transform (no constructor calls)
         *
         * @param  m    Signal to decompose
         * @param  res  Resulting DWT
         */
        inline
//        void
        std::vector <PerformanceInformation>
        Trafo        ( Matrix <T> & m, Matrix <T> & res)
        {

            assert (   m.Dim (0) == _sl1
                    && m.Dim (1) == _sl2
                    && (_dim == 2 || m.Dim (2) == _sl3)
                    && m.Dim () == res.Dim ());
                        
            /* TODO: call kernel */
            oclDataWrapper <T> * p_ocl_m   = oclOperations <T> :: make_GPU_Obj (&m.Container()[0], m.Size ());
            oclDataWrapper <T> * p_ocl_res = oclOperations <T> :: make_GPU_Obj (&res[0], res.Size ());
            oclDataWrapper <T> * p_ocl_lpf = oclOperations <T> :: make_GPU_Obj (_lpf_d, _fl);
            oclDataWrapper <T> * p_ocl_hpf = oclOperations <T> :: make_GPU_Obj (_hpf_d, _fl);
            
            const int num_loc_mem_size = (m.Dim (0) / (_global_size_x/_group_size_x) + _fl) * (m.Dim (0) / (_global_size_x/_group_size_y) + _fl) + (m.Dim (0) / (_global_size_x/_group_size_x)) * (m.Dim (0) / (_global_size_x/_group_size_y) + _fl);
            
            std::vector <PerformanceInformation> vec_perf = oclOperations <T> :: ocl_operator_dwt (p_ocl_m, m.Dim(0), m.Dim(1), m.Dim(2),
                                                   p_ocl_lpf, p_ocl_hpf, _fl,
                                                   p_ocl_res, num_loc_mem_size,
                                                   _group_size_x,
                                                   _group_size_y,
                                                   _global_size_x,
                                                   _global_size_y);
            double time = p_ocl_res->getData();
            p_ocl_m->getData();
            
            std::vector <PerformanceInformation> vec_perf2 = oclOperations <T> :: ocl_operator_perf_dwt (p_ocl_m, p_ocl_lpf, p_ocl_hpf, p_ocl_res, m.Dim(0), num_loc_mem_size, _fl,
                                                   _group_size_x,
                                                   _group_size_y,
                                                   _global_size_x,
                                                   _global_size_y);
                        
            delete p_ocl_m;
            delete p_ocl_res;
            delete p_ocl_lpf;
            delete p_ocl_hpf;
            
            vec_perf [0].time_mem_down += time;
            vec_perf2.push_back (vec_perf [0]);
            
            return vec_perf2;
            
        }


        /**
         * @brief    Adjoint transform (no constructor calls)
         *
         * @param  m    DWT to transform
         * @param  res  Reconstructed signal
         */
        inline
        void
        Adjoint      (const Matrix <T> & m, Matrix <T> & res)
        {

            assert (   m.Dim (0) == _sl1
                    && m.Dim (1) == _sl2
                    && (_dim == 2 || m.Dim (2) == _sl3)
                    && m.Dim () == res.Dim ());

            /* TODO: call kernel */
            res = m;

        }


        /**
         * @brief    Forward transform
         *
         * @param  m To transform
         * @return   Transform
         */
        inline
        Matrix <T>
        operator*    (const Matrix <T> & m) {

            if (_wl_fam == ID)
                return m;
            else
            {
                Matrix <T> res (m);
                Trafo (m, res);
                return res;
            }

        }


        /**
         * @brief    Adjoint transform
         *
         * @param  m To transform
         * @return   Transform
         */
        inline
        Matrix <T>
        operator->* (const Matrix <T> & m) {

            if (_wl_fam == ID)
                return m;
            else
            {
                Matrix <T> res (m);
                Adjoint (m, res);
                return res;
            }

        }


    private:


        /**
         * type definitions
         */
        typedef typename TypeTraits <T> :: RT RT;


        /**
         * variable definitions
         */

        // size of valid matrices
        const size_t _sl1;      // side length in first dimension  ('x')
        const size_t _sl2;      // side length in second dimension ('y')
        const size_t _sl3;      // side length in third dimension  ('z')

        // dimension of DWT
        const int _dim;

        const size_t _min_sl;   // minimum side length

        // wavelet scales => (_max_level - _min_level) decompositions
        const int _min_level;   // min. decomposition level
        const int _max_level;   // max. decomposition level

        // wavelet family
        const wlfamily _wl_fam;

        const int _fl;
        
        const int _group_size_x;
        const int _group_size_y;
        const int _global_size_x;
        const int _global_size_y;

        // low pass filters
        RT * _lpf_d;
        RT * _lpf_r;

        // high pass filters
        RT * _hpf_d;
        RT * _hpf_r;


        /**
         * function definitions
         */


        /**
         * @brief           Calculate start level for decomposition.
         *                  (Depends on minimum of side lengths.)
         *
         * @return          Start level.
         */
        int
        MaxLevel            ()
        {
            // create vars from mex function
            size_t nn = 1, max_level = 0;
            for (; nn < _min_sl; nn *= 2 )
                max_level ++;
            if (nn  !=  _min_sl){
                std::cout << "FWT2 requires dyadic length sides" << std::endl;
                assert (false);
            }
            return max_level;
        }

};

# endif // __OCL_DWT_HPP__
