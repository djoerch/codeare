/*
 *  codeare Copyright (C) 2007-2010 Kaveh Vahedipour
 *                               Forschungszentrum Juelich, Germany
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
 */


# ifndef __DWT2_HPP__

# define __DWT2_HPP__


/**
 * @brief  Supported wavelet families
 */
enum wlfamily {
	
	ID = -1,                  /**< Identity transform*/
	WL_DAUBECHIES,
	WL_DAUBECHIES_CENTERED,
	WL_HAAR,
	WL_HAAR_CENTERED,
	WL_BSPLINE,
	WL_BSPLINE_CENTERED

};


/********************
 ** matrix headers **
 ********************/
# include "Matrix.hpp"

/****************
 ** DWT traits **
 ****************/
# include "DWT2Traits.hpp"



/**
 * @brief 2D Discrete wavelet transform for Matrix template (from GSL)
 */
template <class T>
class DWT2 {


	typedef typename DWT2Traits<T>::Type Type;


public:


	/**
	 * @brief Construct 2D Wavelet transform with wavelet class and side length
	 *
	 * @param  sl      Side length
	 * @param  wf      Wavelet family (default none, i.e. ID)
	 * @param  wm      Familty member (default 4)
	 */
	DWT2 ();
	
	
	virtual 
	~DWT2();


	/**
	 * @brief    Forward transform
	 *
	 * @param  m To transform
	 * @return   Transform
	 */
    Matrix<T>
	Trafo        (const Matrix<T>& m) const {


	}
	

	/**
	 * @brief    Adjoint transform
	 *
	 * @param  m To transform
	 * @return   Transform
	 */
	Matrix<T>
	Adjoint      (const Matrix<T>& m) const {


	}
	

	/**
	 * @brief    Forward transform
	 *
	 * @param  m To transform
	 * @return   Transform
	 */
    Matrix<T>
	operator*    (const Matrix<T>& m) const {

		return Trafo(m);

	}
	

	/**
	 * @brief    Adjoint transform
	 *
	 * @param  m To transform
	 * @return   Transform
	 */
    Matrix<T>
	operator->* (const Matrix<T>& m) const {

		return Adjoint(m);

	}
	
	
private:
	
	/**
	 * @brief       Transform forward.
	 *
	 * @param   m   To transform
	 * @return      Wavelet transform.
	 */
	Matrix <T>
	transForward    (const Matrix <T> & m) const ;


	/**
	 * @brief       Transform backwards.
	 *
	 * @param  m    Wavelet transform.
	 * @return      Original signal.
	 */
	Matrix <T>
	transBackward   (const Matrix <T> & m) const;
	
	Wavelet m_wl;                   // instance of currently used wavelet
	
};



/*****************
 ** definitions **
 *****************/


template <class T>
DWT2<T>::DWT2 () {

	// Checks missing !!!



}


template <class T>
DWT2<T>::~DWT2 () {



}


template <class T>
Matrix <T>
DWT2<T>::transForward    (const Matrix <T> & m)
const
{
	

	
}



# endif // __DWT2_HPP__
