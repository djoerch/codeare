/*
 *  codeare Copyright (C) 2007-2010 Kaveh Vahedipour
 *                                  Forschungszentrum Juelich, Germany
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

#ifndef __FT_HPP__
#define __FT_HPP__

#include "CX.hpp"

/**
 * @brief    General parameters class for Fourier transform constructors
 */
template<class T>
struct FTParams {

	Matrix<T> b0;                 /**< @brief b0 map */
	Matrix< std::complex<T> > pc; /**< @brief phase correction */
	Matrix<T> b1; /**< @brief b1 maps */
	
	Matrix<size_t> sl; /**<@brief side length */
  Matrix<size_t> zpad;     /**<@brief Zero-pad k-space */
	Matrix<T> mask;    /**<@brief k-space mask */

	size_t rank;  /**< @brief single side length applies to all dimensions */

	size_t nk;   /**< @brief # of k-space points*/

};

template <class T>
struct FTCGParams {

	size_t iters;  /**< @brief # of iterations */
	T      lambda; /**< @brief tikhonov weight */
	T      eps;    /**< @brief convergence residual */

};


/**
 * @brief  Base class for single and double precision Fourier transforms
 */
template <class T>
class FT {

public:

	/**
	 * @brief    Default constructor
	 */
	FT () {
		T t;
		Validate (t);
	};


	/**
	 * @brief     Contstruct with parameters
	 */
	FT (FTParams<T> ftp, FTCGParams<T> ftcgp);

	/**
	 * @brief    Default destructor
	 */
	virtual ~FT() {};

	/**
	 * @brief    Forward transform
	 *
	 * @param  m To transform
	 * @return   Transform
	 */
	virtual Matrix< std::complex<T> >
	Trafo       (const Matrix< std::complex<T> >& m) const = 0;
	
	
	/**
	 * @brief    Backward transform
	 *
	 * @param  m To transform
	 * @return   Transform
	 */
	virtual Matrix< std::complex<T> >
	Adjoint     (const Matrix< std::complex<T> >& m) const = 0;
	
	
	/**
	 * @brief    Forward transform
	 *
	 * @param  m To transform
	 * @return   Transform
	 */
	Matrix< std::complex<T> >
	operator*   (const Matrix< std::complex<T> >& m) const {
		return Trafo(m);
	}
	

	/**
	 * @brief    Backward transform
	 *
	 * @param  m To transform
	 * @return   Transform
	 */
	Matrix< std::complex<T> >
	operator->* (const Matrix< std::complex<T> >& m) const {
		return Adjoint (m);
	}

	
protected:

    void Validate (double& t) const {};
	void Validate (float&  t) const {};
	
};

#endif
