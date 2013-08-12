#ifndef __CREATORS_HPP__
#define __CREATORS_HPP__

#include "Matrix.hpp"
#include "Algos.hpp"
#if !defined(_MSC_VER) || _MSC_VER>1200
#  include "RandTraits.hpp"
#endif



/**
 * @brief       Zero matrix
 *
 * @param  col  Column
 * @param  lin  Rows
 * @param  cha  Dimension
 * @param  set  Dimension
 * @param  eco  Dimension
 * @param  phs  Dimension
 * @param  rep  Dimension
 * @param  seg  Dimension
 * @param  par  Dimension
 * @param  slc  Dimension
 * @param  ida  Dimension
 * @param  idb  Dimension
 * @param  idc  Dimension
 * @param  idd  Dimension
 * @param  ide  Dimension
 * @param  ave  Dimension
 *
 * @return      Zero matrix
 *
 */
template <class T> inline static Matrix<T> 
zeros           (const size_t& col, 
				 const size_t& lin,
				 const size_t& cha = 1,
				 const size_t& set = 1,
				 const size_t& eco = 1,
				 const size_t& phs = 1,
				 const size_t& rep = 1,
				 const size_t& seg = 1,
				 const size_t& par = 1,
				 const size_t& slc = 1,
				 const size_t& ida = 1,
				 const size_t& idb = 1,
				 const size_t& idc = 1,
				 const size_t& idd = 1,
				 const size_t& ide = 1,
				 const size_t& ave = 1) {

 	return Matrix<T> (col, lin, cha, set, eco, phs, rep, seg, par, slc, ida, idb, idc, idd, ide, ave);

}


/**
 * @brief       Zero matrix
 *
 * @param  sz   Size vector
 * @return      Zero matrix
 *
 */
template <class T> inline static Matrix<T> 
zeros           (const Matrix<size_t>& sz) {

 	return Matrix<T> (sz.Container());

}

/**
 * @brief       Square matrix of zeros
 *
 * @param  n    Side length
 * @return      Zero matrix
 */
template <class T> inline static Matrix<T>
zeros            (const size_t& n) {
	return zeros<T>(n,n);
}



/**
 * @brief      Ones matrix
 *
 * @param  col  Column
 * @param  lin  Rows
 * @param  cha  Dimension
 * @param  set  Dimension
 * @param  eco  Dimension
 * @param  phs  Dimension
 * @param  rep  Dimension
 * @param  seg  Dimension
 * @param  par  Dimension
 * @param  slc  Dimension
 * @param  ida  Dimension
 * @param  idb  Dimension
 * @param  idc  Dimension
 * @param  idd  Dimension
 * @param  ide  Dimension
 * @param  ave  Dimension
 *
 * @return      Ones matrix
 *
 */
template <class T> inline static Matrix<T> 
ones            (const size_t& col, 
				 const size_t& lin,
				 const size_t& cha = 1,
				 const size_t& set = 1,
				 const size_t& eco = 1,
				 const size_t& phs = 1,
				 const size_t& rep = 1,
				 const size_t& seg = 1,
				 const size_t& par = 1,
				 const size_t& slc = 1,
				 const size_t& ida = 1,
				 const size_t& idb = 1,
				 const size_t& idc = 1,
				 const size_t& idd = 1,
				 const size_t& ide = 1,
				 const size_t& ave = 1) {

 	 Matrix<T> res (col, lin, cha, set, eco, phs, rep, seg, par, slc, ida, idb, idc, idd, ide, ave);
     std::fill (res.Begin(), res.End(), T(1));

	 return res;

}


/**
 * @brief       Square matrix of ones
 *
 * @param  n    Side length
 * @return      Random matrix
 */
template <class T> inline static Matrix<T>
ones            (const size_t& n) {
	return ones<T>(n,n);
}


/**
 * @brief       Zero matrix
 *
 * @param  sz   Size vector
 * @return      Zero matrix
 *
 */
template <class T> inline static Matrix<T>
ones           (const Matrix<size_t>& sz) {
 	return Matrix<T> (sz.Container()) = (T)1;
}


#if !defined(_MSC_VER) || _MSC_VER>1200
/**
 * @brief       Uniformly random matrix
 *
 * @param  col  Column
 * @param  lin  Rows
 * @param  cha  Dimension
 * @param  set  Dimension
 * @param  eco  Dimension
 * @param  phs  Dimension
 * @param  rep  Dimension
 * @param  seg  Dimension
 * @param  par  Dimension
 * @param  slc  Dimension
 * @param  ida  Dimension
 * @param  idb  Dimension
 * @param  idc  Dimension
 * @param  idd  Dimension
 * @param  ide  Dimension
 * @param  ave  Dimension
 *
 * @return      Random matrix
 *
 */
template<class T> static Matrix<T>
rand           (const size_t& col, 
				const size_t& lin,
				const size_t& cha = 1,
				const size_t& set = 1,
				const size_t& eco = 1,
				const size_t& phs = 1,
				const size_t& rep = 1,
				const size_t& seg = 1,
				const size_t& par = 1,
				const size_t& slc = 1,
				const size_t& ida = 1,
				const size_t& idb = 1,
				const size_t& idc = 1,
				const size_t& idd = 1,
				const size_t& ide = 1,
				const size_t& ave = 1) {
	
	Matrix<T> res (col, lin, cha, set, eco, phs, rep, seg, par, slc, ida, idb, idc, idd, ide, ave);
    Random<T>::Uniform(res);
	return res;

}


/**
 * @brief       Uniformly random matrix
 *
 * @param  sz   Size vector
 * @return      Rand matrix
 *
 */
template <class T> inline static Matrix<T>
rand           (const Matrix<size_t>& sz) {

	Matrix<T> res (sz.Container());
    Random<T>::Uniform(res);
 	return res;

}

/**
 * @brief       Random square matrix
 *
 * @param  n    Side length
 * @return      Random matrix
 */
template<class T> static Matrix<T>
rand (const size_t n) {
	return rand<T>(n,n);
}


/**
 * @brief       Uniformly random matrix
 *
 * @param  col  Column
 * @param  lin  Rows
 * @param  cha  Dimension
 * @param  set  Dimension
 * @param  eco  Dimension
 * @param  phs  Dimension
 * @param  rep  Dimension
 * @param  seg  Dimension
 * @param  par  Dimension
 * @param  slc  Dimension
 * @param  ida  Dimension
 * @param  idb  Dimension
 * @param  idc  Dimension
 * @param  idd  Dimension
 * @param  ide  Dimension
 * @param  ave  Dimension
 *
 * @return      Random matrix
 *
 */
template<class T> static Matrix<T>
randn          (const size_t& col, 
				const size_t& lin,
				const size_t& cha = 1,
				const size_t& set = 1,
				const size_t& eco = 1,
				const size_t& phs = 1,
				const size_t& rep = 1,
				const size_t& seg = 1,
				const size_t& par = 1,
				const size_t& slc = 1,
				const size_t& ida = 1,
				const size_t& idb = 1,
				const size_t& idc = 1,
				const size_t& idd = 1,
				const size_t& ide = 1,
				const size_t& ave = 1) {
	
	Matrix<T> res (col, lin, cha, set, eco, phs, rep, seg, par, slc, ida, idb, idc, idd, ide, ave);
    Random<T>::Normal(res);
	return res;

}


/**
 * @brief       Uniformly random matrix
 *
 * @param  sz   Size vector
 * @return      Rand matrix
 *
 */
template <class T> inline static Matrix<T>
randn          (const Matrix<size_t>& sz) {

	Matrix<T> res (sz.Container());
    Random<T>::Normal(res);
 	return res;

}

/**
 * @brief       Random square matrix
 *
 * @param  n    Side length
 * @return      Random matrix
 */
template<class T> static Matrix<T>
randn (const size_t n) {
	return rand<T>(n,n);
}
#endif

/**
 * @brief       nxn square matrix with circle centered at p
 *
 * @param  p    Center point of circle
 * @param  n    Side length of square
 * @param  s    Scaling factor
 * @return      Matrix with circle
 */
template <class T> inline static Matrix<T>
circle (const float* p, const size_t n, const T s = T(1)) {

	Matrix<T> res(n);

	float m[2];
	float rad;

	rad = p[0] * float(n) / 2.0;

	m[0] = (1.0 - p[1]) * float(n) / 2.0;
	m[1] = (1.0 - p[2]) * float(n) / 2.0;

	for (size_t r = 0; r < res.Dim(1); r++)
		for (size_t c = 0; c < res.Dim(0); c++)
			res(c,r) = ( pow(((float)c-m[0])/rad, (float)2.0 ) + pow(((float)r-m[0])/rad, (float)2.0) <= 1.0) ? s : T(0.0);

	return res;

}



/**
 * @brief       nxnxn cube with sphere centered at p
 *
 * @param  p    Center point of sphere
 * @param  n    Side length of cube
 * @param  s    Scaling factor
 * @return      Matrix with circle
 */
template <class T> inline static Matrix<T>
sphere (const float* p, const size_t n, const T s = T(1)) {

	Matrix<T> res (n,n,n);

	float m[3];
	float rad;

	rad = p[0] * float(n) / 2.0;

	m[0] = (1.0 - p[1]) * float(n) / 2.0;
	m[1] = (1.0 - p[2]) * float(n) / 2.0;
	m[2] = (1.0 - p[3]) * float(n) / 2.0;

	for (size_t s = 0; s < res.Dim(2); s++)
		for (size_t r = 0; r < res.Dim(1); r++)
			for (size_t c = 0; c < res.Dim(0); c++)
				res(c,r) = ( pow (((float)c-m[0])/rad, (float)2.0) + pow (((float)r-m[1])/rad, (float)2.0) + pow (((float)s-m[2])/rad, (float)2.0) <= 1.0) ? s : T(0.0);

	return res;

}



/**
 * @brief       nxn square matrix with circle centered at p
 *
 * @param  p    Center point of ellipse and excentricities
 * @param  n    Side length of square
 * @param  s    Scaling 
 * @return      Matrix with circle
 */
template <class T> inline static Matrix<T>
ellipse (const float* p, const size_t n, const T s = T(1)) {

	Matrix<T> res (n);

	float m[2];
	float a[2];

	a[0] = p[0] * float(n) / 2.0;
	a[1] = p[1] * float(n) / 2.0;

	m[0] = (1.0 - p[2]) * float(n) / 2.0;
	m[1] = (1.0 - p[3]) * float(n) / 2.0;

	float cosp = cos(p[4]);
	float sinp = sin(p[4]);
	
#pragma omp parallel default (shared) 
	{
#pragma omp for schedule (dynamic, n / omp_get_num_threads())
		
	for (size_t r = 0; r < n; r++)
		for (size_t c = 0; c < n; c++) {
			float x = (((float)c-m[1])*cosp+((float)r-m[0])*sinp)/a[1];
			float y = (((float)r-m[0])*cosp-((float)c-m[1])*sinp)/a[0];

			res(c,r) = (x*x + y*y) <= 1.0 ? s : T(0.0);
		}
	}

	return res;

}



/**
 * @brief       nxnxn cube with ellipsoid centered at p
 *
 * @param  p    Center point of ellipsoid and excentricities
 * @param  n    Side length of square
 * @param  s    Scaling 
 * @return      Cube with ellipsoid
 */
template <class T> inline static Matrix<T>
ellipsoid (const float* p, const size_t n, const T s) {

	Matrix<T> res (n,n,n);

	float m[3];
	float a[3];
	float d;

	a[0] = p[0] * float(n) / 2.0;
	a[1] = p[1] * float(n) / 2.0;
	a[2] = p[2] * float(n) / 2.0;

	m[0] = (1.0 - p[3]) * float(n) / 2.0;
	m[1] = (1.0 - p[4]) * float(n) / 2.0;
	m[2] = (1.0 - p[5]) * float(n) / 2.0;

	float cosp = cos(p[6]);
	float sinp = sin(p[6]);
	
#pragma omp parallel default (shared) 
	{
#pragma omp for schedule (dynamic, n / omp_get_num_threads())
		
		for (size_t s = 0; s < n; s++)
			for (size_t r = 0; r < n; r++)
				for (size_t c = 0; c < n; c++) {
					float x = (((float)c-m[1])*cosp+((float)r-m[0])*sinp)/a[1];
					float y = (((float)r-m[0])*cosp-((float)c-m[1])*sinp)/a[0];
					float z =  ((float)s-m[2])/a[2];
					res(c,r,s) = (x*x + y*y + z*z) <= 1.0 ? s : T(0.0);
				}
	}

	return res;

}




/**
 * @brief           nxn Shepp-Logan phantom.
 *
 *                  Shepp et al.<br/> 
 *                  The Fourier reconstruction of a head section.<br/> 
 *                  IEEE TNS. 1974; 21: 21-43
 *
 * @param  n        Side length of matrix
 * @return          Shepp-Logan phantom
 */
template<class T> inline static Matrix<T> 
phantom (const size_t& n) {
	
	const size_t ne = 10; // Number of ellipses
	const size_t np = 5;  // Number of geometrical parameters
	
	float p[ne][np] = {
		{ 0.6900, 0.9200,  0.00,  0.0000,  0.0 },
		{ 0.6624, 0.8740,  0.00, -0.0184,  0.0 },
        { 0.1100, 0.3100, -0.22,  0.0000, -0.3 },
		{ 0.1600, 0.4100,  0.22,  0.0000,  0.3 },
		{ 0.2100, 0.2500,  0.00,  0.3500,  0.0 },
		{ 0.0460, 0.0460,  0.00,  0.1000,  0.0 },
		{ 0.0460, 0.0460,  0.00, -0.1000,  0.0 },
		{ 0.0460, 0.0230,  0.08, -0.6050,  0.0 },
		{ 0.0230, 0.0230,  0.00, -0.6060,  0.0 },
		{ 0.0230, 0.0460, -0.06, -0.6050,  0.0 }
	};

	// Size_Tensities
	T v[ne] = {T(1.0), T(-0.8), T(-0.2), T(-0.2), T(0.1), T(0.1), T(0.1), T(0.1), T(0.1), T(0.1)};

	// Empty matrix
	Matrix<T> res (n);
	Matrix<T> e;

	for (size_t i = 0; i < ne; i++) {
		e    = ellipse<T> (p[i], n, v[i]);
		res += e;
	}

	return res;

}





/**
 * @brief           nxnxn Shepp-Logan phantom
 * 
 *                  Koay et al.<br/>
 *                  Three dimensional analytical magnetic resonance imaging phantom in the Fourier domain.<br/>
 *                  MRM. 2007; 58: 430-436
 *
 * @param  n        Side length of matrix
 * @return          nxn zeros
 */
template <class T> inline static Matrix<T> 
phantom3D (const size_t& n) {

	const size_t ne = 10; // Number of ellipses
	const size_t np =  9; // Number of geometrical parameters

	float p[ne][np] = {
		{ 0.690, 0.920, 0.900,  0.00,  0.000,  0.000,  0.0, 0.0, 0.0 },
        { 0.662, 0.874, 0.880,  0.00,  0.000,  0.000,  0.0, 0.0, 0.0 },
        { 0.110, 0.310, 0.220, -0.22,  0.000, -0.250, -0.3, 0.0, 0.0 },
        { 0.160, 0.410, 0.210,  0.22,  0.000, -0.250,  0.3, 0.0, 0.0 },
        { 0.210, 0.250, 0.500,  0.00,  0.350, -0.250,  0.0, 0.0, 0.0 },
        { 0.046, 0.046, 0.046,  0.00,  0.100, -0.250,  0.0, 0.0, 0.0 },
        { 0.046, 0.023, 0.020,  0.08, -0.650, -0.250,  0.0, 0.0, 0.0 },
        { 0.046, 0.023, 0.020,  0.06, -0.650, -0.250,  0.0, 0.0, 0.0 },
        { 0.056, 0.040, 0.100, -0.06, -0.105,  0.625,  0.0, 0.0, 0.0 },
        { 0.056, 0.056, 0.100,  0.00,  0.100,  0.625,  0.0, 0.0, 0.0 }
	};

	T v[ne] = {2.0, -0.8, -0.2, -0.2, 0.2, 0.2, 0.1, 0.1, 0.2, -0.2};

	Matrix<T> res = zeros<T>(n,n,n);
	Matrix<T> e;
	
	for (size_t i = 0; i < ne; i++) {
		e    = ellipsoid<T> (p[i], n, v[i]);
		res += e;
	}

	return res;

}



template <class T>
inline static Matrix<T>
eye (const size_t n) {

 	Matrix<T> M (n);

 	for (size_t i = 0; i < n; i++)
 		M[i*n+i] = T(1.0);

 	return M;

}



template <class T> inline static Matrix<T> 
linspace (const T& start, const T& end, const size_t& n) {
	
	assert (n >= 1);
	
	Matrix<T> res (n, 1);
	T gap;

	gap      = T(end-start) / T(n-1);
	
	res[0]   = start;
	res[n-1] = end;
	
	for (size_t i = 1; i < n-1; i++)
		res[i] = res[i-1] + gap;
	
	return res;
	
}


/**
 * @brief    MATLAB-like meshgrid. x and y vectors must be specified z may be specified optionally.
 *
 * @param x  X-Vector
 * @param y  Y-Vector
 * @param z  Z-Vector (default: unused)
 * @return   Mesh grid O (Ny x Nx x Nz x 3) (if z specified) else O (Ny x Nx x 2)<br/>
 */
template <class T> inline static Matrix<T>
meshgrid (const Matrix<T>& x, const Matrix<T>& y, const Matrix<T>& z = Matrix<T>(1)) {

	size_t nx = numel(x);
	size_t ny = numel(y);
	size_t nz = numel(z);

	assert (nx > 1);
	assert (ny > 1);

	// Column vectors
	assert (size(x,0) == nx); 
	assert (size(y,0) == ny);
	assert (size(z,0) == nz);

	Matrix<T> res (ny, nx, (nz > 1) ? nz : 2, (nz > 1) ? 3 : 1);
	
	for (size_t i = 0; i < ny * nz; i++) 
		Row    (res, i          , x);
	for (size_t i = 0; i < nx * nz; i++) 
		Column (res, i + nx * nz, y);
	if (nz > 1)
		for (size_t i = 0; i < nz; i++)
			Slice  (res, i +  2 * nz, z[i]);
	
	return res;	


}


#endif


