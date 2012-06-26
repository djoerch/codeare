
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

#ifndef __LAPACK_HPP__
#define __LAPACK_HPP__

extern "C" {

	// Cholesky factorization of a complex Hermitian positive definite matrix
	void cpotrf_ (const char* uplo, const int* n, void* a, const int* lda, int *info);
	void dpotrf_ (const char* uplo, const int* n, void* a, const int* lda, int *info);
	void spotrf_ (const char* uplo, const int* n, void* a, const int* lda, int *info);
	void zpotrf_ (const char* uplo, const int* n, void* a, const int* lda, int *info);
	
	// Computes an LU factorization of a general M-by-N matrix A
	void cgetrf_ (int* m, int*n, void *a, int* lda, int*ipiv, int*info);
	void dgetrf_ (int* m, int*n, void *a, int* lda, int*ipiv, int*info);
	void zgetrf_ (int* m, int*n, void *a, int* lda, int*ipiv, int*info);
	void sgetrf_ (int* m, int*n, void *a, int* lda, int*ipiv, int*info);
	
	// Inverse of a complex Hermitian positive definite matrix using cpotrf/cpptrf
	void cpotri_ (const char* uplo, int*n, void *a, int* lda, int*info);
	void dpotri_ (const char* uplo, int*n, void *a, int* lda, int*info);
	void zpotri_ (const char* uplo, int*n, void *a, int* lda, int*info);
	void spotri_ (const char* uplo, int*n, void *a, int* lda, int*info);
	
	// Matrix inversion through cholesky decomposition
	void cgetri_ (int *n, void *a, int* lda, int *ipiv, void *work, int *lwork, int*info);
	void dgetri_ (int *n, void *a, int* lda, int *ipiv, void *work, int *lwork, int*info);
	void zgetri_ (int *n, void *a, int* lda, int *ipiv, void *work, int *lwork, int*info);
	void sgetri_ (int *n, void *a, int* lda, int *ipiv, void *work, int *lwork, int*info);
	
	// Eigen value computations
	void cgeev_  (const char *jvl, const char *jvr, int *n, const void *a, int *lda, void *w ,           void *vl, int *ldvl, void *vr, int *ldvr, void *work, int *lwork, void *rwork, int *info);
	void zgeev_  (const char *jvl, const char *jvr, int *n, const void *a, int *lda, void *w ,           void *vl, int *ldvl, void *vr, int *ldvr, void *work, int *lwork, void *rwork, int *info);
	void dgeev_  (const char *jvl, const char *jvr, int *n, const void *a, int *lda, void *wr, void *wi, void *vl, int *ldvl, void *vr, int *ldvr, void *work, int *lwork,              int *info);
	void sgeev_  (const char *jvl, const char *jvr, int *n, const void *a, int *lda, void *wr, void *wi, void *vl, int *ldvl, void *vr, int *ldvr, void *work, int *lwork,              int *info);
	
	// Singular value decomposition 
	void cgesdd_ (const char *jobz, int*m, int *n, void *a, int *lda, void *s, void*u, int*ldu, void *vt, int *ldvt, void *work, int*lwork, void *rwork, int *iwork, int*info);
	void zgesdd_ (const char *jobz, int*m, int *n, void *a, int *lda, void *s, void*u, int*ldu, void *vt, int *ldvt, void *work, int*lwork, void *rwork, int *iwork, int*info);
	void dgesdd_ (const char *jobz, int*m, int *n, void *a, int *lda, void *s, void*u, int*ldu, void *vt, int *ldvt, void *work, int*lwork,               int *iwork, int*info);
	void sgesdd_ (const char *jobz, int*m, int *n, void *a, int *lda, void *s, void*u, int*ldu, void *vt, int *ldvt, void *work, int*lwork,               int *iwork, int*info);
	
	// Pseudo-inversion 
	void zgelsd_ (int* m, int* n, int* nrhs, const void* a, int* lda, void* b, int* ldb, void* s, void* rcond, int* rank, void* work, int* lwork, void* rwork, int* iwork, int* info);
	void cgelsd_ (int* m, int* n, int* nrhs, const void* a, int* lda, void* b, int* ldb, void* s, void* rcond, int* rank, void* work, int* lwork, void* rwork, int* iwork, int* info);
	void dgelsd_ (int* m, int* n, int* nrhs, const void* a, int* lda, void* b, int* ldb, void *s, void* rcond, int* rank, void* work, int* lwork, void*             iwork, int* info);
	void sgelsd_ (int* m, int* n, int* nrhs, const void* a, int* lda, void* b, int* ldb, void *s, void* rcond, int* rank, void* work, int* lwork, void*             iwork, int* info);

	// Matrix vector multiplication
	void sgemv_  (const char* trans, int* m, int* n, void* alpha, const void *a, int* lda, const void *x, int* incx, void* beta, void *y, int* incy);
	void dgemv_  (const char* trans, int* m, int* n, void* alpha, const void *a, int* lda, const void *x, int* incx, void* beta, void *y, int* incy);
	void cgemv_  (const char* trans, int* m, int* n, void* alpha, const void *a, int* lda, const void *x, int* incx, void* beta, void *y, int* incy);
	void zgemv_  (const char* trans, int* m, int* n, void* alpha, const void *a, int* lda, const void *x, int* incx, void* beta, void *y, int* incy);

	// Matrix matrix multiplication
	void sgemm_  (const char *transa, const char *transb, int  *m, int   *n, int *k, void *alpha, const void *a, int *lda, const void *b, int *ldb, void *beta, void *c, int *ldc);
	void dgemm_  (const char *transa, const char *transb, int  *m, int   *n, int *k, void *alpha, const void *a, int *lda, const void *b, int *ldb, void *beta, void *c, int *ldc);
	void cgemm_  (const char *transa, const char *transb, int  *m, int   *n, int *k, void *alpha, const void *a, int *lda, const void *b, int *ldb, void *beta, void *c, int *ldc);
	void zgemm_  (const char *transa, const char *transb, int  *m, int   *n, int *k, void *alpha, const void *a, int *lda, const void *b, int *ldb, void *beta, void *c, int *ldc);
	
	// vector vector scalar multiplication
	float  sdot_  (int* n, const void* x, int* incx, const void* y, int* incy);
	double ddot_  (int* n, const void* x, int* incx, const void* y, int* incy);
	cxfl   cdotc_ (int* n, const void* x, int* incx, const void* y, int* incy);
	cxfl   cdotu_ (int* n, const void* x, int* incx, const void* y, int* incy);
	cxdb   zdotc_ (int* n, const void* x, int* incx, const void* y, int* incy);
	cxdb   zdotu_ (int* n, const void* x, int* incx, const void* y, int* incy);

    float  cblas_snrm2  (const int N, const void *X, const int incX);
	double cblas_dnrm2  (const int N, const void *X, const int incX);
	cxfl   cblas_scnrm2 (const int N, const void *X, const int incX);
	cxdb   cblas_dznrm2 (const int N, const void *X, const int incX);
	
	void cblas_ddot_sub  (const int N, const void *X, const int incX, const void *Y, const int incY, void* res);	
	void cblas_sdot_sub  (const int N, const void *X, const int incX, const void *Y, const int incY, void* res);	
	void cblas_cdotu_sub (const int N, const void *X, const int incX, const void *Y, const int incY, void* res);	
	void cblas_cdotc_sub (const int N, const void *X, const int incX, const void *Y, const int incY, void* res);	
	void cblas_zdotu_sub (const int N, const void *X, const int incX, const void *Y, const int incY, void* res);	
	void cblas_zdotc_sub (const int N, const void *X, const int incX, const void *Y, const int incY, void* res);	
}

#include "Matrix.hpp"
#include "Algos.hpp"
#include "Creators.hpp"


/**
 * @brief         Eigenvalue decomposition
 * 
 * Usage:
 * @code{.cpp}
 *   Matrix<cxfl> m = rand<cxfl> (10,10), lv, rv;
 *   Matrix<float> ev;
 * 
 *   int res = eig (m, ev, lv, rv, 'N', 'N');
 * @endcode
 * where m is the complex decomposed matrix, lv and rv are the left hand and right 
 * hand side eigen vectors and ev is the real eigenvalue vector.
 *
 * @see           LAPACK driver xGEEV
 *
 * @param  m      Matrix for decomposition
 * @param  ev     Eigenvalues
 * @param  lv     Left  Eigen-vectors
 * @param  rv     Right Eigen-vectors
 * @param  jobvl  Compute left vectors ('N'/'V')
 * @param  jobvr  Compute right vectors ('N'/'V')
 * @return        Status of driver
 */
template <class T, class S> static int
eig (const Matrix<T>& m, Matrix<S>& ev, Matrix<T>& lv, Matrix<T>& rv, const char& jobvl = 'N', const char& jobvr = 'N') {
	
	if (jobvl != 'N' && jobvl !='V') {
		printf ("EIG Error: Parameter jobvl ('%c' provided) must be 'N' or 'V' \n", jobvl);
		return -1;
	}
	
	if (jobvr != 'N' && jobvr !='V') {
		printf ("EIG Error: Parameter jobvl ('%c' provided) must be 'N' or 'V' \n", jobvr);
		return -1;
	}
	
	// 2D 
	if (!Is2D(m)) {
		printf ("EIG Error: Parameter m must be 2D");
		return -2;
	}
	
	// Square matrix
	if (m.Width() != m.Height()){
		printf ("EIG Error: Parameter m must be square");
		return -3;
	}
	
	int    N     =  size(m, 0);
	
	int    lda   =  N;
	int    ldvl  =  (jobvl) ? N : 1;
	int    ldvr  =  (jobvr) ? m.Width() : 1;
	int    info  =  0;
	int    lwork = -1;
	
	T*     w     = 0;
	T*     wi    = 0;
	T*     rwork = 0;
	
	if (typeid(T) == typeid(float) || typeid(T) == typeid(double)) {    // Complex eigen values for real matrices
		w     = (T*) malloc (N * sizeof(T));
		wi    = (T*) malloc (N * sizeof(T));
	} else if (typeid(T) == typeid(cxfl) || typeid(T) == typeid(cxdb))  // Real workspace for complex matrices
		rwork = (T*) malloc (N * sizeof(T));
	
	T  wkopt;
	
	// Workspace query
	if (typeid(T) == typeid(cxfl))
		cgeev_ (&jobvl, &jobvr, &N, m.Data(), &lda, &ev[0], &lv[0], &ldvl, &rv[0], &ldvr, &wkopt, &lwork, rwork, &info);
	else if (typeid(T) == typeid(cxdb))
		zgeev_ (&jobvl, &jobvr, &N, m.Data(), &lda, &ev[0], &lv[0], &ldvl, &rv[0], &ldvr, &wkopt, &lwork, rwork, &info);
	else if (typeid(T) == typeid(double))
		dgeev_ (&jobvl, &jobvr, &N, m.Data(), &lda,  w, wi, &lv[0], &ldvl, &rv[0], &ldvr, &wkopt, &lwork,        &info);
	else if (typeid(T) == typeid(float))
		sgeev_ (&jobvl, &jobvr, &N, m.Data(), &lda,  w, wi, &lv[0], &ldvl, &rv[0], &ldvr, &wkopt, &lwork,        &info);
	
	// Intialise work space
	lwork    = (int) creal (wkopt);
	T* work  = (T*) malloc (lwork * sizeof(T));
	
	// Actual eigen value comp
	if (typeid(T) == typeid(cxfl)) {
		cgeev_ (&jobvl, &jobvr, &N, m.Data(), &lda, &ev[0], &lv[0], &ldvl, &rv[0], &ldvr, work, &lwork, rwork, &info);
	} else if (typeid(T) == typeid(cxdb)) {
		zgeev_ (&jobvl, &jobvr, &N, m.Data(), &lda, &ev[0], &lv[0], &ldvl, &rv[0], &ldvr, work, &lwork, rwork, &info);
	} else if (typeid(T) == typeid(double)) {
		dgeev_ (&jobvl, &jobvr, &N, m.Data(), &lda,  w, wi, &lv[0], &ldvl, &rv[0], &ldvr, work, &lwork,        &info);
		while (N--) {
			double f[2] = {((double*)w)[N], ((double*)wi)[N]};
			memcpy(&ev[N], f, 2 * sizeof(double));
		}
	} else if (typeid(T) == typeid(float)) {
		sgeev_ (&jobvl, &jobvr, &N, m.Data(), &lda,  w, wi, &lv[0], &ldvl, &rv[0], &ldvr, work, &lwork,        &info);
		while (N--) {
			float f[2] = {((float*)w)[N], ((float*)wi)[N]};
			memcpy(&ev[N], f, 2 * sizeof(float));
		}
	}
	
	
	// Clean up
	if (typeid(T) == typeid(float) || typeid(T) == typeid(double)) {
		free (w);
		free (wi);
	} else if (typeid(T) == typeid(cxfl) || typeid(T) == typeid(cxdb)) 
		free (rwork);

	free (work);
	
	if (info > 0)
		printf ("\nERROR - XGEEV: the QR algorithm failed to compute all the\n eigenvalues, and no eigenvectors have been computed;\n elements i+1:N of ev contain eigenvalues which\n have converged.\n\n") ;
	else if (info < 0)
		printf ("\nERROR - XGEEV: the %i-th argument had an illegal value.\n\n", -info); 
	
	return info;
	
}


/**
 * @brief           Singular value decomposition
 *
 * Usage:
 * @code{.cpp}
 *   Matrix<cxfl>  m = rand<cxfl> (20,10), u, v;
 *   Matrix<float> s;
 * 
 *   int res = svd (m, s, u, v, 'N');
 * @endcode
 * where m is the complex decomposed matrix, u and v are the left hand and right 
 * hand side singular vectors and s is the vector of real singular values.
 *
 * @see             LAPACK driver xGESDD
 * 
 * @param  IN       Incoming matrix
 * @param  s        Sorted singular values 
 * @param  U        Left-side singlar vectors
 * @param  V        Right-side singular vectors
 * @param  jobz     Computation mode<br/>
 *                  'A': all m columns of U and all n rows of VT are returned in the arrays u and vt<br/>
 *                  'S', the first min(m, n) columns of U and the first min(m, n) rows of VT are returned in the arrays u and vt;<br/>
 *                  'O', then<br/>
 *                  &nbsp;&nbsp;&nbsp;&nbsp;if m >= n, the first n columns of U are overwritten in the array a and all rows of VT are returned in the array vt;<br/>
 *                  &nbsp;&nbsp;&nbsp;&nbsp;if m < n, all columns of U are returned in the array u and the first m rows of VT are overwritten in the array a;<br/>
 *                  'N', no columns of U or rows of VT are computed.
 * @return          Status of the driver
 */

template<class T, class S> static int 
svd (const Matrix<T>& IN, Matrix<S>& s, Matrix<T>& U, Matrix<T>& V, const char& jobz = 'N') {
	
	Matrix<T> A (IN);
	
	// SVD only defined on 2D data
	if (!Is2D(A))
		return -2;
	
	int   m, n, lwork, info, lda, mn, ldu = 1, ucol = 1, ldvt = 1, vcol = 1;
	T     wopt;
	void* rwork = 0;
	
	m     =  A.Height();
	n     =  A.Width();
	lwork = -1;
	info  =  0;
	lda   =  m;
	mn    =  MIN(m,n);
	
	if    (((jobz == 'A' || jobz == 'O') && m < n)) 
		ucol = m;
	else if (jobz == 'S')
		ucol = mn;
	
	if      (jobz == 'S' || jobz == 'A' || (jobz == 'O' &&  m < n)) 
		ldu = m;
	
	if      (jobz == 'A' || (jobz == 'O' && m >= n))
		ldvt = n;
	else if (jobz == 'S')
		ldvt = mn;
	
	if      (jobz != 'N')
		vcol = n;
	
	s.Resize (mn,1);
	U.Resize (ldu,ucol);
	V.Resize (ldvt,vcol);
	
	int*   iwork =   (int*) malloc (8 * mn * sizeof(int));
	
	// Only needed for complex data
	if (typeid(T) == typeid(cxfl) || typeid(T) == typeid(cxdb)) {
		if (jobz == 'N') rwork = malloc (mn * 7            * sizeof(T) / 2);
		else             rwork = malloc (mn * (5 * mn + 7) * sizeof(T) / 2);
	}
	
	// Workspace query
	if      (typeid(T) == typeid(cxfl))
		cgesdd_ (&jobz, &m, &n, &A[0], &lda, &s[0], &U[0], &ldu, &V[0], &ldvt, &wopt, &lwork, rwork, iwork, &info);
	else if (typeid(T) == typeid(cxdb))
		zgesdd_ (&jobz, &m, &n, &A[0], &lda, &s[0], &U[0], &ldu, &V[0], &ldvt, &wopt, &lwork, rwork, iwork, &info);
	else if (typeid(T) == typeid(double))
		dgesdd_ (&jobz, &m, &n, &A[0], &lda, &s[0], &U[0], &ldu, &V[0], &ldvt, &wopt, &lwork,        iwork, &info);
	else if (typeid(T) == typeid(float))
		sgesdd_ (&jobz, &m, &n, &A[0], &lda, &s[0], &U[0], &ldu, &V[0], &ldvt, &wopt, &lwork,        iwork, &info);
	
	// Resize work according to ws query
	lwork   = (int) creal (wopt);
	T* work = (T*) malloc (lwork * sizeof(T));
	
	//SVD
	if      (typeid(T) == typeid(cxfl))
		cgesdd_ (&jobz, &m, &n, &A[0], &lda, &s[0], &U[0], &ldu, &V[0], &ldvt, work, &lwork, rwork, iwork, &info);
	else if (typeid(T) == typeid(cxdb))
		zgesdd_ (&jobz, &m, &n, &A[0], &lda, &s[0], &U[0], &ldu, &V[0], &ldvt, work, &lwork, rwork, iwork, &info);
	else if (typeid(T) == typeid(double))
		dgesdd_ (&jobz, &m, &n, &A[0], &lda, &s[0], &U[0], &ldu, &V[0], &ldvt, work, &lwork,        iwork, &info);
	else if (typeid(T) == typeid(float))
		sgesdd_ (&jobz, &m, &n, &A[0], &lda, &s[0], &U[0], &ldu, &V[0], &ldvt, work, &lwork,        iwork, &info);
	
	V = !V;
	
	// Clean up
	if (typeid (T) == typeid (cxfl) || typeid (T) == typeid (cxdb)) 
		free (rwork);
	
	free (work);
	free (iwork);
	
	if (info > 0)
		printf ("\nERROR - XGESDD: The updating process of SBDSDC did not converge.\n\n");
	else if (info < 0)
		printf ("\nERROR - XGESDD: The %i-th argument had an illegal value.\n\n", -info); 
	
	return info;
	
} 
	


/**
 * @brief                Invert quadratic well conditioned matrix
 *
 * Usage:
 * @code{.cpp}
 *   Matrix<cxfl> m = rand<cxfl> (10,10);
 *  
 *   m = inv (m);
 * @endcode
 *
 * @see                  Lapack xGETRF/xGETRI
 * 
 * @param  m             Matrix
 * @return               Inverse
*/
template <class T> static Matrix<T> 
inv (const Matrix<T>& m) {
	
	// 2D 
	if (!Is2D(m))       printf ("Inv Error: Parameter m must be 2D");
	
	// Square matrix
	if (size(m,1) != size (m,0)) printf ("Inv Error: Parameter m must be square");
		
	int N = (int) size (m,0);	
	Matrix<T> res = m;
	int  info = 0;
	int *ipiv = (int*) malloc (N * sizeof(int));
	
	// LU Factorisation -------------------
	
	if      (typeid (T) == typeid (cxfl))   cgetrf_ (&N, &N, &res[0], &N, ipiv, &info);
	else if (typeid (T) == typeid (cxdb))   zgetrf_ (&N, &N, &res[0], &N, ipiv, &info);
	else if (typeid (T) == typeid (double)) dgetrf_ (&N, &N, &res[0], &N, ipiv, &info);
	else if (typeid (T) == typeid (float))  sgetrf_ (&N, &N, &res[0], &N, ipiv, &info);
	// ------------------------------------
	
	if (info < 0)
		printf ("\nERROR - DPOTRI: the %i-th argument had an illegal value.\n\n", -info);
	else if (info > 1)
		printf ("\nERROR - DPOTRI: the (%i,%i) element of the factor U or L is\n zero, and the inverse could not be computed.\n\n", info, info);
	
	int lwork = -1; 
	T   wopt;
	
	// Workspace determination ------------
	
	if      (typeid (T) == typeid (cxfl))   cgetri_ (&N, &res[0], &N, ipiv, &wopt, &lwork, &info);
	else if (typeid (T) == typeid (cxdb))   zgetri_ (&N, &res[0], &N, ipiv, &wopt, &lwork, &info);
	else if (typeid (T) == typeid (double)) dgetri_ (&N, &res[0], &N, ipiv, &wopt, &lwork, &info);
	else if (typeid (T) == typeid (float))  sgetri_ (&N, &res[0], &N, ipiv, &wopt, &lwork, &info);
	// ------------------------------------
	
	// Work memory allocation -------------
	
	lwork   = (int) creal (wopt);
	T* work = (T*) malloc (lwork * sizeof(T));
	// ------------------------------------
	
	// Inversion --------------------------
	
	if      (typeid (T) == typeid (cxfl))   cgetri_ (&N, &res[0], &N, ipiv, work, &lwork, &info);
	else if (typeid (T) == typeid (cxdb))   zgetri_ (&N, &res[0], &N, ipiv, work, &lwork, &info);
	else if (typeid (T) == typeid (double)) dgetri_ (&N, &res[0], &N, ipiv, work, &lwork, &info);
	else if (typeid (T) == typeid (float))  sgetri_ (&N, &res[0], &N, ipiv, work, &lwork, &info);
	// ------------------------------------
	
	free (ipiv);
	free (work);
	
	if (info < 0)
		printf ("\nERROR - XGETRI: The %i-th argument had an illegal value.\n\n", -info);
	else if (info > 0)
		printf ("\nERROR - XGETRI: The leading minor of order %i is not\n positive definite, and the factorization could not be\n completed.", info);
	
	return res;
	
} 


/**
 * @brief                Pseudo invert though SVD
 * 
 * Usage:
 * @code{.cpp}
 *   Matrix<cxfl> m = rand<cxfl> (20,10);
 *
 *   m = pinv (m);
 * @endcode
 *
 * @see                  LAPACK driver xGELSD
 * 
 * @param  m             Matrix
 * @param  rcond         Condition number
 * @return               Moore-Penrose pseudoinverse
*/
template<class T> static Matrix<T> 
pinv (const Matrix<T>& m, double rcond = 1.0) {
	
	void *s = 0, *rwork = 0;
	T    *work = 0, wopt = T(0), rwopt = T(0);
	int  *iwork = 0, iwopt = 0;
	
	int  M      =  size(m, 0);
	int  N      =  size(m, 1);
	
	int  nrhs   =  M;
	int  lda    =  M;
	int  ldb    =  MAX(M,N);
	int  lwork  = -1; 
	int  rank   =  0;
	int  info   =  0;
	int  swork  =  sizeof(T) * MIN(M,N);
	
	if (typeid (T) == typeid(cxfl) || typeid (T) == typeid(cxdb))
		swork /= 2;
	
	s      =        malloc (swork);
	
	Matrix<T> b = eye<T>(ldb);
	
	float frcond  = rcond;
	
	if      (typeid(T) == typeid(cxfl))
		cgelsd_ (&M, &N, &nrhs, m.Data(), &lda, &b[0], &ldb, s, &frcond, &rank, &wopt, &lwork, &rwopt, &iwopt, &info);
	else if (typeid(T) == typeid(cxdb))
		zgelsd_ (&M, &N, &nrhs, m.Data(), &lda, &b[0], &ldb, s, &rcond,  &rank, &wopt, &lwork, &rwopt, &iwopt, &info);
	else if (typeid(T) == typeid(double))
		dgelsd_ (&M, &N, &nrhs, m.Data(), &lda, &b[0], &ldb, s, &rcond,  &rank, &wopt, &lwork,         &iwopt, &info);
	else if (typeid(T) == typeid(float))
		sgelsd_ (&M, &N, &nrhs, m.Data(), &lda, &b[0], &ldb, s, &frcond, &rank, &wopt, &lwork,         &iwopt, &info);
	
	lwork = (int) creal(wopt);
	
	if      (typeid (T) == typeid(cxfl) || typeid (T) == typeid(cxdb))
		rwork =    malloc ((sizeof(T)/2) * (int) creal(rwopt));
	
	iwork = (int*) malloc (sizeof(int) * iwopt);
	work  = (T*)   malloc (sizeof(T) * lwork);
	
	if (typeid(T) == typeid(cxfl))
		cgelsd_ (&M, &N, &nrhs, m.Data(), &lda, &b[0], &ldb, s, &frcond, &rank, work, &lwork, rwork, iwork, &info);
	else if (typeid(T) == typeid(cxdb))
		zgelsd_ (&M, &N, &nrhs, m.Data(), &lda, &b[0], &ldb, s, &rcond,  &rank, work, &lwork, rwork, iwork, &info);
	else if (typeid(T) == typeid(double))
		dgelsd_ (&M, &N, &nrhs, m.Data(), &lda, &b[0], &ldb, s, &rcond,  &rank, work, &lwork,        iwork, &info);
	else if (typeid(T) == typeid(float))
		sgelsd_ (&M, &N, &nrhs, m.Data(), &lda, &b[0], &ldb, s, &frcond, &rank, work, &lwork,        iwork, &info);

	
	if (M > N)
		for (int i = 0; i < M; i++)
			memcpy (&b[i*N], &b[i*M], N * sizeof(T));
	
	b.Resize(N,M);
	
	if      (typeid (T) == typeid(cxfl) || typeid (T) == typeid(cxdb))
		free (rwork);
	
	free (s);
	free (work);
	free (iwork);
	
	if (info > 0)
		printf ("ERROR XGELSD: the algorithm for computing the SVD failed to converge;\n %i off-diagonal elements of an intermediate bidiagonal form\n did not converge to zero.", info);
	else if (info < 0)
		printf ("ERROR XGELSD: the %i-th argument had an illegal value.", -info);
	
	return b;
	
}
	
	
/**
 * @brief        Cholesky decomposition of positive definite quadratic matrix
 *
 * Usage:
 * @code{.cpp}
 *   Matrix<cxfl> m = rand<cxfl> (20,10);
 *   
 *   m = m.prodt(m); // m*m' Must be positive definite 
 *   m = chol (m);
 * @endcode
 *
 * @see          LAPACK driver xPOTRF
 * 
 * @param  A     Incoming matrix
 * @param  uplo  Use upper/lower triangle for decomposition ('U': default/'L')
 * @return       Cholesky decomposition
 */
template<class T> static Matrix<T> 
chol (const Matrix<T>& A, const char uplo = 'U') {
	
	Matrix<T> res  = A;
	int       info = 0, n = A.Height();
	
	if      (typeid(T) == typeid(double)) dpotrf_ (&uplo, &n, &res[0], &n, &info);
	else if (typeid(T) == typeid(float))  spotrf_ (&uplo, &n, &res[0], &n, &info);
	else if (typeid(T) == typeid(cxdb))   zpotrf_ (&uplo, &n, &res[0], &n, &info);
	else if (typeid(T) == typeid(cxfl))   cpotrf_ (&uplo, &n, &res[0], &n, &info);
	
	if (info > 0)
		printf ("\nERROR - XPOTRF: the leading minor of order %i is not\n positive definite, and the factorization could not be\n completed!\n\n", info);
	else if (info < 0)
		printf ("\nERROR - XPOTRF: the %i-th argument had an illegal value.\n\n!", -info); 
	
	return res;
	
}


/**
 * @brief          Matrix matrix multiplication
 *
 * Usage:
 * @code{.cpp}
 *   Matrix<cxfl> m = rand<cxfl> (20,10);
 *   Matrix<cxfl> x = rand<cxfl> (10, 6);
 *  
 *   m   = gemm (m, b, 'N', 'C');
 * @endcode
 *
 * @see            BLAS routine xGEMM
 *
 * @param  A       Left factor
 * @param  B       Right factor
 * @param  transa  (N: A*... | T: A.'*... | C: A'*...) transpose left factor
 * @param  transb  (N: ...*B | T: ...*B.' | C: ...*B') transpose right factor
 * @return         Product
 */
template<class T> static Matrix<T> 
gemm (const Matrix<T>& A, const Matrix<T>& B, char transa = 'N', char transb = 'N') {
	
	int aw, ah, bw, bh, m, n, k, ldc;
	T   alpha, beta;

	aw = (int)size(A,1); ah = (int)size(A,0), bw = (int)size(B,1), bh = (int)size(B,0);
	
	if      ( transa == 'N'                   &&  transb == 'N'                  ) assert (aw == bh);
	else if ( transa == 'N'                   && (transb == 'T' || transb == 'C')) assert (aw == bw);
	else if ((transa == 'T' || transa == 'C') &&  transb == 'N'                  ) assert (ah == bh);
	else if ((transa == 'T' || transa == 'C') && (transb == 'T' || transb == 'C')) assert (ah == bw);
	
	if (transa == 'N') {
		m = ah;
		k = aw;
	} else if (transa == 'T' || transa == 'C') {
		m = aw;
		k = ah;
	}
	
	if (transb == 'N')
		n = bw;
	else if (transb == 'T' || transb == 'C')
		n = bh;
	
	ldc = m;
	
	alpha =       T(1.0);
	beta  =       T(0.0);
	
	Matrix<T> C  (m, n);
	
	if      (typeid(T) == typeid(double))
		dgemm_ (&transa, &transb, &m, &n, &k, &alpha, A.Data(), &ah, B.Data(), &bh, &beta, &C[0], &ldc);
	else if (typeid(T) == typeid(float))
		sgemm_ (&transa, &transb, &m, &n, &k, &alpha, A.Data(), &ah, B.Data(), &bh, &beta, &C[0], &ldc);
	else if (typeid(T) == typeid(cxfl))
		cgemm_ (&transa, &transb, &m, &n, &k, &alpha, A.Data(), &ah, B.Data(), &bh, &beta, &C[0], &ldc);
	else if (typeid(T) == typeid(cxdb))
		zgemm_ (&transa, &transb, &m, &n, &k, &alpha, A.Data(), &ah, B.Data(), &bh, &beta, &C[0], &ldc);
	
	return C;
	
}



/**
 * @brief              Eclidean norm
 *
 * Usage:
 * @code{.cpp}
 *   Matrix<cxfl> m = rand<cxfl> (20,10);
 *   float normm    = creal(norm (m)); // Lapack driver below produces complex variable with real value
 * @cendode
 *
 * @param  M           Input
 * @return             Eclidean norm
 */
template<class T> static T
norm (const Matrix<T>& M) {
	
	T   res  = T(0);
	
	int n    = (int) numel (M);
	int incx = 1;
	
	if      (typeid(T) == typeid(  cxfl)) res = cblas_scnrm2 (n, M.Data(), incx);
	else if (typeid(T) == typeid(  cxdb)) res = cblas_dznrm2 (n, M.Data(), incx);
	else if (typeid(T) == typeid(double)) res = cblas_dnrm2  (n, M.Data(), incx);
	else if (typeid(T) == typeid( float)) res = cblas_snrm2  (n, M.Data(), incx);
	else {
		while (n--)
			res += pow(M[n],2.0);
		sqrt (res);
	}
	
	return res;
	
}



template <class T> static T 
DOTC (const Matrix<T>& A, const Matrix<T>& B) {
	
	int n, one;
	T   res;

	n   = (int) numel(A);
	assert (n == (int) numel(B));
	
	res = T(0.0);
	one = 1;
	
	if      (typeid(T) == typeid(cxfl)) cblas_cdotc_sub (n, A.Data(), one, B.Data(), one, &res);
	else if (typeid(T) == typeid(cxdb)) cblas_zdotc_sub (n, A.Data(), one, B.Data(), one, &res);
	
	return res;
	
}

/**
 * @brief              Complex dot product (A'*B) on data vector
 *
 * Usage:
 * @code{.cpp}
 *   Matrix<cxdb> a = rand<cxdb> (20,1);
 *   Matrix<cxdb> b = rand<cxdb> (20,1);
 *   double dotpr   = dotc (a, b);
 * @cendode
 *
 * @param  A           Left factor (is conjugated)
 * @param  B           Right factor 
 * @return             A'*B
 */
template <class T> static T 
dotc (const Matrix<T>& A, const Matrix<T>& B) {
	return DOTC (A,B);
}


template <class T> static T 
DOTU (const Matrix<T>& A, const Matrix<T>& B) {

	int n, one;
	T   res;

	n   = (int) numel(A);
	assert (n == (int) numel(B));
	
	res = T(0.0);
	one = 1;
	
	if      (typeid(T) == typeid(cxfl)) cblas_cdotu_sub (n, A.Data(), one, B.Data(), one, &res);
	else if (typeid(T) == typeid(cxdb)) cblas_zdotu_sub (n, A.Data(), one, B.Data(), one, &res);
	
	return res;
	
}

/**
 * @brief              Complex dot product (A*B) on data vector
 *
 * Usage: 
 * @code{.cpp}
 *   Matrix<cxdb> a = rand<cxdb> (20,1);
 *   Matrix<cxdb> b = rand<cxdb> (20,1);
 *   double dotpr   = dotu (a, b);
 * @cendode
 *
 * @param  A           Left factor
 * @param  B           Right factor 
 * @return             A*B
 */
template <class T> static T 
dotu (const Matrix<T>& A, const Matrix<T>& B) {
	DOTU (A, B);
}


template <class T> T 
DOT  (const Matrix<T>& A, const Matrix<T>& B) {
	
	int n, one;
	T   res;

	n   = (int) numel(A);
	assert (n == (int) numel(B));
	
	res = T(0.0);
	one = 1;
	
	if      (typeid(T) == typeid(cxfl))   cblas_cdotu_sub (n, A.Data(), one, B.Data(), one, &res);
	else if (typeid(T) == typeid(cxdb))   cblas_zdotu_sub (n, A.Data(), one, B.Data(), one, &res);
	else if (typeid(T) == typeid(double)) cblas_ddot_sub  (n, A.Data(), one, B.Data(), one, &res);
	else if (typeid(T) == typeid(float))  cblas_sdot_sub  (n, A.Data(), one, B.Data(), one, &res);
	
	return res;
	
}


/**
 * @brief              Dot product (A*B) on data vector
 *
 * Usage:
 * @code{.cpp}
 *   Matrix<float> a = rand<float> (20,1);
 *   Matrix<float> b = rand<float> (20,1);
 *   double dotpr    = dot (a, b);
 * @cendode
 *
 * @param  A           Left factor
 * @param  B           Right factor 
 * @return             A*B
 */
template <class T> T 
dot  (const Matrix<T>& A, const Matrix<T>& B) {
	return DOT (A, B);
}


/**
 * @brief             Matrix vector product A*x
 *
 * Usage: 
 * @code{.cpp}
 *   Matrix<cxdb> A  = rand<cxdb> (20,5);
 *   Matrix<cxdb> x  = rand<cxdb> (20,1);
 *   double prod     = gemv (A, x, 'C');
 * @cendode
 *
 * @param  A          left factor matrix
 * @param  x          Right factor vector
 * @param  trans      Transpose A?
 * 
 * @return            A*x
 */
template<class T> Matrix<T> 
gemv (const Matrix<T>& A, const Matrix<T>& x, char trans = 'N') {
	
	int aw, ah, xh, m, n, one;
	T   alpha, beta;

	// Column vector
	assert (size(x, 1) == 1);
	
	aw  = (int) size (A, 1);
	ah  = (int) size (A, 0);
	xh  = (int) size (x, 0);
	
	m   = ah;
	n   = aw; 
	one = 1;
	
	if (trans == 'N')
		assert (aw == xh);
	else if (trans == 'T' || trans == 'C')
		assert (ah == xh);
	
	alpha  = T(1.0);
	beta   = T(0.0);
	
	Matrix<T> y ((trans == 'N') ? m : n, 1);
	
	if      (typeid(T) == typeid(double)) dgemv_ (&trans, &m, &n, &alpha, A.Data(), &ah, x.Data(), &one, &beta, &y[0], &one);
	else if (typeid(T) == typeid(float))  sgemv_ (&trans, &m, &n, &alpha, A.Data(), &ah, x.Data(), &one, &beta, &y[0], &one);
	else if (typeid(T) == typeid(cxfl))   cgemv_ (&trans, &m, &n, &alpha, A.Data(), &ah, x.Data(), &one, &beta, &y[0], &one);
	else if (typeid(T) == typeid(cxdb))   zgemv_ (&trans, &m, &n, &alpha, A.Data(), &ah, x.Data(), &one, &beta, &y[0], &one);
	
	return y;
	
}



#endif // __LAPACK_HPP__
