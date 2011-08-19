extern "C" {
	
	// Cholesky factorization of a complex Hermitian positive definite matrix
	void cpotrf_   (char* uplo, int* n, void* a, int* lda, int *info);
	void dpotrf_   (char* uplo, int* n, void* a, int* lda, int *info);
	
	// Computes an LU factorization of a general M-by-N matrix A
	void cgetrf_   (int* m, int    *n, void   *a, int*   lda, int  *ipiv, int  *info);
	void dgetrf_   (int* m, int    *n, void   *a, int*   lda, int  *ipiv, int  *info);
	
	// Inverse of a complex Hermitian positive definite matrix using cpotrf/cpptrf
	void cpotri_   (char* uplo, int    *n, void   *a, int*   lda, int  *info);
	void dpotri_   (char* uplo, int    *n, void   *a, int*   lda, int  *info);
	
	// Genral matrix inversion through cholesky decomposition
	void cgetri_   (int     *n, void   *a, int*   lda, int   *ipiv, void   *work, int   *lwork, int  *info);
	void dgetri_   (int     *n, void   *a, int*   lda, int   *ipiv, void   *work, int   *lwork, int  *info);
	
	// Eigen value computations
	void cgeev_    (char *jobvl, char *jobvr, int    *n, void    *a, int   *lda, void      *w,             void   *vl,
				    int   *ldvl, void    *vr, int *ldvr, void *work, int *lwork, float *rwork, int  *info);
	void dgeev_    (char *jobvl, char *jobvr, int    *n, void    *a, int   *lda, void     *wr, void   *wi, void   *vl, 
				    int   *ldvl, void    *vr, int *ldvr, void *work, int *lwork,               int  *info);
	
	// Singular value decomposition 
	void cgesdd_   (const char *jobz, int    *m, int     *n, void     *a, int     *lda, float     *s, void    *u, int  *ldu, 
				    void   *vt, int *ldvt, void *work, int  *lwork, float *rwork, int   *iwork, int  *info);
	void dgesdd_   (const char *jobz, int    *m, int     *n, void     *a, int     *lda, void      *s, void    *u, int  *ldu, 
				    void   *vt, int *ldvt, void *work, int  *lwork,               int   *iwork, int  *info);

	// Minimize L2-norm (|b-A*x|) - b = id delivers pinv
	void cgels_    (char *trans, int     *m, int    *n, int  *nrhs, void     *a, int     *lda, void      *b, int  *ldb,
					void  *work, int *lwork, int *info);
	void dgels_    (char *trans, int     *m, int    *n, int  *nrhs, void     *a, int     *lda, void      *b, int  *ldb,
					void  *work, int *lwork, int *info);
	
	void dgelss_   (int *m, int *n, int *nrhs,
					void *a, int *lda, void *b, int *ldb, void *
					s, void *rcond, int *rank, void *work, int *lwork,
					int *info);

	void cgelss_(int *m, int *n, int *nrhs,
				 void *a, int *lda, void *b, int *ldb, void *
				 s, void *rcond, int *rank, void *work, int *lwork,
				 void* rwork, int *info);

}