#ifndef LIBTENSOR_LAPACK_GENERIC_H
#define LIBTENSOR_LAPACK_GENERIC_H


extern "C" {
	int dgesv_(int*, int*, double*, int*, int*, double*, int*, int*);
	int dgesvd_(char*, char*, int*, int*, double*, int*, double*, double*,
		int*, double*, int*, double*, int*, int*);
	int dsyev_(char*, char*, int*, double*, int*, double*, double*, int*,
		int*);
	int dgeev_(char*, char*, int*, double*, int*, double*, double*, double*,
		int*, double*, int*, double*, int*, int*);
}


namespace libtensor {


/**	\brief LAPACK function dgesv (generic)

	\ingroup libtensor_linalg
 **/
inline int lapack_dgesv(size_t n, size_t nrhs, double *a, size_t lda,
	int *ipiv, double *b, size_t ldb) {

	int gen_n = n;
	int gen_nrhs = nrhs;
	int gen_lda = lda;
	int gen_ldb = ldb;
	int gen_info = 0;
	dgesv_(&gen_n, &gen_nrhs, a, &gen_lda, ipiv, b, &gen_ldb, &gen_info);
	return gen_info;
}


/**	\brief LAPACK function dgesvd (generic)

	\ingroup libtensor_linalg
 **/
inline int lapack_dgesvd(char jobu, char jobvt, size_t m, size_t n, double *a,
	size_t lda, double *s, double *u, size_t ldu, double *vt,
	size_t ldvt, double *work, size_t lwork) {

	int gen_m = m;
	int gen_n = n;
	int gen_lda = lda;
	int gen_ldu = ldu;
	int gen_ldvt = ldvt;
	int gen_lwork = lwork;
	int gen_info = 0;
	dgesvd_(&jobu, &jobvt, &gen_m, &gen_n, a, &gen_lda, s, u, &gen_ldu,
		vt, &gen_ldvt, work, &gen_lwork, &gen_info);
	return gen_info;
}


/**	\brief LAPACK function dsyev (generic)

	\ingroup libtensor_linalg
 **/
inline int lapack_dsyev(char jobz, char uplo, size_t n, double *a, size_t lda,
	double *w, double *work, size_t lwork) {

	int gen_n = n;
	int gen_lda = lda;
	int gen_lwork = lwork;
	int gen_info = 0;
	dsyev_(&jobz, &uplo, &gen_n, a, &gen_lda, w, work, &gen_lwork,
		&gen_info);
	return gen_info;
}


/**	\brief LAPACK function dgeev (generic)

	\ingroup libtensor_linalg
 **/
inline int lapack_dgeev(char jobvl, char jobvr, size_t n, double *a, size_t lda,
	double *wr, double *wi, double *vl, size_t ldvl, double *vr,
	size_t ldvr, double *work, size_t lwork) {

	int gen_n = n;
	int gen_lda = lda;
	int gen_ldvl = ldvl;
	int gen_ldvr = ldvr;
	int gen_lwork = lwork;
	int gen_info = 0;
	dgeev_(&jobvl, &jobvr, &gen_n, a, &gen_lda, wr, wi, vl, &gen_ldvl,
		vr, &gen_ldvr, work, &gen_lwork, &gen_info);
	return gen_info;
}


} // namespace libtensor

#endif // LIBTENSOR_LAPACK_GENERIC_H
