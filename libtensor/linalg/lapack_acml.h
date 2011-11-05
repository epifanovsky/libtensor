#ifndef LIBTENSOR_LAPACK_ACML_H
#define LIBTENSOR_LAPACK_ACML_H

#include "acml/acml_h.h"


namespace libtensor {


/**	\brief LAPACK function dgesv (ACML)

	\ingroup libtensor_linalg
 **/
inline int lapack_dgesv(size_t n, size_t nrhs, double *a, size_t lda,
	int *ipiv, double *b, size_t ldb) {

	int info = 0;
	dgesv(n, nrhs, a, lda, ipiv, b, ldb, &info);
	return info;
}


/**	\brief LAPACK function dgesvd (ACML)

	\ingroup libtensor_linalg
 **/
inline int lapack_dgesvd(char jobu, char jobvt, size_t m, size_t n, double *a,
	size_t lda, double *s, double *u, size_t ldu, double *vt,
	size_t ldvt, double *work, size_t lwork) {

	int info = 0;
	dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, &info);
	return info;
}


/**     \brief LAPACK function zgesvd (ACML)

        \ingroup libtensor_linalg
 **/
inline int lapack_zgesvd(char jobu, char jobvt, size_t m, size_t n,
	std::complex<double> *a, size_t lda, double *s,
	std::complex<double> *u, size_t ldu, std::complex<double> *vt,
	size_t ldvt, std::complex<double> *work, size_t lwork, double *rwork) {

        int info = 0;
        zgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, &info);
        return info;
}


/**	\brief LAPACK function dsyev (ACML)

	\ingroup libtensor_linalg
 **/
inline int lapack_dsyev(char jobz, char uplo, size_t n, double *a, size_t lda,
	double *w, double *work, size_t lwork) {

	int info = 0;
	dsyev(jobz, uplo, n, a, lda, w, &info);
	return info;
}


/**	\brief LAPACK function dgeev (ACML)

	\ingroup libtensor_linalg
 **/
inline int lapack_dgeev(char jobvl, char jobvr, size_t n, double *a, size_t lda,
	double *wr, double *wi, double *vl, size_t ldvl, double *vr,
	size_t ldvr, double *work, size_t lwork) {

	int info = 0;
	dgeev(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, &info);
	return info;
}

/**     \brief LAPACK function zgeev (ACML)

        \ingroup libtensor_linalg
 **/
inline int lapack_zgeev(char jobvl, char jobvr, size_t n, std::complex <double> *a, size_t lda,
        std::complex <double> *w, std::complex <double> *vl, size_t ldvl, std::complex <double> *vr,
        size_t ldvr, std::complex <double> *work, size_t lwork, double *rwork) {

        int info = 0;
        zgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, &info);
        return info;
}

inline int lapack_dpotrf(char uplo, size_t n, double *a, size_t lda) {
        int info = 0;
        dpotrf(uplo, n, a, lda, &info);
        return info;
}

inline int lapack_dpstrf(char uplo, size_t n, double *a, size_t lda, int *p, int *rank, double tol, double *work) {
        int info = 0;
        dpstrf(uplo, n, a, lda, p, rank, tol, work, &info);
        return info;
}

} // namespace libtensor

#endif // LIBTENSOR_LAPACK_ACML_H
