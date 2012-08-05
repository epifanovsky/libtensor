#ifndef LIBTENSOR_LAPACK_ACML_H
#define LIBTENSOR_LAPACK_ACML_H

namespace libtensor {


/** \brief LAPACK function dgesv (ACML)

    \ingroup libtensor_linalg
 **/
int lapack_dgesv(size_t n, size_t nrhs, double *a, size_t lda, int *ipiv,
    double *b, size_t ldb);


/** \brief LAPACK function dgesvd (ACML)

    \ingroup libtensor_linalg
 **/
int lapack_dgesvd(char jobu, char jobvt, size_t m, size_t n, double *a,
    size_t lda, double *s, double *u, size_t ldu, double *vt, size_t ldvt,
    double *work, size_t lwork);


/** \brief LAPACK function zgesvd (ACML)

    \ingroup libtensor_linalg
 **/
int lapack_zgesvd(char jobu, char jobvt, size_t m, size_t n,
    std::complex<double> *a, size_t lda, double *s, std::complex<double> *u,
    size_t ldu, std::complex<double> *vt, size_t ldvt,
    std::complex<double> *work, size_t lwork, double *rwork);


/** \brief LAPACK function dsyev (ACML)

    \ingroup libtensor_linalg
 **/
int lapack_dsyev(char jobz, char uplo, size_t n, double *a, size_t lda,
    double *w, double *work, size_t lwork);


/** \brief LAPACK function dgeev (ACML)

    \ingroup libtensor_linalg
 **/
int lapack_dgeev(char jobvl, char jobvr, size_t n, double *a,
    size_t lda, double *wr, double *wi, double *vl, size_t ldvl, double *vr,
    size_t ldvr, double *work, size_t lwork);


/** \brief LAPACK function zgeev (ACML)

    \ingroup libtensor_linalg
 **/
int lapack_zgeev(char jobvl, char jobvr, size_t n,
    std::complex<double> *a, size_t lda, std::complex<double> *w,
    std::complex<double> *vl, size_t ldvl, std::complex<double> *vr,
    size_t ldvr, std::complex<double> *work, size_t lwork, double *rwork);


/** \brief LAPACK function dgelss (ACML)

    \ingroup libtensor_linalg
 **/
int lapack_dgelss(size_t m, size_t n, size_t nrhs, double *B, size_t lda,
    double *rhs, size_t ldb, double *S, double rcond, int *rank, double *work,
    size_t lwork);


/** \brief LAPACK function dgelsd (ACML)

    \ingroup libtensor_linalg
 **/
int lapack_dgelsd(size_t m, size_t n, size_t nrhs, double *B, size_t lda,
    double *rhs, size_t ldb, double *S, double rcond, int *rank, double *work,
    size_t lwork, int *iwork);


/** \brief LAPACK function dpotrf (ACML)

    \ingroup libtensor_linalg
 **/
int lapack_dpotrf(char uplo, size_t n, double *a, size_t lda);


/** \brief LAPACK function dlarnv (ACML)

    \ingroup libtensor_linalg
 **/
int lapack_dlarnv(size_t idist, int *iseed, size_t n, double *x);


/** \brief LAPACK function dpotrf (ACML)

    \ingroup libtensor_linalg
 **/
int lapack_dpstrf(char uplo, size_t n, double *a, size_t lda, int *p,
    int *rank, double tol, double *work);


} // namespace libtensor

#endif // LIBTENSOR_LAPACK_ACML_H
