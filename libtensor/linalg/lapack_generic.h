#ifndef LIBTENSOR_LAPACK_GENERIC_H
#define LIBTENSOR_LAPACK_GENERIC_H

#include <complex>

#ifdef USE_QCHEM
#include <qchem.h>
#endif // USE_QCHEM

extern "C" {
    int dgesv_(int*, int*, double*, int*, int*, double*, int*, int*);
    int dgesvd_(char*, char*, int*, int*, double*, int*, double*, double*,
        int*, double*, int*, double*, int*, int*);
    int zgesvd_(char*, char*, int*, int*, std::complex<double>*, int*,
        double*, std::complex<double>*, int*, std::complex<double>*, int*,
        std::complex<double>*, int*, double*, int*);
    int dsyev_(char*, char*, int*, double*, int*, double*, double*, int*, int*);
    int dgeev_(char*, char*, int*, double*, int*, double*, double*, double*,
        int*, double*, int*, double*, int*, int*);
    int zgeev_(char*, char*, int*, std::complex<double>*, int*,
        std::complex<double>*, std::complex<double>*, int*,
        std::complex<double>*, int*, std::complex<double>*, int*, double*,
        int*);
    int dgelss_(int*, int*, int*, double*, int*, double*, int*, double*,
        double*, int*, double*, int*, int*);
    int dgelsd_(int*, int*, int*, double*, int*, double*, int*, double*,
        double*, int*, double*, int*, int*, int*);
    int dlarnv_(int*, int*, int*, double*);
#ifndef USE_QCHEM
    int dpotrf_(char*, int*, double*, int*, int*);
#endif // USE_QCHEM
    int dpstrf_(char*, int*, double*, int*, int*, int*, double*, double*, int*);
}


namespace libtensor {


/** \brief LAPACK function dgesv (generic)

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


/** \brief LAPACK function dgesvd (generic)

    \ingroup libtensor_linalg
 **/
inline int lapack_dgesvd(char jobu, char jobvt, size_t m, size_t n, double *a,
    size_t lda, double *s, double *u, size_t ldu, double *vt, size_t ldvt,
    double *work, size_t lwork) {

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

/** \brief LAPACK function zgesvd (generic)

    \ingroup libtensor_linalg
 **/
inline int lapack_zgesvd(char jobu, char jobvt, size_t m, size_t n,
    std::complex<double> *a, size_t lda, double *s, std::complex<double> *u,
    size_t ldu, std::complex<double> *vt, size_t ldvt,
    std::complex<double> *work, size_t lwork, double *rwork) {
	
    int gen_m = m;
    int gen_n = n;
    int gen_lda = lda;
    int gen_ldu = ldu;
    int gen_ldvt = ldvt;
    int gen_lwork = lwork;
    int gen_info = 0;

    zgesvd_(&jobu, &jobvt, &gen_m, &gen_n, a, &gen_lda, s, u, &gen_ldu,
        vt, &gen_ldvt, work, &gen_lwork, rwork, &gen_info);
    return gen_info;
}


/** \brief LAPACK function dsyev (generic)

    \ingroup libtensor_linalg
 **/
inline int lapack_dsyev(char jobz, char uplo, size_t n, double *a, size_t lda,
    double *w, double *work, size_t lwork) {

    int gen_n = n;
    int gen_lda = lda;
    int gen_lwork = lwork;
    int gen_info = 0;
    dsyev_(&jobz, &uplo, &gen_n, a, &gen_lda, w, work, &gen_lwork, &gen_info);
    return gen_info;
}


/** \brief LAPACK function dgeev (generic)

    \ingroup libtensor_linalg
 **/
inline int lapack_dgeev(char jobvl, char jobvr, size_t n, double *a, size_t lda,
    double *wr, double *wi, double *vl, size_t ldvl, double *vr, size_t ldvr,
    double *work, size_t lwork) {

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


/** \brief LAPACK function zgeev (generic)

    \ingroup libtensor_linalg
 **/
inline int lapack_zgeev(char jobvl, char jobvr, size_t n,
    std::complex<double> *a, size_t lda, std::complex<double> *w,
    std::complex<double> *vl, size_t ldvl, std::complex<double> *vr,
    size_t ldvr, std::complex<double> *work, size_t lwork, double *rwork) {   

    int gen_n = n;
    int gen_lda = lda;
    int gen_ldvl = ldvl;
    int gen_ldvr = ldvr;
    int gen_lwork = lwork;
    int gen_info = 0;
    zgeev_(&jobvl, &jobvr, &gen_n, a, &gen_lda, w, vl, &gen_ldvl, vr, &gen_ldvr,
        work, &gen_lwork, rwork, &gen_info);     
    return gen_info;
}


/** \brief LAPACK function dgelss (generic)

    \ingroup libtensor_linalg
 **/
inline int lapack_dgelss(size_t m, size_t n, size_t nrhs, double *B, size_t lda,
    double *rhs, size_t ldb, double *S, double rcond, int *rank, double *work,
    size_t lwork) {

    int gen_m = m;
    int gen_n = n;
    int gen_nrhs = nrhs;
    int gen_lda = lda;
    int gen_ldb = ldb;
    double gen_rcond = rcond;
    int gen_lwork = lwork;
    int gen_info = 0;
    dgelss_(&gen_m, &gen_n, &gen_nrhs, B, &gen_lda, rhs, &gen_ldb, S,
        &gen_rcond, rank, work, &gen_lwork, &gen_info);
    return gen_info;
}


/** \brief LAPACK function dgelsd (generic)

    \ingroup libtensor_linalg
 **/
inline int lapack_dgelsd(size_t m, size_t n, size_t nrhs, double *B, size_t lda,
    double *rhs, size_t ldb, double *S, double rcond, int *rank, double *work,
    size_t lwork, int *iwork) {

    int gen_m = m;
    int gen_n = n;
    int gen_nrhs = nrhs;
    int gen_lda = lda;
    int gen_ldb = ldb;
    double gen_rcond = rcond;
    int gen_lwork = lwork;
    int gen_info = 0;
    dgelsd_(&gen_m, &gen_n, &gen_nrhs, B, &gen_lda, rhs, &gen_ldb, S,
        &gen_rcond, rank, work, &gen_lwork, iwork, &gen_info);
    return gen_info;
}


/** \brief LAPACK function dpotrf (generic)

    \ingroup libtensor_linalg
 **/
inline int lapack_dpotrf(char uplo, size_t n, double *a, size_t lda) {

    int gen_n = n;
    int gen_lda = lda;
    int gen_info = 0;
    dpotrf_(&uplo, &gen_n, a, &gen_lda, &gen_info);
    return gen_info;
}


/** \brief LAPACK function dlarnv (generic)

    \ingroup libtensor_linalg
 **/
inline int lapack_dlarnv(size_t idist, int *iseed, size_t n, double *x) {

    int gen_idist = idist;
    int gen_n = n;
    dlarnv_(&gen_idist, iseed, &gen_n, x);
    return 0;
}


/** \brief LAPACK function dpstrf (generic)

    \ingroup libtensor_linalg
 **/
inline int lapack_dpstrf(char uplo, size_t n, double *a, size_t lda, int *p,
    int *rank, double tol, double *work) {

    int gen_n = n;
    int gen_lda = lda;
    double gen_tol = tol;
    int gen_info = 0;
    dpstrf_(&uplo, &gen_n, a, &gen_lda, p, rank, &gen_tol, work, &gen_info);
    return gen_info;
}


} // namespace libtensor

#endif // LIBTENSOR_LAPACK_GENERIC_H
