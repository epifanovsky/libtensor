#ifndef LIBTENSOR_LAPACK_GENERIC_H
#define LIBTENSOR_LAPACK_GENERIC_H

#include <complex>

#ifdef USE_QCHEM

#include <qchem.h>

#else

extern "C" {
#define dgesv dgesv_
    int dgesv_(int*, int*, double*, int*, int*, double*, int*, int*);
#define dgesvd dgesvd_
    int dgesvd_(char*, char*, int*, int*, double*, int*, double*, double*,
        int*, double*, int*, double*, int*, int*);
#define zgesvd zgesvd_
    int zgesvd_(char*, char*, int*, int*, std::complex<double>*, int*,
        double*, std::complex<double>*, int*, std::complex<double>*, int*,
        std::complex<double>*, int*, double*, int*);
#define dgeev dgeev_
    int dgeev_(char*, char*, int*, double*, int*, double*, double*, double*,
        int*, double*, int*, double*, int*, int*);
#define dggev dggev_
    int dggev_(char*, char*, int*, double*, int*, double* , int *, double*, double*, double*,
        double *, int*, double*, int*, double*, int*, int*);
#define zgeev zgeev_
    int zgeev_(char*, char*, int*, std::complex<double>*, int*,
        std::complex<double>*, std::complex<double>*, int*,
        std::complex<double>*, int*, std::complex<double>*, int*, double*,
        int*);
#define dgelss dgelss_
    int dgelss_(int*, int*, int*, double*, int*, double*, int*, double*,
        double*, int*, double*, int*, int*);
#define dgelsd dgelsd_
    int dgelsd_(int*, int*, int*, double*, int*, double*, int*, double*,
        double*, int*, double*, int*, int*, int*);
#define dlarnv dlarnv_
    int dlarnv_(int*, int*, int*, double*);
#define dpotrf dpotrf_
    int dpotrf_(char*, int*, double*, int*, int*);
#define dpotri dpotri_
    int dpotri_(char*, int*, double*, int*, int*);
#define dpstrf dpstrf_
    int dpstrf_(char*, int*, double*, int*, int*, int*, double*, double*, int*);
#define dpteqr dpteqr_
    int dpteqr_(char*, int*, double*, double*, double*, int*, double*, int*);
#define dsteqr dsteqr_
    int dsteqr_(char*, int*, double*, double*, double*, int*, double*, int*);
}

#endif // USE_QCHEM

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
    dgesv(&gen_n, &gen_nrhs, a, &gen_lda, ipiv, b, &gen_ldb, &gen_info);
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
    dgesvd(&jobu, &jobvt, &gen_m, &gen_n, a, &gen_lda, s, u, &gen_ldu,
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

    zgesvd(&jobu, &jobvt, &gen_m, &gen_n, a, &gen_lda, s, u, &gen_ldu,
        vt, &gen_ldvt, work, &gen_lwork, rwork, &gen_info);
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
    dgeev(&jobvl, &jobvr, &gen_n, a, &gen_lda, wr, wi, vl, &gen_ldvl,
        vr, &gen_ldvr, work, &gen_lwork, &gen_info);
    return gen_info;
}


/** \brief LAPACK function dggev (generic)
 *
 *     \ingroup libtensor_linalg
 *      **/
inline int lapack_dggev(char jobvl, char jobvr, size_t n, double *a,
    size_t lda, double * b, size_t ldb, double *alphar, double *alphai, double * beta, double *vl, size_t ldvl, double *vr,
    size_t ldvr, double *work, size_t lwork) {

    int gen_n = n;
    int gen_lda = lda;
    int gen_ldb = ldb;
    int gen_ldvl = ldvl;
    int gen_ldvr = ldvr;
    int gen_lwork = lwork;
    int gen_info = 0;
    dggev(&jobvl, &jobvr, &gen_n, a, &gen_lda, b, &gen_ldb, alphar, alphai, beta, vl, &gen_ldvl, vr, &gen_ldvr, work, &gen_lwork, &gen_info);
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
    zgeev(&jobvl, &jobvr, &gen_n, a, &gen_lda, w, vl, &gen_ldvl, vr, &gen_ldvr,
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
    dgelss(&gen_m, &gen_n, &gen_nrhs, B, &gen_lda, rhs, &gen_ldb, S,
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
    dgelsd(&gen_m, &gen_n, &gen_nrhs, B, &gen_lda, rhs, &gen_ldb, S,
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
    dpotrf(&uplo, &gen_n, a, &gen_lda, &gen_info);
    return gen_info;
}


/** \brief LAPACK function dpotri (generic)

    \ingroup libtensor_linalg
 **/
inline int lapack_dpotri(char uplo, size_t n, double *a, size_t lda) {

    int gen_n = n;
    int gen_lda = lda;
    int gen_info = 0;
    dpotri(&uplo, &gen_n, a, &gen_lda, &gen_info);
    return gen_info;
}


/** \brief LAPACK function dlarnv (generic)

    \ingroup libtensor_linalg
 **/
inline int lapack_dlarnv(size_t idist, int *iseed, size_t n, double *x) {

    int gen_idist = idist;
    int gen_n = n;
    dlarnv(&gen_idist, iseed, &gen_n, x);
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
    //dpstrf(&uplo, &gen_n, a, &gen_lda, p, rank, &gen_tol, work, &gen_info);
    return gen_info;
}


/** \brief LAPACK function dpteqr (generic)

    \ingroup libtensor_linalg
 **/
inline int lapack_dpteqr(char compz, size_t n, double *d, double *e, double *z,
    size_t ldz, double *work) {

    int gen_n = n;
    int gen_ldz = ldz;
    int gen_info = 0;
    dpteqr(&compz, &gen_n, d, e, z, &gen_ldz, work, &gen_info);
    return gen_info;
}


/** \brief LAPACK function dsteqr (generic)

    \ingroup libtensor_linalg
 **/
inline int lapack_dsteqr(char compz, size_t n, double *d, double *e, double *z,
    size_t ldz, double *work) {

    int gen_n = n;
    int gen_ldz = ldz;
    int gen_info = 0;
    dsteqr(&compz, &gen_n, d, e, z, &gen_ldz, work, &gen_info);
    return gen_info;
}


} // namespace libtensor

#endif // LIBTENSOR_LAPACK_GENERIC_H
