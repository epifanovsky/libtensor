#include <libtensor/not_implemented.h>
#include "acml_h.h"
#include "../lapack_acml.h"


namespace libtensor {


int lapack_dgesv(size_t n, size_t nrhs, double *a, size_t lda, int *ipiv,
    double *b, size_t ldb) {

    int info = 0;
    dgesv(n, nrhs, a, lda, ipiv, b, ldb, &info);
    return info;
}


int lapack_dgesvd(char jobu, char jobvt, size_t m, size_t n, double *a,
    size_t lda, double *s, double *u, size_t ldu, double *vt, size_t ldvt,
    double *work, size_t lwork) {

    int info = 0;
    dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, &info);
    return info;
}


int lapack_zgesvd(char jobu, char jobvt, size_t m, size_t n,
    std::complex<double> *a, size_t lda, double *s, std::complex<double> *u,
    size_t ldu, std::complex<double> *vt, size_t ldvt,
    std::complex<double> *work, size_t lwork, double *rwork) {

    int info = 0;
    zgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, &info);
    return info;
}


int lapack_dsyev(char jobz, char uplo, size_t n, double *a, size_t lda,
    double *w, double *work, size_t lwork) {

    int info = 0;
    dsyev(jobz, uplo, n, a, lda, w, &info);
    return info;
}


int lapack_dgeev(char jobvl, char jobvr, size_t n, double *a,
    size_t lda, double *wr, double *wi, double *vl, size_t ldvl, double *vr,
    size_t ldvr, double *work, size_t lwork) {

    int info = 0;
    dgeev(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, &info);
    return info;
}


int lapack_zgeev(char jobvl, char jobvr, size_t n,
    std::complex<double> *a, size_t lda, std::complex<double> *w,
    std::complex<double> *vl, size_t ldvl, std::complex<double> *vr,
    size_t ldvr, std::complex<double> *work, size_t lwork, double *rwork) {

    int info = 0;
    zgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, &info);
    return info;
}


int lapack_dgelss(size_t m, size_t n, size_t nrhs, double *B, size_t lda,
    double *rhs, size_t ldb, double *S, double rcond, int *rank, double *work,
    size_t lwork) {

    int info = 0;
    dgelss(m, n, nrhs, B, lda, rhs, ldb, S, rcond, rank, &info);
    return info;
}


int lapack_dgelsd(size_t m, size_t n, size_t nrhs, double *B, size_t lda,
    double *rhs, size_t ldb, double *S, double rcond, int *rank, double *work,
    size_t lwork, int *iwork) {

    int info = 0;
    dgelsd(m, n, nrhs, B, lda, rhs, ldb, S, rcond, rank, &info);
    return info;
}


int lapack_dpotrf(char uplo, size_t n, double *a, size_t lda) {

    int info = 0;
    dpotrf(uplo, n, a, lda, &info);
    return info;
}


int lapack_dlarnv(size_t idist, int *iseed, size_t n, double *x) {

    throw not_implemented(g_ns, 0, "lapack_dlarnv", __FILE__, __LINE__);
    //dlarnv(idist, iseed, n, x);
    return 0;
}


int lapack_dpstrf(char uplo, size_t n, double *a, size_t lda, int *p,
    int *rank, double tol, double *work) {

    int info = 0;
#ifdef HAVE_LAPACK_DPSTRF
    dpstrf(uplo, n, a, lda, p, rank, tol, work, &info);
#else // HAVE_LAPACK_DPSTRF
    throw not_implemented(g_ns, 0, "lapack_dpstrf", __FILE__, __LINE__);
#endif // HAVE_LAPACK_DPSTRF

    return info;
}


int lapack_dpteqr(char compz, size_t n, double *d, double *e, double *z,
    size_t ldz, double *work) {

    int info = 0;
    dpteqr(compz, n, d, e, z, ldz, &info);

    return info;
}


int lapack_dsteqr(char compz, size_t n, double *d, double *e, double *z,
    size_t ldz, double *work) {

    int info = 0;
    dsteqr(compz, n, d, e, z, ldz, &info);

    return info;
}


} // namespace libtensor
