#include <mkl.h>
#ifdef HAVE_MKL_DOMATCOPY
#include <mkl_trans.h>
#endif // HAVE_MKL_DOMATCOPY
#include "linalg_base_level2_mkl.h"

namespace libtensor {


const char *linalg_base_level2_mkl::k_clazz = "mkl";


void linalg_base_level2_mkl::i_ip_p_x(
    size_t ni, size_t np,
    const double *a, size_t sia,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    start_timer("dgemv");
    cblas_dgemv(CblasRowMajor, CblasNoTrans, ni, np, d, a, sia, b, spb, 1.0,
        c, sic);
    stop_timer("dgemv");
}


void linalg_base_level2_mkl::i_pi_p_x(
    size_t ni, size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    start_timer("dgemv");
    cblas_dgemv(CblasRowMajor, CblasTrans, np, ni, d, a, spa, b, spb, 1.0,
        c, sic);
    stop_timer("dgemv");
}


void linalg_base_level2_mkl::ij_i_j_x(
    size_t ni, size_t nj,
    const double *a, size_t sia,
    const double *b, size_t sjb,
    double *c, size_t sic,
    double d) {

    start_timer("dger");
    cblas_dger(CblasRowMajor, ni, nj, d, a, sia, b, sjb, c, sic);
    stop_timer("dger");
}


void linalg_base_level2_mkl::ij_ji(
    size_t ni, size_t nj,
    const double *a, size_t sja,
    double *c, size_t sic) {

#ifdef HAVE_MKL_DOMATCOPY
    start_timer("mkl_domatcopy");
    mkl_domatcopy('R', 'T', nj, ni, 1.0, a, sja, c, sic);
    stop_timer("mkl_domatcopy");
#else // HAVE_MKL_DOMATCOPY
    start_timer("dcopy");
    if(ni < nj) {
        double *c1 = c;
        for(size_t i = 0; i < ni; i++, c1 += sic) {
            cblas_dcopy(nj, a + i, sja, c1, 1);
        }
    } else {
        const double *a1 = a;
        for(size_t j = 0; j < nj; j++, a1 += sja) {
            cblas_dcopy(ni, a1, 1, c + j, sic);
        }
    }
    stop_timer("dcopy");
#endif // HAVE_MKL_DOMATCOPY
}


} // namespace libtensor

