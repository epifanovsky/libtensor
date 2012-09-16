#include <cstring>
#include <mkl.h>
#ifdef HAVE_MKL_VML
#include <mkl_vml_functions.h>
#endif // HAVE_MKL_VML
#include "linalg_mkl_level1.h"

namespace libtensor {


const char *linalg_mkl_level1::k_clazz = "mkl";


void linalg_mkl_level1::add_i_i_x_x(
    void*,
    size_t ni,
    const double *a, size_t sia, double ka,
    double b, double kb,
    double *c, size_t sic,
    double d) {

    start_timer("daxpy");
    cblas_daxpy(ni, d * ka, a, sia, c, sic);
    double db = d * kb * b;
    if(sic == 1) {
        for(size_t i = 0; i < ni; i++) c[i] += db;
    } else {
        for(size_t i = 0; i < ni; i++) c[i * sic] += db;
    }
    stop_timer("daxpy");
}


void linalg_mkl_level1::copy_i_i(
    void*,
    size_t ni,
    const double *a, size_t sia,
    double *c, size_t sic) {

    if(sia == 1 && sic == 1) {
        start_timer("memcpy");
        ::memcpy(c, a, ni * sizeof(double));
        stop_timer("memcpy");
    } else {
        start_timer("dcopy");
        cblas_dcopy(ni, a, sia, c, sic);
        stop_timer("dcopy");
    }
}


void linalg_mkl_level1::mul1_i_x(
    void*,
    size_t ni,
    double a,
    double *c, size_t sic) {

    start_timer("dscal");
    cblas_dscal(ni, a, c, sic);
    stop_timer("dscal");
}


double linalg_mkl_level1::mul2_x_p_p(
    void*,
    size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb) {

    start_timer("ddot");
    double d = cblas_ddot(np, a, spa, b, spb);
    stop_timer("ddot");
    return d;
}


void linalg_mkl_level1::mul2_i_i_x(
    void*,
    size_t ni,
    const double *a, size_t sia,
    double b,
    double *c, size_t sic) {

    start_timer("daxpy");
    cblas_daxpy(ni, b, a, sia, c, sic);
    stop_timer("daxpy");
}


void linalg_mkl_level1::mul2_i_i_i_x(
    void*,
    size_t ni,
    const double *a, size_t sia,
    const double *b, size_t sib,
    double *c, size_t sic,
    double d) {

#if defined(HAVE_MKL_VML)
    if(sia == 1 && sib == 1) {
        start_timer("vdmul+daxpy");
        double buf[256];
        size_t len = 256;
        while(ni > 0) {
            if(ni < len) len = ni;
            vdMul(len, a, b, buf);
            cblas_daxpy(len, d, buf, 1, c, sic);
            ni -= len;
            a += len;
            b += len;
            c += len * sic;
        }
        stop_timer("vdmul+daxpy");
    } else
#endif
    {
        start_timer("nonblas");
        for(size_t i = 0; i < ni; i++) {
            c[i * sic] += d * a[i * sia] * b[i * sib];
        }
        stop_timer("nonblas");
    }
}


} // namespace libtensor

