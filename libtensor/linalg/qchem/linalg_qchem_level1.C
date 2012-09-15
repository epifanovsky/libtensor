#include <cstdlib>
#include <qchem.h>
#include <libmathtools/general/blas_include.h>
#include "linalg_qchem_level1.h"

namespace libtensor {


const char *linalg_qchem_level1::k_clazz = "linalg";


void linalg_qchem_level1::add_i_i_x_x(
    size_t ni,
    const double *a, size_t sia, double ka,
    double b, double kb,
    double *c, size_t sic,
    double d) {

    start_timer("daxpy");
    CL_DAXPY(ni, d * ka, (double*)a, sia, c, sic);
    double db = d * kb * b;
    if(sic == 1) {
        for(size_t i = 0; i < ni; i++) c[i] += db;
    } else {
        for(size_t i = 0; i < ni; i++) c[i * sic] += db;
    }
    stop_timer("daxpy");
}


void linalg_qchem_level1::copy_i_i(
    size_t ni,
    const double *a, size_t sia,
    double *c, size_t sic) {

    if(sia == 1 && sic == 1) {
        start_timer("memcpy");
        memcpy(c, a, ni * sizeof(double));
        stop_timer("memcpy");
    } else {
        start_timer("dcopy");
        CL_DCOPY(ni, (double*)a, sia, c, sic);
        stop_timer("dcopy");
    }
}


void linalg_qchem_level1::mul1_i_x(
    size_t ni,
    double a,
    double *c, size_t sic) {

    start_timer("dscal");
    CL_DSCAL(ni, a, c, sic);
    stop_timer("dscal");
}


double linalg_qchem_level1::mul2_x_p_p(
    size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb) {

    start_timer("ddot");
    double d = CL_DDOT(np, (double*)a, spa, (double*)b, spb);
    stop_timer("ddot");
    return d;
}


void linalg_qchem_level1::mul2_i_i_x(
    size_t ni,
    const double *a, size_t sia,
    double b,
    double *c, size_t sic) {

    if(b == 1.0) {
        start_timer("nonblas");
        if(sia == 1) {
            if(sic == 1) {
                mul2_i_i_x_p11(ni, a, b, c);
            } else {
                mul2_i_i_x_pxx(ni, a, sia, b, c, sic);
            }
        } else {
            mul2_i_i_x_pxx(ni, a, sia, b, c, sic);
        }
        stop_timer("nonblas");
    } else if(b == -1.0) {
        start_timer("nonblas");
        if(sia == 1) {
            if(sic == 1) {
                mul2_i_i_x_m11(ni, a, b, c);
            } else {
                mul2_i_i_x_mxx(ni, a, sia, b, c, sic);
            }
        } else {
            mul2_i_i_x_mxx(ni, a, sia, b, c, sic);
        }
        stop_timer("nonblas");
    } else {
        start_timer("daxpy");
        CL_DAXPY(ni, b, (double*)a, sia, c, sic);
        stop_timer("daxpy");
    }
}


void linalg_qchem_level1::mul2_i_i_x_p11(size_t ni,
    const double *a, double b, double *c) {

    #pragma ivdep
    for(size_t i = 0; i < ni; i++) {
        c[i] += a[i];
    }
}


void linalg_qchem_level1::mul2_i_i_x_pxx(size_t ni,
    const double *a, size_t sia, double b, double *c, size_t sic) {

    for(size_t i = 0; i < ni; i++) {
        c[sic * i] += a[sia * i];
    }
}


void linalg_qchem_level1::mul2_i_i_x_m11(size_t ni,
    const double *a, double b, double *c) {

    #pragma ivdep
    for(size_t i = 0; i < ni; i++) {
        c[i] -= a[i];
    }
}


void linalg_qchem_level1::mul2_i_i_x_mxx(size_t ni,
    const double *a, size_t sia, double b, double *c, size_t sic) {

    for(size_t i = 0; i < ni; i++) {
        c[sic * i] -= a[sia * i];
    }
}


} // namespace libtensor
