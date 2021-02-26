#include <cstdlib>
#include <liblas/liblas.h>
#include "linalg_qchem_level1_double.h"

namespace libtensor {


const char linalg_qchem_level1_double::k_clazz[] = "linalg";


void linalg_qchem_level1_double::add_i_i_x_x(
    void*,
    size_t ni,
    const double *a, size_t sia, double ka,
    double b, double kb,
    double *c, size_t sic,
    double d) {

    timings_base::start_timer("daxpy");
    INTEGER ni_ = ni, sia_ = sia, sic_ = sic;
    double d_ = d * ka;
    daxpy(&ni_, &d_, (double*)a, &sia_, c, &sic_);
    double db = d * kb * b;
    if(sic == 1) {
        for(size_t i = 0; i < ni; i++) c[i] += db;
    } else {
        for(size_t i = 0; i < ni; i++) c[i * sic] += db;
    }
    timings_base::stop_timer("daxpy");
}


void linalg_qchem_level1_double::copy_i_i(
    void*,
    size_t ni,
    const double *a, size_t sia,
    double *c, size_t sic) {

    if(sia == 1 && sic == 1) {
        timings_base::start_timer("memcpy");
        memcpy(c, a, ni * sizeof(double));
        timings_base::stop_timer("memcpy");
    } else {
        timings_base::start_timer("dcopy");
        INTEGER ni_ = ni, sia_ = sia, sic_ = sic;
        dcopy(&ni_, (double*)a, &sia_, c, &sic_);
        timings_base::stop_timer("dcopy");
    }
}


void linalg_qchem_level1_double::mul1_i_x(
    void*,
    size_t ni,
    double a,
    double *c, size_t sic) {

    timings_base::start_timer("dscal");
    INTEGER ni_ = ni, sic_ = sic;
    dscal(&ni_, &a, c, &sic_);
    timings_base::stop_timer("dscal");
}


double linalg_qchem_level1_double::mul2_x_p_p(
    void*,
    size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb) {

    timings_base::start_timer("ddot");
    INTEGER np_ = np, spa_ = spa, spb_ = spb;
    double d = ddot(&np_, (double*)a, &spa_, (double*)b, &spb_);
    timings_base::stop_timer("ddot");
    return d;
}


void linalg_qchem_level1_double::mul2_i_i_x(
    void*,
    size_t ni,
    const double *a, size_t sia,
    double b,
    double *c, size_t sic) {

    if(b == 1.0) {
        timings_base::start_timer("nonblas");
        if(sia == 1) {
            if(sic == 1) {
                mul2_i_i_x_p11(ni, a, b, c);
            } else {
                mul2_i_i_x_pxx(ni, a, sia, b, c, sic);
            }
        } else {
            mul2_i_i_x_pxx(ni, a, sia, b, c, sic);
        }
        timings_base::stop_timer("nonblas");
    } else if(b == -1.0) {
        timings_base::start_timer("nonblas");
        if(sia == 1) {
            if(sic == 1) {
                mul2_i_i_x_m11(ni, a, b, c);
            } else {
                mul2_i_i_x_mxx(ni, a, sia, b, c, sic);
            }
        } else {
            mul2_i_i_x_mxx(ni, a, sia, b, c, sic);
        }
        timings_base::stop_timer("nonblas");
    } else {
        timings_base::start_timer("daxpy");
        int ni_ = ni, sia_ = sia, sic_ = sic;
        daxpy(&ni_, &b, (double*)a, &sia_, c, &sic_);
        timings_base::stop_timer("daxpy");
    }
}


void linalg_qchem_level1_double::mul2_i_i_x_p11(size_t ni,
    const double *a, double b, double *c) {

    #pragma ivdep
    for(size_t i = 0; i < ni; i++) {
        c[i] += a[i];
    }
}


void linalg_qchem_level1_double::mul2_i_i_x_pxx(size_t ni,
    const double *a, size_t sia, double b, double *c, size_t sic) {

    for(size_t i = 0; i < ni; i++) {
        c[sic * i] += a[sia * i];
    }
}


void linalg_qchem_level1_double::mul2_i_i_x_m11(size_t ni,
    const double *a, double b, double *c) {

    #pragma ivdep
    for(size_t i = 0; i < ni; i++) {
        c[i] -= a[i];
    }
}


void linalg_qchem_level1_double::mul2_i_i_x_mxx(size_t ni,
    const double *a, size_t sia, double b, double *c, size_t sic) {

    for(size_t i = 0; i < ni; i++) {
        c[sic * i] -= a[sia * i];
    }
}


} // namespace libtensor
