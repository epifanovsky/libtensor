#include <acml.h>
#include "linalg_acml_level1.h"

namespace libtensor {


const char *linalg_acml_level1::k_clazz = "acml";


void linalg_acml_level1::add_i_i_x_x(
    void*,
    size_t ni,
    const double *a, size_t sia, double ka,
    double b, double kb,
    double *c, size_t sic,
    double d) {

    timings_base::start_timer("daxpy+nonblas");
    daxpy(ni, d * ka, (double*)a, sia, c, sic);
    double db = d * kb * b;
    if(sic == 1) {
        for(size_t i = 0; i < ni; i++) c[i] += db;
    } else {
        for(size_t i = 0; i < ni; i++) c[i * sic] += db;
    }
    timings_base::stop_timer("daxpy+nonblas");
}


void linalg_acml_level1::mul1_i_x(
    void*,
    size_t ni,
    double a,
    double *c, size_t sic) {

    timings_base::start_timer("dscal");
    dscal(ni, a, c, sic);
    timings_base::stop_timer("dscal");
}


double linalg_acml_level1::mul2_x_p_p(
    void*,
    size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb) {

    timings_base::start_timer("ddot");
    double d = ddot(np, (double*)a, spa, (double*)b, spb);
    timings_base::stop_timer("ddot");
    return d;
}


void linalg_acml_level1::mul2_i_i_x(
    void*,
    size_t ni,
    const double *a, size_t sia,
    double b,
    double *c, size_t sic) {

    timings_base::start_timer("daxpy");
    daxpy(ni, b, (double*)a, sia, c, sic);
    timings_base::stop_timer("daxpy");
}


} // namespace libtensor
