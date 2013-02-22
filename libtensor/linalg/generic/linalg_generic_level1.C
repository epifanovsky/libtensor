#include <ctime>
#include <cstdlib>
#include "linalg_generic_level1.h"

namespace libtensor {


const char *linalg_generic_level1::k_clazz = "generic";


void linalg_generic_level1::add_i_i_x_x(
    void*,
    size_t ni,
    const double *a, size_t sia, double ka,
    double b, double kb,
    double *c, size_t sic,
    double d) {

    timings_base::start_timer("add_i_i_x_x");
    for(size_t i = 0; i < ni; i++) {
        c[i * sic] += (ka * a[i * sia] + kb * b) * d;
    }
    timings_base::stop_timer("add_i_i_x_x");
}


void linalg_generic_level1::copy_i_i(
    void*,
    size_t ni,
    const double *a, size_t sia,
    double *c, size_t sic) {

    timings_base::start_timer("copy_i_i");
    for(size_t i = 0; i < ni; i++) c[i * sic] = a[i * sia];
    timings_base::stop_timer("copy_i_i");
}


void linalg_generic_level1::div1_i_i_x(
    void *,
    size_t ni,
    const double *a, size_t sia,
    double *c, size_t sic,
    double d) {

    timings_base::start_timer("div1_i_i_x");
    for(size_t i = 0; i < ni; i++) c[i * sic] = c[i * sic] * d / a[i * sia];
    timings_base::stop_timer("div1_i_i_x");
}


void linalg_generic_level1::mul1_i_x(
    void*,
    size_t ni,
    double a,
    double *c, size_t sic) {

    timings_base::start_timer("mul1_i_x");
    for(size_t i = 0; i < ni; i++) c[i * sic] *= a;
    timings_base::stop_timer("mul1_i_x");
}


double linalg_generic_level1::mul2_x_p_p(
    void*,
    size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb) {

    timings_base::start_timer("mul2_x_p_p");
    double c = 0.0;
    for(size_t p = 0; p < np; p++) c += a[p * spa] * b[p * spb];
    timings_base::stop_timer("mul2_x_p_p");
    return c;
}


void linalg_generic_level1::mul2_i_i_x(
    void*,
    size_t ni,
    const double *a, size_t sia,
    double b,
    double *c, size_t sic) {

    timings_base::start_timer("mul2_i_i_x");
    for(size_t i = 0; i < ni; i++) c[i * sic] += a[i * sia] * b;
    timings_base::stop_timer("mul2_i_i_x");
}


void linalg_generic_level1::mul2_i_i_i_x(
    void*,
    size_t ni,
    const double *a, size_t sia,
    const double *b, size_t sib,
    double *c, size_t sic,
    double d) {

    timings_base::start_timer("mul2_i_i_i_x");
    for(size_t i = 0; i < ni; i++) c[i * sic] += d * a[i * sia] * b[i * sib];
    timings_base::stop_timer("mul2_i_i_i_x");
}


void linalg_generic_level1::rng_setup(
    void*) {

#if defined(HAVE_DRAND48)
    ::srand48(::time(0));
#else // HAVE_DRAND48
    ::srand(::time(0));
#endif // HAVE_DRAND48
}


void linalg_generic_level1::rng_set_i_x(
    void*,
    size_t ni,
    double *a, size_t sia,
    double c) {

#if defined(HAVE_DRAND48)
    for(size_t i = 0; i < ni; i++) a[i * sia] = c * ::drand48();
#else // HAVE_DRAND48
    for(size_t i = 0; i < ni; i++) {
        a[i * sia] = c * double(::rand()) / double(RAND_MAX);
    }
#endif // HAVE_DRAND48
}


void linalg_generic_level1::rng_add_i_x(
    void*,
    size_t ni,
    double *a, size_t sia,
    double c) {

#if defined(HAVE_DRAND48)
    for(size_t i = 0; i < ni; i++) a[i * sia] += c * ::drand48();
#else // HAVE_DRAND48
    for(size_t i = 0; i < ni; i++) {
        a[i * sia] += c * double(::rand()) / double(RAND_MAX);
    }
#endif // HAVE_DRAND48
}


} // namespace libtensor
