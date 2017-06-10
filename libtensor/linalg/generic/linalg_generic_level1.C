#include <ctime>
#include <cstdlib>
#include "linalg_generic_level1.h"

namespace libtensor {


template<typename T>
const char linalg_generic_level1<T>::k_clazz[] = "generic";


template<typename T>
void linalg_generic_level1<T>::add_i_i_x_x(
    void*,
    size_t ni,
    const T *a, size_t sia, T ka,
    T b, T kb,
    T *c, size_t sic,
    T d) {

    timings_base::start_timer("add_i_i_x_x");
    for(size_t i = 0; i < ni; i++) {
        c[i * sic] += (ka * a[i * sia] + kb * b) * d;
    }
    timings_base::stop_timer("add_i_i_x_x");
}


template<typename T>
void linalg_generic_level1<T>::copy_i_i(
    void*,
    size_t ni,
    const T *a, size_t sia,
    T *c, size_t sic) {

    timings_base::start_timer("copy_i_i");
    for(size_t i = 0; i < ni; i++) c[i * sic] = a[i * sia];
    timings_base::stop_timer("copy_i_i");
}


template<typename T>
void linalg_generic_level1<T>::div1_i_i_x(
    void *,
    size_t ni,
    const T *a, size_t sia,
    T *c, size_t sic,
    T d) {

    timings_base::start_timer("div1_i_i_x");
    for(size_t i = 0; i < ni; i++) c[i * sic] = c[i * sic] * d / a[i * sia];
    timings_base::stop_timer("div1_i_i_x");
}


template<typename T>
void linalg_generic_level1<T>::mul1_i_x(
    void*,
    size_t ni,
    T a,
    T *c, size_t sic) {

    timings_base::start_timer("mul1_i_x");
    for(size_t i = 0; i < ni; i++) c[i * sic] *= a;
    timings_base::stop_timer("mul1_i_x");
}


template<typename T>
T linalg_generic_level1<T>::mul2_x_p_p(
    void*,
    size_t np,
    const T *a, size_t spa,
    const T *b, size_t spb) {

    timings_base::start_timer("mul2_x_p_p");
    T c = 0.0;
    for(size_t p = 0; p < np; p++) c += a[p * spa] * b[p * spb];
    timings_base::stop_timer("mul2_x_p_p");
    return c;
}


template<typename T>
void linalg_generic_level1<T>::mul2_i_i_x(
    void*,
    size_t ni,
    const T *a, size_t sia,
    T b,
    T *c, size_t sic) {

    timings_base::start_timer("mul2_i_i_x");
    for(size_t i = 0; i < ni; i++) c[i * sic] += a[i * sia] * b;
    timings_base::stop_timer("mul2_i_i_x");
}


template<typename T>
void linalg_generic_level1<T>::mul2_i_i_i_x(
    void*,
    size_t ni,
    const T *a, size_t sia,
    const T *b, size_t sib,
    T *c, size_t sic,
    T d) {

    timings_base::start_timer("mul2_i_i_i_x");
    for(size_t i = 0; i < ni; i++) c[i * sic] += d * a[i * sia] * b[i * sib];
    timings_base::stop_timer("mul2_i_i_i_x");
}


template<typename T>
void linalg_generic_level1<T>::rng_setup(
    void*) {

#if defined(HAVE_DRAND48)
    ::srand48(::time(0));
#else // HAVE_DRAND48
    ::srand(::time(0));
#endif // HAVE_DRAND48
}


template<typename T>
void linalg_generic_level1<T>::rng_set_i_x(
    void*,
    size_t ni,
    T *a, size_t sia,
    T c) {

#if defined(HAVE_DRAND48)
    for(size_t i = 0; i < ni; i++) a[i * sia] = c * ::drand48();
#else // HAVE_DRAND48
    for(size_t i = 0; i < ni; i++) {
        a[i * sia] = c * T(::rand()) / T(RAND_MAX);
    }
#endif // HAVE_DRAND48
}


template<typename T>
void linalg_generic_level1<T>::rng_add_i_x(
    void*,
    size_t ni,
    T *a, size_t sia,
    T c) {

#if defined(HAVE_DRAND48)
    for(size_t i = 0; i < ni; i++) a[i * sia] += c * ::drand48();
#else // HAVE_DRAND48
    for(size_t i = 0; i < ni; i++) {
        a[i * sia] += c * T(::rand()) / T(RAND_MAX);
    }
#endif // HAVE_DRAND48
}


template class linalg_generic_level1<double>; 
template class linalg_generic_level1<float>; 

} // namespace libtensor
