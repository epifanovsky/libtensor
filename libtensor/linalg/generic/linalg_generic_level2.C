#include "linalg_generic_level2.h"

namespace libtensor {


template<typename T>
const char linalg_generic_level2<T>::k_clazz[] = "generic";


template<typename T>
void linalg_generic_level2<T>::add1_ij_ij_x(
    void*,
    size_t ni, size_t nj,
    const T *a, size_t sia,
    T b,
    T *c, size_t sic) {

    timings_base::start_timer("add1_ij_ij_x");
    for(size_t i = 0; i < ni; i++)
    for(size_t j = 0; j < nj; j++) {
        c[i * sic + j] += a[i * sia + j] * b;
    }
    timings_base::stop_timer("add1_ij_ij_x");
}


template<typename T>
void linalg_generic_level2<T>::add1_ij_ji_x(
    void*,
    size_t ni, size_t nj,
    const T *a, size_t sja,
    T b,
    T *c, size_t sic) {

    timings_base::start_timer("add1_ij_ji_x");
    for(size_t j = 0; j < nj; j++)
    for(size_t i = 0; i < ni; i++) {
        c[i * sic + j] += a[j * sja + i] * b;
    }
    timings_base::stop_timer("add1_ij_ji_x");
}


template<typename T>
void linalg_generic_level2<T>::copy_ij_ij_x(
    void*,
    size_t ni, size_t nj,
    const T *a, size_t sia,
    T b,
    T *c, size_t sic) {

    timings_base::start_timer("copy_ij_ij_x");
    for(size_t i = 0; i < ni; i++)
    for(size_t j = 0; j < nj; j++) {
        c[i * sic + j] = a[i * sia + j] * b;
    }
    timings_base::stop_timer("copy_ij_ij_x");
}


template<typename T>
void linalg_generic_level2<T>::copy_ij_ji(
    void*,
    size_t ni, size_t nj,
    const T *a, size_t sja,
    T *c, size_t sic) {

    timings_base::start_timer("copy_ij_ji");
    for(size_t j = 0; j < nj; j++)
    for(size_t i = 0; i < ni; i++) {
        c[i * sic + j] = a[j * sja + i];
    }
    timings_base::stop_timer("copy_ij_ji");
}


template<typename T>
void linalg_generic_level2<T>::copy_ij_ji_x(
    void*,
    size_t ni, size_t nj,
    const T *a, size_t sja,
    T b,
    T *c, size_t sic) {

    timings_base::start_timer("copy_ij_ji_x");
    for(size_t j = 0; j < nj; j++)
    for(size_t i = 0; i < ni; i++) {
        c[i * sic + j] = a[j * sja + i] * b;
    }
    timings_base::stop_timer("copy_ij_ji_x");
}


template<typename T>
void linalg_generic_level2<T>::mul2_i_ip_p_x(
    void*,
    size_t ni, size_t np,
    const T *a, size_t sia,
    const T *b, size_t spb,
    T *c, size_t sic,
    T d) {

    timings_base::start_timer("mul2_i_ip_p_x");
    for(size_t i = 0; i < ni; i++) {
        T ci = 0.0;
        for(size_t p = 0; p < np; p++) {
            ci += a[i * sia + p] * b[p * spb];
        }
        c[i * sic] += d * ci;
    }
    timings_base::stop_timer("mul2_i_ip_p_x");
}


template<typename T>
void linalg_generic_level2<T>::mul2_i_pi_p_x(
    void*,
    size_t ni, size_t np,
    const T *a, size_t spa,
    const T *b, size_t spb,
    T *c, size_t sic,
    T d) {

    timings_base::start_timer("mul2_i_pi_p_x");
    for(size_t p = 0; p < np; p++)
    for(size_t i = 0; i < ni; i++) {
        c[i * sic] += d * a[p * spa + i] * b[p * spb];
    }
    timings_base::stop_timer("mul2_i_pi_p_x");
}


template<typename T>
void linalg_generic_level2<T>::mul2_ij_i_j_x(
    void*,
    size_t ni, size_t nj,
    const T *a, size_t sia,
    const T *b, size_t sjb,
    T *c, size_t sic,
    T d) {

    timings_base::start_timer("mul2_ij_i_j_x");
    for(size_t i = 0; i < ni; i++)
    for(size_t j = 0; j < nj; j++) {
        c[i * sic + j] += d * a[i * sia] * b[j * sjb];
    }
    timings_base::stop_timer("mul2_ij_i_j_x");
}


template<typename T>
T linalg_generic_level2<T>::mul2_x_pq_pq(
    void*,
    size_t np, size_t nq,
    const T *a, size_t spa,
    const T *b, size_t spb) {

    timings_base::start_timer("mul2_x_pq_pq");
    T c = 0.0;
    for(size_t p = 0; p < np; p++) {
        const T *a1 = a + p * spa, *b1 = b + p * spb;
        for(size_t q = 0; q < nq; q++) {
            c += a1[q] * b1[q];
        }
    }
    timings_base::stop_timer("mul2_x_pq_pq");
    return c;
}


template<typename T>
T linalg_generic_level2<T>::mul2_x_pq_qp(
    void*,
    size_t np, size_t nq,
    const T *a, size_t spa,
    const T *b, size_t sqb) {

    timings_base::start_timer("mul2_x_pq_qp");
    T c = 0.0;
    for(size_t p = 0; p < np; p++)
    for(size_t q = 0; q < nq; q++) {
        c += a[p * spa + q] * b[q * sqb + p];
    }
    timings_base::stop_timer("mul2_x_pq_qp");
    return c;
}

template class linalg_generic_level2<double>; 
template class linalg_generic_level2<float>; 

} // namespace libtensor
