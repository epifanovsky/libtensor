#include "linalg_generic_level2.h"
#include "linalg_generic_level3.h"

namespace libtensor {


template<typename T>
const char linalg_generic_level3<T>::k_clazz[] = "generic";


template<typename T>
void linalg_generic_level3<T>::mul2_i_ipq_qp_x(
    void *ctx,
    size_t ni, size_t np, size_t nq,
    const T *a, size_t spa, size_t sia,
    const T *b, size_t sqb,
    T *c, size_t sic,
    T d) {

    timings_base::start_timer("mul2_i_ipq_qp_x");
    for(size_t i = 0; i < ni; i++) {
        c[i * sic] += d * linalg_generic_level2<T>::mul2_x_pq_qp(ctx, np, nq,
            a + i * sia, spa, b, sqb);
    }
    timings_base::stop_timer("mul2_i_ipq_qp_x");
}


template<typename T>
void linalg_generic_level3<T>::mul2_ij_ip_jp_x(
    void*,
    size_t ni, size_t nj, size_t np,
    const T *a, size_t sia,
    const T *b, size_t sjb,
    T *c, size_t sic,
    T d) {

    timings_base::start_timer("mul2_ij_ip_jp_x");
    for(size_t i = 0; i < ni; i++)
    for(size_t j = 0; j < nj; j++) {
        T cij = 0.0;
        for(size_t p = 0; p < np; p++) {
            cij += a[i * sia + p] * b[j * sjb + p];
        }
        c[i * sic + j] += d * cij;
    }
    timings_base::stop_timer("mul2_ij_ip_jp_x");
}


template<typename T>
void linalg_generic_level3<T>::mul2_ij_ip_pj_x(
    void*,
    size_t ni, size_t nj, size_t np,
    const T *a, size_t sia,
    const T *b, size_t spb,
    T *c, size_t sic,
    T d) {

    timings_base::start_timer("mul2_ij_ip_pj_x");
    for(size_t i = 0; i < ni; i++)
    for(size_t p = 0; p < np; p++) {
        T aip = a[i * sia + p];
        for(size_t j = 0; j < nj; j++) {
            c[i * sic + j] += d * aip * b[p * spb + j];
        }
    }
    timings_base::stop_timer("mul2_ij_ip_pj_x");
}


template<typename T>
void linalg_generic_level3<T>::mul2_ij_pi_jp_x(
    void*,
    size_t ni, size_t nj, size_t np,
    const T *a, size_t spa,
    const T *b, size_t sjb,
    T *c, size_t sic,
    T d) {

    timings_base::start_timer("mul2_ij_pi_jp_x");
    for(size_t i = 0; i < ni; i++)
    for(size_t j = 0; j < nj; j++)
    for(size_t p = 0; p < np; p++) {
        c[i * sic + j] += d * a[p * spa + i] * b[j * sjb + p];
    }
    timings_base::stop_timer("mul2_ij_pi_jp_x");
}


template<typename T>
void linalg_generic_level3<T>::mul2_ij_pi_pj_x(
    void*,
    size_t ni, size_t nj, size_t np,
    const T *a, size_t spa,
    const T *b, size_t spb,
    T *c, size_t sic,
    T d) {

    timings_base::start_timer("mul2_ij_pi_pj_x");
    for(size_t p = 0; p < np; p++)
    for(size_t i = 0; i < ni; i++)
    for(size_t j = 0; j < nj; j++) {
        c[i * sic + j] += d * a[p * spa + i] * b[p * spb + j];
    }
    timings_base::stop_timer("mul2_ij_pi_pj_x");
}

template class linalg_generic_level3<double>; 
template class linalg_generic_level3<float>; 

} // namespace libtensor
