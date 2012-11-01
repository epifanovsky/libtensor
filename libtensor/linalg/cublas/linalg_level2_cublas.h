#ifndef LIBTENSOR_LINALG_LEVEL2_CUBLAS_H
#define LIBTENSOR_LINALG_LEVEL2_CUBLAS_H

#include <libtensor/timings.h>

namespace libtensor {


/** \brief Level-2 linear algebra operations (cuBLAS)

    \ingroup libtensor_linalg
 **/
class linalg_level2_cublas : public timings<linalg_level2_cublas> {
public:
    static const char *k_clazz; //!< Class name

public:
    static void add1_ij_ij_x(
        cublasHandle_t h,
        size_t ni, size_t nj,
        const double *a, size_t sia,
        double b,
        double *c, size_t sic);

    static void add1_ij_ji_x(
        cublasHandle_t h,
        size_t ni, size_t nj,
        const double *a, size_t sja,
        double b,
        double *c, size_t sic);

    static void copy_ij_ij_x(
        cublasHandle_t h,
        size_t ni, size_t nj,
        const double *a, size_t sia,
        double b,
        double *c, size_t sic);

    static void copy_ij_ji_x(
        cublasHandle_t h,
        size_t ni, size_t nj,
        const double *a, size_t sja,
        double b,
        double *c, size_t sic);

    static void mul2_i_ip_p_x(
        cublasHandle_t h,
        size_t ni, size_t np,
        const double *a, size_t sia,
        const double *b, size_t spb,
        double *c, size_t sic,
        double d);

    static void mul2_i_pi_p_x(
        cublasHandle_t h,
        size_t ni, size_t np,
        const double *a, size_t spa,
        const double *b, size_t spb,
        double *c, size_t sic,
        double d);

    static void mul2_ij_i_j_x(
        cublasHandle_t h,
        size_t ni, size_t nj,
        const double *a, size_t sia,
        const double *b, size_t sjb,
        double *c, size_t sic,
        double d);

    static double mul2_x_pq_qp(
        cublasHandle_t h,
        size_t np, size_t nq,
        const double *a, size_t spa,
        const double *b, size_t sqb);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_LEVEL2_CUBLAS_H
