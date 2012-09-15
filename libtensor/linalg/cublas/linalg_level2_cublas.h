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
    static void i_ip_p_x(
    	cublasHandle_t h,
        size_t ni, size_t np,
        const double *a, size_t sia,
        const double *b, size_t spb,
        double *c, size_t sic,
        double d);

    static void i_pi_p_x(
    	cublasHandle_t h,
        size_t ni, size_t np,
        const double *a, size_t spa,
        const double *b, size_t spb,
        double *c, size_t sic,
        double d);

    static void ij_i_j_x(
    	cublasHandle_t h,
        size_t ni, size_t nj,
        const double *a, size_t sia,
        const double *b, size_t sjb,
        double *c, size_t sic,
        double d);

    static void ij_ji_x(
        cublasHandle_t h,
        size_t ni, size_t nj,
        const double *a, size_t sja,
        double b,
        double *c, size_t sic);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_LEVEL2_CUBLAS_H
