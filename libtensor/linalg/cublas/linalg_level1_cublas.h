#ifndef LIBTENSOR_LINALG_LEVEL1_CUBLAS_H
#define LIBTENSOR_LINALG_LEVEL1_CUBLAS_H

#include <libtensor/timings.h>

namespace libtensor {


/** \brief Level-1 linear algebra operations (cuBLAS)

    \ingroup libtensor_linalg
 **/
class linalg_level1_cublas : public timings<linalg_level1_cublas> {
public:
    static const char *k_clazz; //!< Class name

public:
    static void i_x(
        size_t ni,
        double a,
        double *c, size_t sic);

    static double x_p_p(
        cublasHandle_t h,
        size_t np,
        const double *a, size_t spa,
        const double *b, size_t spb);

    static void i_i_x(
        size_t ni,
        const double *a, size_t sia,
        double b,
        double *c, size_t sic);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_LEVEL1_CUBLAS_H
