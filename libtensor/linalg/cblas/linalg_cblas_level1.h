#ifndef LIBTENSOR_LINALG_CBLAS_LEVEL1_H
#define LIBTENSOR_LINALG_CBLAS_LEVEL1_H

#include "../generic/linalg_generic_level1.h"

namespace libtensor {


/** \brief Level-1 linear algebra operations (CBLAS)

    \ingroup libtensor_linalg
 **/
struct linalg_cblas_level1 : public linalg_generic_level1 {


    static void add_i_i_x_x(
        size_t ni,
        const double *a, size_t sia, double ka,
        double b, double kb,
        double *c, size_t sic,
        double d) {

        cblas_daxpy(ni, d * ka, a, sia, c, sic);
        double db = d * kb * b;
        if(sic == 1) {
            for(size_t i = 0; i < ni; i++) c[i] += db;
        } else {
            for(size_t i = 0; i < ni; i++) c[i * sic] += db;
        }
    }


    static void i_i(
        size_t ni,
        const double *a, size_t sia,
        double *c, size_t sic) {

        cblas_dcopy(ni, a, sia, c, sic);
    }


    static void i_x(
        size_t ni,
        double a,
        double *c, size_t sic) {

        cblas_dscal(ni, a, c, sic);
    }


    static double x_p_p(
        size_t np,
        const double *a, size_t spa,
        const double *b, size_t spb) {

        return cblas_ddot(np, a, spa, b, spb);
    }


    static void i_i_x(
        size_t ni,
        const double *a, size_t sia,
        double b,
        double *c, size_t sic) {

        cblas_daxpy(ni, b, a, sia, c, sic);
    }

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_CBLAS_LEVEL1_H
