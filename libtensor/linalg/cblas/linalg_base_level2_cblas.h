#ifndef LIBTENSOR_LINALG_BASE_LEVEL2_CBLAS_H
#define LIBTENSOR_LINALG_BASE_LEVEL2_CBLAS_H

#include "../generic/linalg_base_level2_generic.h"

namespace libtensor {


/** \brief Level-2 linear algebra operations (CBLAS)

    \ingroup libtensor_linalg
 **/
struct linalg_base_level2_cblas : public linalg_base_level2_generic {


    static void i_ip_p_x(
        size_t ni, size_t np,
        const double *a, size_t sia,
        const double *b, size_t spb,
        double *c, size_t sic,
        double d) {

        cblas_dgemv(CblasRowMajor, CblasNoTrans, ni, np, d, a, sia, b,
            spb, 1.0, c, sic);
    }


    static void i_pi_p_x(
        size_t ni, size_t np,
        const double *a, size_t spa,
        const double *b, size_t spb,
        double *c, size_t sic,
        double d) {

        cblas_dgemv(CblasRowMajor, CblasTrans, np, ni, d, a, spa, b,
            spb, 1.0, c, sic);
    }


    static void ij_i_j_x(
        size_t ni, size_t nj,
        const double *a, size_t sia,
        const double *b, size_t sjb,
        double *c, size_t sic,
        double d) {

        cblas_dger(CblasRowMajor, ni, nj, d, a, sia, b, sjb, c, sic);
    }


    static void ij_ji(
        size_t ni, size_t nj,
        const double *a, size_t sja,
        double *c, size_t sic) {

        if(ni < nj) {
            double *c1 = c;
            for(size_t i = 0; i < ni; i++, c1 += sic) {
                cblas_dcopy(nj, a + i, sja, c1, 1);
            }
        } else {
            const double *a1 = a;
            for(size_t j = 0; j < nj; j++, a1 += sja) {
                cblas_dcopy(ni, a1, 1, c + j, sic);
            }
        }
    }


};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_LEVEL2_CBLAS_H
