#ifndef LIBTENSOR_LINALG_BASE_LEVEL3_CBLAS_H
#define LIBTENSOR_LINALG_BASE_LEVEL3_CBLAS_H

#include "../generic/linalg_base_level3_generic.h"

namespace libtensor {


/** \brief Level-3 linear algebra operations (CBLAS)

    \ingroup libtensor_linalg
 **/
struct linalg_base_level3_cblas : public linalg_base_level3_generic {


    static void ij_ip_jp_x(
        size_t ni, size_t nj, size_t np,
        const double *a, size_t sia,
        const double *b, size_t sjb,
        double *c, size_t sic,
        double d) {

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ni, nj, np,
            d, a, sia, b, sjb, 1.0, c, sic);
    }


    static void ij_ip_pj_x(
        size_t ni, size_t nj, size_t np,
        const double *a, size_t sia,
        const double *b, size_t spb,
        double *c, size_t sic,
        double d) {

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ni, nj,
            np, d, a, sia, b, spb, 1.0, c, sic);
    }


    static void ij_pi_jp_x(
        size_t ni, size_t nj, size_t np,
        const double *a, size_t spa,
        const double *b, size_t sjb,
        double *c, size_t sic,
        double d) {

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, ni, nj, np,
            d, a, spa, b, sjb, 1.0, c, sic);
    }


    static void ij_pi_pj_x(
        size_t ni, size_t nj, size_t np,
        const double *a, size_t spa,
        const double *b, size_t spb,
        double *c, size_t sic,
        double d) {

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ni, nj, np,
            d, a, spa, b, spb, 1.0, c, sic);
    }


};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_LEVEL3_CBLAS_H
