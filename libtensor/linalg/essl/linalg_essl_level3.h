#ifndef LIBTENSOR_LINALG_ESSL_LEVEL3_H
#define LIBTENSOR_LINALG_ESSL_LEVEL3_H

#include "essl_h.h"
#include "../generic/linalg_generic_level3.h"

namespace libtensor {


/** \brief Level-3 linear algebra operations (ESSL)

    \ingroup libtensor_linalg
 **/
class linalg_essl_level3 : public linalg_generic_level3 {


    static void ij_ip_jp_x(
        size_t ni, size_t nj, size_t np,
        const double *a, size_t sia,
        const double *b, size_t sjb,
        double *c, size_t sic,
        double d) {

        dgemm("T", "N", nj, ni, np, d, (double*)b, sjb, (double*)a, sia,
            1.0, c, sic);
    }


    static void ij_ip_pj_x(
        size_t ni, size_t nj, size_t np,
        const double *a, size_t sia,
        const double *b, size_t spb,
        double *c, size_t sic,
        double d) {

        dgemm("N", "N", nj, ni, np, d, (double*)b, spb, (double*)a, sia,
            1.0, c, sic);
    }


    static void ij_pi_jp_x(
        size_t ni, size_t nj, size_t np,
        const double *a, size_t spa,
        const double *b, size_t sjb,
        double *c, size_t sic,
        double d) {

        dgemm("T", "T", nj, ni, np, d, (double*)b, sjb, (double*)a, spa,
            1.0, c, sic);
    }


    static void ij_pi_pj_x(
        size_t ni, size_t nj, size_t np,
        const double *a, size_t spa,
        const double *b, size_t spb,
        double *c, size_t sic,
        double d) {

        dgemm("N", "T", nj, ni, np, d, (double*)b, spb, (double*)a, spa,
            1.0, c, sic);
    }


};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_ESSL_LEVEL3_H
