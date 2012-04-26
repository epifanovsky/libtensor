#ifndef LIBTENSOR_LINALG_ADAPTIVE_TRP_IJKL_KJIL_H
#define LIBTENSOR_LINALG_ADAPTIVE_TRP_IJKL_KJIL_H

#include <cstring> // for memcpy

namespace libtensor {


/** \brief Transpose \f$ b_{ijkl} = a_{kjil} \f$

    \ingroup libtensor_linalg
 **/
struct trp_ijkl_kjil {

    static void transpose(
        size_t ni, size_t nj, size_t nk, size_t nl,
        const double *a, size_t sja,
        double *b, size_t sjb) {

        const double *aa = a;
        for(size_t k = 0; k < nk; k++) {
        for(size_t j = 0; j < nj; j++, aa += sja) {
        for(size_t i = 0; i < ni; i++) {
            memcpy(b + (i * nj + j) * sjb + k * nl, aa + i * nl,
                sizeof(double) * nl);
        }
        }
        }
    }
};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_ADAPTIVE_TRP_IJKL_KJIL_H
