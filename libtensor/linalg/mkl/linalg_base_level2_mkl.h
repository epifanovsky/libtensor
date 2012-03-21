#ifndef LIBTENSOR_LINALG_BASE_LEVEL2_MKL_H
#define LIBTENSOR_LINALG_BASE_LEVEL2_MKL_H

#if defined(HAVE_MKL_DOMATCOPY)
#include <mkl_trans.h>
#endif

#include "../cblas/linalg_base_level2_cblas.h"

namespace libtensor {


/** \brief Level-2 linear algebra operations (MKL)

    \ingroup libtensor_linalg
 **/
struct linalg_base_level2_mkl : public linalg_base_level2_cblas {


    static void ij_ji(
        size_t ni, size_t nj,
        const double *a, size_t sja,
        double *c, size_t sic) {

#if defined(HAVE_MKL_DOMATCOPY)
        mkl_domatcopy('R', 'T', nj, ni, 1.0, a, sja, c, sic);
#else
        linalg_base_level2_cblas::ij_ji(ni, nj, a, sja, c, sic);
#endif
    }


};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_LEVEL2_MKL_H
