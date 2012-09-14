#ifndef LIBTENSOR_LINALG_BASE_CBLAS_H
#define LIBTENSOR_LINALG_BASE_CBLAS_H

extern "C" { // Fixes older cblas.h versions without extern "C"
#include <cblas.h>
}

#include "../generic/linalg_base_lowlevel.h"
#include "../generic/linalg_base_memory_generic.h"
#include "linalg_base_level1_cblas.h"
#include "linalg_base_level2_cblas.h"
#include "linalg_base_level3_cblas.h"

namespace libtensor {


/** \brief Linear algebra implementation based on CBLAS

    \ingroup libtensor_linalg
 **/
struct linalg_base_cblas :
    public linalg_base_lowlevel<
        linalg_base_memory_generic,
        linalg_base_level1_cblas,
        linalg_base_level2_cblas,
        linalg_base_level3_cblas>
{ };


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_CBLAS_H
