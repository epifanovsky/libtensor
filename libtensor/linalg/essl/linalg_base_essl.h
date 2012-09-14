#ifndef LIBTENSOR_LINALG_BASE_ESSL_H
#define LIBTENSOR_LINALG_BASE_ESSL_H

#include "essl_h.h"
#include "../generic/linalg_base_lowlevel.h"
#include "../generic/linalg_base_memory_generic.h"
#include "linalg_base_level1_essl.h"
#include "linalg_base_level2_essl.h"
#include "linalg_base_level3_essl.h"

namespace libtensor {


/** \brief Linear algebra implementation based on
        IBM Engineering and Scientific Subroutine Library (ESSL)

    \ingroup libtensor_linalg
 **/
struct linalg_base_essl :
    public linalg_base_lowlevel<
        linalg_base_memory_generic,
        linalg_base_level1_essl,
        linalg_base_level2_essl,
        linalg_base_level3_essl>
{ };


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_ESSL_H
