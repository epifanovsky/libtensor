#ifndef LIBTENSOR_LINALG_CBLAS_H
#define LIBTENSOR_LINALG_CBLAS_H

extern "C" { // Fixes older cblas.h versions without extern "C"
#include <cblas.h>
}

#include "linalg_cblas_level1.h"
#include "linalg_cblas_level2.h"
#include "linalg_cblas_level3.h"

namespace libtensor {


/** \brief Linear algebra implementation based on CBLAS

    \ingroup libtensor_linalg
 **/
class linalg_cblas :
    public linalg_cblas_level1,
    public linalg_cblas_level2,
    public linalg_cblas_level3 {

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_CBLAS_H
