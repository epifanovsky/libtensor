#ifndef LIBTENSOR_LINALG_ESSL_H
#define LIBTENSOR_LINALG_ESSL_H

#include "linalg_essl_level1.h"
#include "linalg_essl_level2.h"
#include "linalg_essl_level3.h"

namespace libtensor {


/** \brief Linear algebra implementation based on
        IBM Engineering and Scientific Subroutine Library (ESSL)

    \ingroup libtensor_linalg
 **/
class linalg_essl :
    public linalg_essl_level1,
    public linalg_essl_level2,
    public linalg_essl_level3 {

public:
    typedef double element_type; //!< Data type
    typedef void *device_context_type; //!< Device context
    typedef void *device_context_ref; //!< Reference type to device context

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_ESSL_H
