#ifndef LIBTENSOR_LINALG_GENERIC_H
#define LIBTENSOR_LINALG_GENERIC_H

#include "linalg_generic_level1.h"
#include "linalg_generic_level2.h"
#include "linalg_generic_level3.h"

namespace libtensor {


/** \brief Generic linear algebra implementation

    \ingroup libtensor_linalg
 **/
class linalg_generic :
    public linalg_generic_level1,
    public linalg_generic_level2,
    public linalg_generic_level3 {

public:
    typedef double element_type; //!< Data type
    typedef void *device_context_type; //!< Device context
    typedef void *device_context_ref; //!< Reference type to device context

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_GENERIC_H
