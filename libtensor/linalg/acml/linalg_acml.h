#ifndef LIBTENSOR_LINALG_ACML_H
#define LIBTENSOR_LINALG_ACML_H

#include "linalg_acml_level1.h"
#include "linalg_acml_level2.h"
#include "linalg_acml_level3.h"

namespace libtensor {


/** \brief Linear algebra implementation based on
        AMD Core Math Library (ACML)

    \ingroup libtensor_linalg
 **/
class linalg_acml :
    public linalg_acml_level1,
    public linalg_acml_level2,
    public linalg_acml_level3 {

public:
    typedef double element_type; //!< Data type
    typedef void *device_context_type; //!< Device context
    typedef void *device_context_ref; //!< Reference type to device context

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_ACML_H
