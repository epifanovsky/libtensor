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

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_ACML_H
