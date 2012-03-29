#ifndef LIBTENSOR_LINALG_BASE_ACML_H
#define LIBTENSOR_LINALG_BASE_ACML_H

#include "acml_h.h"
#include "../generic/linalg_base_lowlevel.h"
#include "../adaptive/linalg_base_highlevel.h"
#include "../generic/linalg_base_memory_generic.h"
#include "linalg_base_level1_acml.h"
#include "linalg_base_level2_acml.h"
#include "linalg_base_level3_acml.h"

namespace libtensor {


/** \brief Linear algebra implementation based on
        AMD Core Math Library (ACML)

    \ingroup libtensor_linalg
 **/
struct linalg_base_acml :
    public linalg_base_lowlevel<
        linalg_base_memory_generic,
        linalg_base_level1_acml,
        linalg_base_level2_acml,
        linalg_base_level3_acml>,
    public linalg_base_highlevel<
        linalg_base_memory_generic,
        linalg_base_level1_acml,
        linalg_base_level2_acml,
        linalg_base_level3_acml>
{ };


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_ACML_H
