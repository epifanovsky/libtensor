#ifndef LIBTENSOR_LINALG_BASE_QCHEM_H
#define LIBTENSOR_LINALG_BASE_QCHEM_H

#include "../generic/linalg_base_lowlevel.h"
#include "../adaptive/linalg_base_highlevel.h"
#include "linalg_base_memory_qchem.h"
#include "linalg_base_level1_qchem.h"
#include "linalg_base_level2_qchem.h"
#include "linalg_base_level3_qchem.h"

namespace libtensor {


/** \brief Linear algebra implementation based on Q-Chem's imported BLAS

    \ingroup libtensor_linalg
 **/
struct linalg_base_qchem :
    public linalg_base_lowlevel<
        linalg_base_memory_qchem,
        linalg_base_level1_qchem,
        linalg_base_level2_qchem,
        linalg_base_level3_qchem>,
    public linalg_base_highlevel<
        linalg_base_memory_qchem,
        linalg_base_level1_qchem,
        linalg_base_level2_qchem,
        linalg_base_level3_qchem>
{ };


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_QCHEM_H
