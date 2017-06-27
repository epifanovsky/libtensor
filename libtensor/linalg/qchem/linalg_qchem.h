#ifndef LIBTENSOR_LINALG_QCHEM_H
#define LIBTENSOR_LINALG_QCHEM_H

#include "linalg_qchem_level1_double.h"
#include "linalg_qchem_level2_double.h"
#include "linalg_qchem_level3_double.h"
#include "linalg_qchem_level3_float.h"

namespace libtensor {


/** \brief Linear algebra implementation based on Q-Chem's imported BLAS

    \ingroup libtensor_linalg
 **/
class linalg_qchem :
    public linalg_qchem_level1_double,
    public linalg_qchem_level2_double,
    public linalg_qchem_level3_double,
    public linalg_generic_level1<float>,
    public linalg_generic_level2<float>,
    public linalg_qchem_level3_float {

public:
    typedef void *device_context_type; //!< Device context
    typedef void *device_context_ref; //!< Reference type to device context

public:
using linalg_qchem_level1_double::add_i_i_x_x;
using linalg_qchem_level1_double::copy_i_i;
using linalg_qchem_level1_double::div1_i_i_x;
using linalg_qchem_level1_double::mul1_i_x;
using linalg_qchem_level1_double::mul2_x_p_p;
using linalg_qchem_level1_double::mul2_i_i_x;
using linalg_qchem_level1_double::mul2_i_i_i_x;
using linalg_qchem_level1_double::rng_setup;
using linalg_qchem_level1_double::rng_set_i_x;
using linalg_qchem_level1_double::rng_add_i_x;

using linalg_qchem_level2_double::add1_ij_ij_x;
using linalg_qchem_level2_double::add1_ij_ji_x;
using linalg_qchem_level2_double::copy_ij_ij_x;
using linalg_qchem_level2_double::copy_ij_ji;
using linalg_qchem_level2_double::copy_ij_ji_x;
using linalg_qchem_level2_double::mul2_i_ip_p_x;
using linalg_qchem_level2_double::mul2_i_pi_p_x;
using linalg_qchem_level2_double::mul2_ij_i_j_x;
using linalg_qchem_level2_double::mul2_x_pq_pq;
using linalg_qchem_level2_double::mul2_x_pq_qp;

using linalg_qchem_level3_double::mul2_i_ipq_qp_x;
using linalg_qchem_level3_double::mul2_ij_ip_jp_x;
using linalg_qchem_level3_double::mul2_ij_ip_pj_x;
using linalg_qchem_level3_double::mul2_ij_pi_jp_x;
using linalg_qchem_level3_double::mul2_ij_pi_pj_x;

using linalg_generic_level1<float>::add_i_i_x_x;
using linalg_generic_level1<float>::copy_i_i;
using linalg_generic_level1<float>::div1_i_i_x;
using linalg_generic_level1<float>::mul1_i_x;
using linalg_generic_level1<float>::mul2_x_p_p;
using linalg_generic_level1<float>::mul2_i_i_x;
using linalg_generic_level1<float>::mul2_i_i_i_x;
using linalg_generic_level1<float>::rng_setup;
using linalg_generic_level1<float>::rng_set_i_x;
using linalg_generic_level1<float>::rng_add_i_x;

using linalg_generic_level2<float>::add1_ij_ij_x;
using linalg_generic_level2<float>::add1_ij_ji_x;
using linalg_generic_level2<float>::copy_ij_ij_x;
using linalg_generic_level2<float>::copy_ij_ji;
using linalg_generic_level2<float>::copy_ij_ji_x;
using linalg_generic_level2<float>::mul2_i_ip_p_x;
using linalg_generic_level2<float>::mul2_i_pi_p_x;
using linalg_generic_level2<float>::mul2_ij_i_j_x;
using linalg_generic_level2<float>::mul2_x_pq_pq;
using linalg_generic_level2<float>::mul2_x_pq_qp;

using linalg_qchem_level3_float::mul2_i_ipq_qp_x;
using linalg_qchem_level3_float::mul2_ij_ip_jp_x;
using linalg_qchem_level3_float::mul2_ij_ip_pj_x;
using linalg_qchem_level3_float::mul2_ij_pi_jp_x;
using linalg_qchem_level3_float::mul2_ij_pi_pj_x;
};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_QCHEM_H
