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
    public linalg_generic_level1<double>,
    public linalg_generic_level2<double>,
    public linalg_generic_level3<double>, 
    public linalg_generic_level1<float>,
    public linalg_generic_level2<float>,
    public linalg_generic_level3<float> {

public:
//    typedef double element_type; //!< Data type
    typedef void *device_context_type; //!< Device context
    typedef void *device_context_ref; //!< Reference type to device context

public:

// using const char linalg_generic_level1<double>::k_clazz[];
using linalg_generic_level1<double>::add_i_i_x_x;
using linalg_generic_level1<double>::copy_i_i;
using linalg_generic_level1<double>::div1_i_i_x;
using linalg_generic_level1<double>::mul1_i_x;
using linalg_generic_level1<double>::mul2_x_p_p;
using linalg_generic_level1<double>::mul2_i_i_x;
using linalg_generic_level1<double>::mul2_i_i_i_x;
using linalg_generic_level1<double>::rng_setup;
using linalg_generic_level1<double>::rng_set_i_x;
using linalg_generic_level1<double>::rng_add_i_x;

using linalg_generic_level2<double>::add1_ij_ij_x;
using linalg_generic_level2<double>::add1_ij_ji_x;
using linalg_generic_level2<double>::copy_ij_ij_x;
using linalg_generic_level2<double>::copy_ij_ji;
using linalg_generic_level2<double>::copy_ij_ji_x;
using linalg_generic_level2<double>::mul2_i_ip_p_x;
using linalg_generic_level2<double>::mul2_i_pi_p_x;
using linalg_generic_level2<double>::mul2_ij_i_j_x;
using linalg_generic_level2<double>::mul2_x_pq_pq;
using linalg_generic_level2<double>::mul2_x_pq_qp;

using linalg_generic_level3<double>::mul2_i_ipq_qp_x;
using linalg_generic_level3<double>::mul2_ij_ip_jp_x;
using linalg_generic_level3<double>::mul2_ij_ip_pj_x;
using linalg_generic_level3<double>::mul2_ij_pi_jp_x;
using linalg_generic_level3<double>::mul2_ij_pi_pj_x;

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

using linalg_generic_level3<float>::mul2_i_ipq_qp_x;
using linalg_generic_level3<float>::mul2_ij_ip_jp_x;
using linalg_generic_level3<float>::mul2_ij_ip_pj_x;
using linalg_generic_level3<float>::mul2_ij_pi_jp_x;
using linalg_generic_level3<float>::mul2_ij_pi_pj_x;
};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_GENERIC_H
