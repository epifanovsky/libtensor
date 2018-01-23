#ifndef LIBTENSOR_LINALG_CBLAS_H
#define LIBTENSOR_LINALG_CBLAS_H

#include "linalg_cblas_level1.h"
#include "linalg_cblas_level2.h"
#include "linalg_cblas_level3.h"

namespace libtensor {


/** \brief Linear algebra implementation based on CBLAS

    \ingroup libtensor_linalg
 **/
class linalg_cblas :
    public linalg_cblas_level1<double>,
    public linalg_cblas_level2<double>,
    public linalg_cblas_level3<double>,
    public linalg_cblas_level1<float>,
    public linalg_cblas_level2<float>,
    public linalg_cblas_level3<float> {

public:
//    typedef double element_type; //!< Data type
    typedef void *device_context_type; //!< Device context
    typedef void *device_context_ref; //!< Reference type to device context


using linalg_cblas_level1<double>::k_clazz;
using linalg_cblas_level1<double>::add_i_i_x_x;
using linalg_cblas_level1<double>::copy_i_i;
using linalg_cblas_level1<double>::div1_i_i_x;
using linalg_cblas_level1<double>::mul1_i_x;
using linalg_cblas_level1<double>::mul2_x_p_p;
using linalg_cblas_level1<double>::mul2_i_i_x;
using linalg_cblas_level1<double>::mul2_i_i_i_x;
using linalg_cblas_level1<double>::rng_setup;
using linalg_cblas_level1<double>::rng_set_i_x;
using linalg_cblas_level1<double>::rng_add_i_x;

using linalg_cblas_level2<double>::add1_ij_ij_x;
using linalg_cblas_level2<double>::add1_ij_ji_x;
using linalg_cblas_level2<double>::copy_ij_ij_x;
using linalg_cblas_level2<double>::copy_ij_ji;
using linalg_cblas_level2<double>::copy_ij_ji_x;
using linalg_cblas_level2<double>::mul2_i_ip_p_x;
using linalg_cblas_level2<double>::mul2_i_pi_p_x;
using linalg_cblas_level2<double>::mul2_ij_i_j_x;
using linalg_cblas_level2<double>::mul2_x_pq_pq;
using linalg_cblas_level2<double>::mul2_x_pq_qp;

using linalg_cblas_level3<double>::mul2_i_ipq_qp_x;
using linalg_cblas_level3<double>::mul2_ij_ip_jp_x;
using linalg_cblas_level3<double>::mul2_ij_ip_pj_x;
using linalg_cblas_level3<double>::mul2_ij_pi_jp_x;
using linalg_cblas_level3<double>::mul2_ij_pi_pj_x;

using linalg_cblas_level1<float>::add_i_i_x_x;
using linalg_cblas_level1<float>::copy_i_i;
using linalg_cblas_level1<float>::div1_i_i_x;
using linalg_cblas_level1<float>::mul1_i_x;
using linalg_cblas_level1<float>::mul2_x_p_p;
using linalg_cblas_level1<float>::mul2_i_i_x;
using linalg_cblas_level1<float>::mul2_i_i_i_x;
using linalg_cblas_level1<float>::rng_setup;
using linalg_cblas_level1<float>::rng_set_i_x;
using linalg_cblas_level1<float>::rng_add_i_x;

using linalg_cblas_level2<float>::add1_ij_ij_x;
using linalg_cblas_level2<float>::add1_ij_ji_x;
using linalg_cblas_level2<float>::copy_ij_ij_x;
using linalg_cblas_level2<float>::copy_ij_ji;
using linalg_cblas_level2<float>::copy_ij_ji_x;
using linalg_cblas_level2<float>::mul2_i_ip_p_x;
using linalg_cblas_level2<float>::mul2_i_pi_p_x;
using linalg_cblas_level2<float>::mul2_ij_i_j_x;
using linalg_cblas_level2<float>::mul2_x_pq_pq;
using linalg_cblas_level2<float>::mul2_x_pq_qp;

using linalg_cblas_level3<float>::mul2_i_ipq_qp_x;
using linalg_cblas_level3<float>::mul2_ij_ip_jp_x;
using linalg_cblas_level3<float>::mul2_ij_ip_pj_x;
using linalg_cblas_level3<float>::mul2_ij_pi_jp_x;
using linalg_cblas_level3<float>::mul2_ij_pi_pj_x;
};

} // namespace libtensor

#endif // LIBTENSOR_LINALG_CBLAS_H
