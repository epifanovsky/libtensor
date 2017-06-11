#include <libtensor/linalg/linalg.h>
#include "kern_mul2_impl.h"
#include "kern_mul2_i_i_i_impl.h"
#include "kern_mul2_i_x_i_impl.h"
#include "kern_mul2_i_i_x_impl.h"
#include "kern_mul2_x_p_p_impl.h"
#include "kern_mul2_i_p_ip_impl.h"
#include "kern_mul2_i_p_pi_impl.h"
#include "kern_mul2_i_ip_p_impl.h"
#include "kern_mul2_i_pi_p_impl.h"
#include "kern_mul2_ij_i_j_impl.h"
#include "kern_mul2_ij_j_i_impl.h"
#include "kern_mul2_x_pq_pq_impl.h"
#include "kern_mul2_x_pq_qp_impl.h"
#include "kern_mul2_ij_ip_jp_impl.h"
#include "kern_mul2_ij_ip_pj_impl.h"
#include "kern_mul2_ij_jp_ip_impl.h"
#include "kern_mul2_ij_jp_pi_impl.h"
#include "kern_mul2_ij_pi_jp_impl.h"
#include "kern_mul2_ij_pi_pj_impl.h"
#include "kern_mul2_ij_pj_ip_impl.h"
#include "kern_mul2_ij_pj_pi_impl.h"

namespace libtensor {


template class kern_mul2<linalg, double>;


} // namespace libtensor
