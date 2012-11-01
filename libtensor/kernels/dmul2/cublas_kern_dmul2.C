#include <libtensor/linalg/cublas/linalg_cublas.h>
#include "kern_dmul2_impl.h"
#include "kern_dmul2_i_i_i_impl.h"
#include "kern_dmul2_i_x_i_impl.h"
#include "kern_dmul2_i_i_x_impl.h"
#include "kern_dmul2_x_p_p_impl.h"
#include "kern_dmul2_i_p_ip_impl.h"
#include "kern_dmul2_i_p_pi_impl.h"
#include "kern_dmul2_i_ip_p_impl.h"
#include "kern_dmul2_i_pi_p_impl.h"
#include "kern_dmul2_ij_i_j_impl.h"
#include "kern_dmul2_ij_j_i_impl.h"
#include "kern_dmul2_x_pq_qp_impl.h"
#include "kern_dmul2_ij_ip_jp_impl.h"
#include "kern_dmul2_ij_ip_pj_impl.h"
#include "kern_dmul2_ij_jp_ip_impl.h"
#include "kern_dmul2_ij_jp_pi_impl.h"
#include "kern_dmul2_ij_pi_jp_impl.h"
#include "kern_dmul2_ij_pi_pj_impl.h"
#include "kern_dmul2_ij_pj_ip_impl.h"
#include "kern_dmul2_ij_pj_pi_impl.h"

namespace libtensor {


template class kern_dmul2<linalg_cublas>;


} // namespace libtensor
