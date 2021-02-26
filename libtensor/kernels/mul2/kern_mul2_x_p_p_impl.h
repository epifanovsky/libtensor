#ifndef LIBTENSOR_KERN_MUL2_X_P_P_IMPL_H
#define LIBTENSOR_KERN_MUL2_X_P_P_IMPL_H

#include "kern_mul2_x_p_p.h"
#include "kern_mul2_i_ip_p.h"
#include "kern_mul2_i_p_ip.h"
#include "kern_mul2_x_pq_pq.h"
#include "kern_mul2_x_pq_qp.h"

namespace libtensor {


template<typename LA, typename T>
const char *kern_mul2_x_p_p<LA, T>::k_clazz = "kern_mul2_x_p_p";


template<typename LA, typename T>
void kern_mul2_x_p_p<LA, T>::run(
    device_context_ref ctx,
    const loop_registers_x<2, 1, T> &r) {

    r.m_ptrb[0][0] += LA::mul2_x_p_p(ctx, m_np, r.m_ptra[0], m_spa,
        r.m_ptra[1], m_spb) * m_d;
}


template<typename LA, typename T>
kernel_base<LA, 2, 1, T> *kern_mul2_x_p_p<LA, T>::match(const kern_mul2<LA, T> &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //    1. Minimize spa > 0:
    //    ------------
    //    w   a   b  c
    //    np  spa 1  0  -->  c_# = a_p# b_p
    //    ------------       sz(p) = np, sz(#) = spa
    //                       [mul2_x_p_p]

    iterator_t ip = in.end();
    size_t spa_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) > 0 && i->stepa(1) == 1 && i->stepb(0) == 0) {
            if(spa_min == 0 || spa_min > i->stepa(0)) {
                ip = i; spa_min = i->stepa(0);
            }
        }
    }
    if(ip == in.end()) return 0;

    kern_mul2_x_p_p zz;
    zz.m_d = z.m_d;
    zz.m_np = ip->weight();
    zz.m_spa = ip->stepa(0);
    zz.m_spb = 1;
    in.splice(out.begin(), out, ip);

    kernel_base<LA, 2, 1, T> *kern = 0;

    if((kern = kern_mul2_i_ip_p<LA, T>::match(zz, in, out))) return kern;
    if((kern = kern_mul2_i_p_ip<LA, T>::match(zz, in, out))) return kern;
    if((kern = kern_mul2_x_pq_pq<LA, T>::match(zz, in, out))) return kern;
    if((kern = kern_mul2_x_pq_qp<LA, T>::match(zz, in, out))) return kern;

    return new kern_mul2_x_p_p(zz);
}


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL2_X_P_P_IMPL_H
