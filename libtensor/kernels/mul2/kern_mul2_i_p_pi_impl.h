#ifndef LIBTENSOR_KERN_MUL2_I_P_PI_IMPL_H
#define LIBTENSOR_KERN_MUL2_I_P_PI_IMPL_H

#include "kern_mul2_i_p_pi.h"
#include "kern_mul2_ij_ip_pj.h"
#include "kern_mul2_ij_jp_pi.h"
#include "kern_mul2_ij_pi_pj.h"

namespace libtensor {


template<typename LA, typename T>
const char *kern_mul2_i_p_pi<LA, T>::k_clazz = "kern_mul2_i_p_pi";


template<typename LA, typename T>
void kern_mul2_i_p_pi<LA, T>::run(
    device_context_ref ctx,
    const loop_registers_x<2, 1, T> &r) {

    LA::mul2_i_pi_p_x(ctx, m_ni, m_np, r.m_ptra[1], m_spb, r.m_ptra[0], m_spa,
        r.m_ptrb[0], m_sic, m_d);
}


template<typename LA, typename T>
kernel_base<LA, 2, 1, T> *kern_mul2_i_p_pi<LA, T>::match(const kern_mul2_i_x_i<LA, T> &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Minimize spa > 0:
    //  -----------------
    //  w   a    b    c
    //  ni  0    1    sic
    //  np  spa  spb  0    -->  c_i# = a_p# b_p#i
    //  -----------------       [i_p_pi]
    //

    iterator_t ip = in.end();
    size_t spa_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) > 0 && i->stepa(1) > 0 && i->stepb(0) == 0) {
            if(i->stepa(1) % z.m_ni) continue;
            if(spa_min == 0 || spa_min > i->stepa(0)) {
                ip = i; spa_min = i->stepa(0);
            }
        }
    }
    if(ip == in.end()) return 0;

    kern_mul2_i_p_pi zz;
    zz.m_d = z.m_d;
    zz.m_ni = z.m_ni;
    zz.m_np = ip->weight();
    zz.m_spa = ip->stepa(0);
    zz.m_spb = ip->stepa(1);
    zz.m_sic = z.m_sic;
    in.splice(out.begin(), out, ip);

    kernel_base<LA, 2, 1, T> *kern = 0;

    if((kern = kern_mul2_ij_jp_pi<LA, T>::match(zz, in, out))) return kern;
    if((kern = kern_mul2_ij_ip_pj<LA, T>::match(zz, in, out))) return kern;
    if((kern = kern_mul2_ij_pi_pj<LA, T>::match(zz, in, out))) return kern;

    return new kern_mul2_i_p_pi(zz);
}


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL2_I_P_PI_IMPL_H
