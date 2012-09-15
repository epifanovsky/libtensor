#ifndef LIBTENSOR_KERN_DMUL2_I_PI_P_IMPL_H
#define LIBTENSOR_KERN_DMUL2_I_PI_P_IMPL_H

#include "kern_dmul2_i_pi_p.h"
#include "kern_dmul2_ij_pi_jp.h"
#include "kern_dmul2_ij_pi_pj.h"
#include "kern_dmul2_ij_pj_ip.h"
#include "kern_dmul2_ij_pj_pi.h"

namespace libtensor {


template<typename LA>
const char *kern_dmul2_i_pi_p<LA>::k_clazz = "kern_dmul2_i_pi_p";


template<typename LA>
void kern_dmul2_i_pi_p<LA>::run(const loop_registers<2, 1> &r) {

    LA::mul2_i_pi_p_x(0, m_ni, m_np, r.m_ptra[0], m_spa, r.m_ptra[1], m_spb,
        r.m_ptrb[0], m_sic, m_d);
}


template<typename LA>
kernel_base<2, 1> *kern_dmul2_i_pi_p<LA>::match(const kern_dmul2_i_i_x<LA> &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Minimize spb > 0:
    //  -------------------
    //  w   a    b     c
    //  ni  1    0     sic
    //  np  spa  spb   0     -->  c_i# = a_p#i b_p#
    //  -------------------       [i_pi_p]
    //

    iterator_t ip = in.end();
    size_t spb_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) > 0 && i->stepa(1) > 0 && i->stepb(0) == 0) {
            if(i->stepa(0) % z.m_ni) continue;
            if(spb_min == 0 || spb_min > i->stepa(1)) {
                ip = i; spb_min = i->stepa(1);
            }
        }
    }
    if(ip == in.end()) return 0;

    kern_dmul2_i_pi_p zz;
    zz.m_d = z.m_d;
    zz.m_ni = z.m_ni;
    zz.m_np = ip->weight();
    zz.m_spa = ip->stepa(0);
    zz.m_spb = ip->stepa(1);
    zz.m_sic = z.m_sic;
    in.splice(out.begin(), out, ip);

    kernel_base<2, 1> *kern = 0;

    if(kern = kern_dmul2_ij_pi_jp<LA>::match(zz, in, out)) return kern;
    if(kern = kern_dmul2_ij_pi_pj<LA>::match(zz, in, out)) return kern;
    if(kern = kern_dmul2_ij_pj_ip<LA>::match(zz, in, out)) return kern;
    if(kern = kern_dmul2_ij_pj_pi<LA>::match(zz, in, out)) return kern;

    return new kern_dmul2_i_pi_p(zz);
}


} // namespace libtensor

#endif // LIBTENSOR_KERN_DMUL2_I_PI_P_IMPL_H
