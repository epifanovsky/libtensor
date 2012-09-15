#ifndef LIBTENSOR_KERN_DMUL2_I_IP_P_IMPL_H
#define LIBTENSOR_KERN_DMUL2_I_IP_P_IMPL_H

#include "kern_dmul2_i_ip_p.h"
#include "kern_dmul2_ij_jp_ip.h"

namespace libtensor {


template<typename LA>
const char *kern_dmul2_i_ip_p<LA>::k_clazz = "kern_dmul2_i_ip_p";


template<typename LA>
void kern_dmul2_i_ip_p<LA>::run(const loop_registers<2, 1> &r) {

    LA::mul2_i_ip_p_x(0, m_ni, m_np, r.m_ptra[0], m_sia, r.m_ptra[1], m_spb,
        r.m_ptrb[0], m_sic, m_d);
}


template<typename LA>
kernel_base<2, 1> *kern_dmul2_i_ip_p<LA>::match(const kern_dmul2_x_p_p<LA> &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;
    if(z.m_spa != 1) return 0;

    //  1. If spa == 1, minimize sia > 0:
    //  ----------------
    //  w   a       b  c
    //  np  1       1  0
    //  ni  k2a*np  0  1  -->  c_i = a_i$p b_p
    //  ----------------       sz(i) = w2, sz(p) = np, sz($) = k2a
    //                         [i_ip_p]

    iterator_t ii = in.end();
    size_t sia_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) > 0 && i->stepa(1) == 0 && i->stepb(0) == 1) {
            if(i->stepa(0) % z.m_np) continue;
            if(sia_min == 0 || sia_min > i->stepa(0)) {
                ii = i; sia_min = i->stepa(0);
            }
        }
    }
    if(ii == in.end()) return 0;

    kern_dmul2_i_ip_p zz;
    zz.m_d = z.m_d;
    zz.m_ni = ii->weight();
    zz.m_np = z.m_np;
    zz.m_sia = ii->stepa(0);
    zz.m_spb = z.m_spb;
    zz.m_sic = ii->stepb(0);
    in.splice(out.begin(), out, ii);

    kernel_base<2, 1> *kern = 0;

    if(kern = kern_dmul2_ij_jp_ip<LA>::match(zz, in, out)) return kern;

    return new kern_dmul2_i_ip_p(zz);
}


} // namespace libtensor

#endif // LIBTENSOR_KERN_DMUL2_I_IP_P_IMPL_H
