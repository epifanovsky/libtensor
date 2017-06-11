#ifndef LIBTENSOR_KERN_MUL2_IJ_IP_JP_IMPL_H
#define LIBTENSOR_KERN_MUL2_IJ_IP_JP_IMPL_H

#include "kern_mul2_ij_ip_jp.h"

namespace libtensor {


template<typename LA, typename T>
const char *kern_mul2_ij_ip_jp<LA, T>::k_clazz = "kern_mul2_ij_ip_jp";


template<typename LA, typename T>
void kern_mul2_ij_ip_jp<LA, T>::run(
    device_context_ref ctx,
    const loop_registers_x<2, 1, T> &r) {

    LA::mul2_ij_ip_jp_x(ctx, m_ni, m_nj, m_np, r.m_ptra[0], m_sia,
        r.m_ptra[1], m_sjb, r.m_ptrb[0], m_sic, m_d);
}


template<typename LA, typename T>
kernel_base<LA, 2, 1, T> *kern_mul2_ij_ip_jp<LA, T>::match(
    const kern_mul2_i_p_ip<LA, T> &z, list_t &in, list_t &out) {

    if(in.empty()) return 0;
    if(z.m_spa != 1 || z.m_sic != 1) return 0;

    //  Rename i->j

    //  1. Minimize sia > 0:
    //  -----------------
    //  w   a    b    c
    //  np  1    1    0
    //  nj  0    sjb  1
    //  ni  sia  0    sic  -->  c_j#i = a_j$p b_i%p
    //  -----------------       sz(i) = w2, sz(j) = w3,
    //                          sz(p) = w1
    //                          sz(#) = k3, sz($) = k2,
    //                          sz(%) = k1
    //                          [ij_ip_jp]

    iterator_t ii = in.end();
    size_t sia_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) > 0 && i->stepa(1) == 0 && i->stepb(0) > 0) {
            if(i->stepa(0) % z.m_np) continue;
            if(i->stepb(0) % z.m_ni) continue;
            if(sia_min == 0 || sia_min > i->stepa(0)) {
                ii = i; sia_min = i->stepa(0);
            }
        }
    }
    if(ii == in.end()) return 0;

    kern_mul2_ij_ip_jp zz;
    zz.m_d = z.m_d;
    zz.m_ni = ii->weight();
    zz.m_nj = z.m_ni;
    zz.m_np = z.m_np;
    zz.m_sia = ii->stepa(0);
    zz.m_sjb = z.m_sib;
    zz.m_sic = ii->stepb(0);
    in.splice(out.begin(), out, ii);

    return new kern_mul2_ij_ip_jp(zz);
}


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL2_IJ_IP_JP_IMPL_H
