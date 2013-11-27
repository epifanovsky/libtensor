#ifndef LIBTENSOR_KERN_DMUL2_IJ_IP_PJ_IMPL_H
#define LIBTENSOR_KERN_DMUL2_IJ_IP_PJ_IMPL_H

#include "kern_dmul2_ij_ip_pj.h"

namespace libtensor {


template<typename LA>
const char *kern_dmul2_ij_ip_pj<LA>::k_clazz = "kern_dmul2_ij_ip_pj";


template<typename LA>
void kern_dmul2_ij_ip_pj<LA>::run(
    device_context_ref ctx,
    const loop_registers<2, 1> &r) {

    LA::mul2_ij_ip_pj_x(ctx, m_ni, m_nj, m_np, r.m_ptra[0], m_sia,
        r.m_ptra[1], m_spb, r.m_ptrb[0], m_sic, m_d);
}


template<typename LA>
kernel_base<LA, 2, 1> *kern_dmul2_ij_ip_pj<LA>::match(
    const kern_dmul2_i_p_pi<LA> &z, list_t &in, list_t &out) {

    if(in.empty()) return 0;
    if(z.m_spa != 1 || z.m_sic != 1) return 0;

    //  Rename i -> j.

    //  1. Minimize sia > 0.
    //  -----------------
    //  w   a    b    c
    //  nj  0    1    1
    //  np  1    spb  0
    //  ni  sia  0    sic  --> c_j#i = a_j#p b_p#i
    //  -----------------      [ij_ip_pj]
    //

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

    kern_dmul2_ij_ip_pj zz;
    zz.m_d = z.m_d;
    zz.m_ni = ii->weight();
    zz.m_nj = z.m_ni;
    zz.m_np = z.m_np;
    zz.m_sia = ii->stepa(0);
    zz.m_spb = z.m_spb;
    zz.m_sic = ii->stepb(0);
    in.splice(out.begin(), out, ii);

    return new kern_dmul2_ij_ip_pj(zz);
}


} // namespace libtensor

#endif // LIBTENSOR_KERN_DMUL2_IJ_IP_PJ_IMPL_H
