#ifndef LIBTENSOR_KERN_DMUL2_IJ_PJ_PI_IMPL_H
#define LIBTENSOR_KERN_DMUL2_IJ_PJ_PI_IMPL_H

#include "kern_dmul2_ij_pj_pi.h"

namespace libtensor {


template<typename LA>
const char *kern_dmul2_ij_pj_pi<LA>::k_clazz = "kern_dmul2_ij_pj_pi";


template<typename LA>
void kern_dmul2_ij_pj_pi<LA>::run(
    device_context_ref ctx,
    const loop_registers<2, 1> &r) {

    LA::mul2_ij_pi_pj_x(ctx, m_ni, m_nj, m_np, r.m_ptra[1], m_spb,
        r.m_ptra[0], m_spa, r.m_ptrb[0], m_sic, m_d);
}


template<typename LA>
kernel_base<LA, 2, 1, double> *kern_dmul2_ij_pj_pi<LA>::match(
    const kern_dmul2_i_pi_p<LA> &z, list_t &in, list_t &out) {

    if(in.empty()) return 0;
    if(z.m_sic != 1) return 0;

    //  Rename i->j

    //  2. Minimize sic > 0:
    //  ------------------
    //  w   a    b    c
    //  nj  1    0    1
    //  np  spa  spb  0
    //  ni  0    1    sic  --> c_j#i = a_p$i b_p%j
    //  -----------------      sz(i) = w1, sz(j) = w3,
    //                         sz(p) = w2,
    //                         sz(#) = k4, sz($) = k2,
    //                         sz(%) = k3'
    //                         [ij_pj_pi]

    iterator_t ii = in.end();
    size_t sic_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) == 0 && i->stepa(1) == 1 && i->stepb(0) > 0) {
            if(z.m_spb % i->weight()) continue;
            if(i->stepb(0) % z.m_ni) continue;
            if(sic_min == 0 || sic_min > i->stepb(0)) {
                ii = i; sic_min = i->stepb(0);
            }
        }
    }
    if(ii == in.end()) return 0;

    kern_dmul2_ij_pj_pi zz;
    zz.m_d = z.m_d;
    zz.m_ni = ii->weight();
    zz.m_nj = z.m_ni;
    zz.m_np = z.m_np;
    zz.m_spa = z.m_spa;
    zz.m_spb = z.m_spb;
    zz.m_sic = ii->stepb(0);
    in.splice(out.begin(), out, ii);

    return new kern_dmul2_ij_pj_pi(zz);
}


} // namespace libtensor

#endif // LIBTENSOR_KERN_DMUL2_IJ_PJ_PI_IMPL_H
