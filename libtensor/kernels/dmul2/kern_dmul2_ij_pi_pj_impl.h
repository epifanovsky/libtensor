#ifndef LIBTENSOR_KERN_DMUL2_IJ_PI_PJ_IMPL_H
#define LIBTENSOR_KERN_DMUL2_IJ_PI_PJ_IMPL_H

#include "kern_dmul2_ij_pi_pj.h"

namespace libtensor {


template<typename LA>
const char *kern_dmul2_ij_pi_pj<LA>::k_clazz = "kern_dmul2_ij_pi_pj";


template<typename LA>
void kern_dmul2_ij_pi_pj<LA>::run(const loop_registers<2, 1> &r) {

    LA::mul2_ij_pi_pj_x(0, m_ni, m_nj, m_np, r.m_ptra[0], m_spa,
        r.m_ptra[1], m_spb, r.m_ptrb[0], m_sic, m_d);
}


template<typename LA>
kernel_base<2, 1> *kern_dmul2_ij_pi_pj<LA>::match(
    const kern_dmul2_i_p_pi<LA> &z, list_t &in, list_t &out) {

    if(in.empty()) return 0;
    if(z.m_sic != 1) return 0;

    //  Rename i -> j

    //  1. Minimize sic > 0:
    //  -----------------
    //  w   a    b    c
    //  nj  0    1    1
    //  np  spa  spb  0
    //  ni  1    0    sic  -->  c_i#j = a_p#i b_p#j
    //  -----------------       [ij_pi_pj]
    //

    iterator_t ii = in.end();
    size_t sic_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) == 1 && i->stepa(1) == 0 && i->stepb(0) > 0) {
            if(i->stepb(0) % z.m_ni) continue;
            if(z.m_spa % i->weight()) continue;
            if(sic_min == 0 || sic_min > i->stepb(0)) {
                ii = i; sic_min = i->stepb(0);
            }
        }
    }
    if(ii == in.end()) return 0;

    kern_dmul2_ij_pi_pj zz;
    zz.m_d = z.m_d;
    zz.m_ni = ii->weight();
    zz.m_nj = z.m_ni;
    zz.m_np = z.m_np;
    zz.m_spa = z.m_spa;
    zz.m_spb = z.m_spb;
    zz.m_sic = ii->stepb(0);
    in.splice(out.begin(), out, ii);

    kernel_base<2, 1> *kern = 0;

    return new kern_dmul2_ij_pi_pj(zz);
}


template<typename LA>
kernel_base<2, 1> *kern_dmul2_ij_pi_pj<LA>::match(
    const kern_dmul2_i_pi_p<LA> &z, list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Minimize sjb > 0:
    //  ------------------
    //  w   a    b     c
    //  ni  1    0     sic
    //  np  spa  spb   0
    //  nj  0    1     1    -->  c_i#j = a_p#i b_p#j
    //  ------------------       [ij_pi_pj]
    //

    iterator_t ij = in.end();
    size_t sjb_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) == 0 && i->stepa(1) == 1 && i->stepb(0) == 1) {
            if(z.m_sic % i->weight()) continue;
            ij = i; break;
        }
    }
    if(ij == in.end()) return 0;

    kern_dmul2_ij_pi_pj zz;
    zz.m_d = z.m_d;
    zz.m_ni = z.m_ni;
    zz.m_nj = ij->weight();
    zz.m_np = z.m_np;
    zz.m_spa = z.m_spa;
    zz.m_spb = z.m_spb;
    zz.m_sic = z.m_sic;
    in.splice(out.begin(), out, ij);

    kernel_base<2, 1> *kern = 0;

    return new kern_dmul2_ij_pi_pj(zz);
}


} // namespace libtensor

#endif // LIBTENSOR_KERN_DMUL2_IJ_PI_PJ_IMPL_H
