#ifndef LIBTENSOR_KERN_MUL2_IJ_JP_PI_IMPL_H
#define LIBTENSOR_KERN_MUL2_IJ_JP_PI_IMPL_H

#include "kern_mul2_ij_jp_pi.h"

namespace libtensor {


template<typename LA, typename T>
const char *kern_mul2_ij_jp_pi<LA, T>::k_clazz = "kern_mul2_ij_jp_pi";


template<typename LA, typename T>
void kern_mul2_ij_jp_pi<LA, T>::run(
    device_context_ref ctx,
    const loop_registers_x<2, 1, T> &r) {

    LA::mul2_ij_pi_jp_x(ctx, m_ni, m_nj, m_np, r.m_ptra[1], m_spb,
        r.m_ptra[0], m_sja, r.m_ptrb[0], m_sic, m_d);
}


template<typename LA, typename T>
kernel_base<LA, 2, 1, T> *kern_mul2_ij_jp_pi<LA, T>::match(
    const kern_mul2_i_p_pi<LA, T> &z, list_t &in, list_t &out) {

    if(in.empty()) return 0;
    if(z.m_spa != 1) return 0;

    //  Minimize sja > 0:
    //  -----------------
    //  w   a    b    c
    //  ni  0    1    sic
    //  np  1    spb  0
    //  nj  sja  0    1     -->  c_i#j = a_j#p b_p#i
    //  ------------------       [ij_jp_pi]
    //

    iterator_t ij = in.end();
    size_t sja_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) > 0 && i->stepa(1) == 0 && i->stepb(0) == 1) {
            if(i->stepa(0) % z.m_np) continue;
            if(z.m_sic % i->weight()) continue;
            if(sja_min == 0 || sja_min > i->stepa(0)) {
                ij = i; sja_min = i->stepa(0);
            }
        }
    }
    if(ij == in.end()) return 0;

    kern_mul2_ij_jp_pi zz;
    zz.m_d = z.m_d;
    zz.m_ni = z.m_ni;
    zz.m_nj = ij->weight();
    zz.m_np = z.m_np;
    zz.m_sja = ij->stepa(0);
    zz.m_spb = z.m_spb;
    zz.m_sic = z.m_sic;
    in.splice(out.begin(), out, ij);

    return new kern_mul2_ij_jp_pi(zz);
}


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL2_IJ_JP_PI_IMPL_H
