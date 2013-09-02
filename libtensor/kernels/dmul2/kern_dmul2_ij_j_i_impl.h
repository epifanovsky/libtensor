#ifndef LIBTENSOR_KERN_DMUL2_IJ_J_I_IMPL_H
#define LIBTENSOR_KERN_DMUL2_IJ_J_I_IMPL_H

#include "kern_dmul2_ij_j_i.h"

namespace libtensor {


template<typename LA>
const char *kern_dmul2_ij_j_i<LA>::k_clazz = "kern_dmul2_ij_j_i";


template<typename LA>
void kern_dmul2_ij_j_i<LA>::run(
    device_context_ref ctx,
    const loop_registers<2, 1> &r) {

    LA::mul2_ij_i_j_x(ctx, m_ni, m_nj, r.m_ptra[1], m_sib, r.m_ptra[0], m_sja,
        r.m_ptrb[0], m_sic, m_d);
}


template<typename LA>
kernel_base<LA, 2, 1> *kern_dmul2_ij_j_i<LA>::match(
    const kern_dmul2_i_i_x<LA> &z, list_t &in, list_t &out) {

    if(in.empty()) return 0;
    if(z.m_sic != 1) return 0;

    //  Rename i -> j.

    //  Minimize sib > 0:
    //  ---------------
    //  w   a  b    c
    //  nj  1  0    1
    //  ni  0  sib  sic  -->  c_i#j = a_j b_i#
    //  ---------------       [ij_j_i]
    //

    iterator_t ii = in.end();
    size_t sib_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepb(0) > 0) {
            if(i->stepb(0) % z.m_ni) continue;
            if(sib_min == 0 || sib_min > i->stepa(1)) {
                ii = i; sib_min = i->stepa(1);
            }
        }
    }
    if(ii == in.end()) return 0;

    kern_dmul2_ij_j_i zz;
    zz.m_d = z.m_d;
    zz.m_ni = ii->weight();
    zz.m_nj = z.m_ni;
    zz.m_sja = z.m_sia;
    zz.m_sib = ii->stepa(1);
    zz.m_sic = ii->stepb(0);
    in.splice(out.begin(), out, ii);

    return new kern_dmul2_ij_j_i(zz);
}


} // namespace libtensor

#endif // LIBTENSOR_KERN_DMUL2_IJ_J_I_IMPL_H
