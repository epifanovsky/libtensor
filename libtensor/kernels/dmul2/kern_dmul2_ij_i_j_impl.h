#ifndef LIBTENSOR_KERN_DMUL2_IJ_I_J_IMPL_H
#define LIBTENSOR_KERN_DMUL2_IJ_I_J_IMPL_H

#include "kern_dmul2_ij_i_j.h"
#include "kern_dmul2_ij_ip_pj.h"

namespace libtensor {


template<typename LA>
const char *kern_dmul2_ij_i_j<LA>::k_clazz = "kern_dmul2_ij_i_j";


template<typename LA>
void kern_dmul2_ij_i_j<LA>::run(
    device_context_ref ctx,
    const loop_registers<2, 1> &r) {

    LA::mul2_ij_i_j_x(ctx, m_ni, m_nj, r.m_ptra[0], m_sia, r.m_ptra[1], m_sjb,
        r.m_ptrb[0], m_sic, m_d);
}


template<typename LA>
kernel_base<LA, 2, 1> *kern_dmul2_ij_i_j<LA>::match(
    const kern_dmul2_i_i_x<LA> &z, list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Minimize sjb > 0:
    //  ---------------
    //  w   a  b    c
    //  ni  1  0    sic
    //  nj  0  sjb  1    -->  c_i#j = a_i b_j#
    //  ---------------       [ij_i_j]
    //

    iterator_t ij = in.end();
    size_t sjb_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepb(0) == 1) {
            if(z.m_sic % (i->weight() * i->stepb(0))) continue;
            if(sjb_min == 0 || sjb_min > i->stepa(1)) {
                ij = i; sjb_min = i->stepa(1);
            }
        }
    }
    if(ij == in.end()) return 0;

    kern_dmul2_ij_i_j zz;
    zz.m_d = z.m_d;
    zz.m_ni = z.m_ni;
    zz.m_nj = ij->weight();
    zz.m_sia = z.m_sia;
    zz.m_sjb = ij->stepa(1);
    zz.m_sic = z.m_sic;
    in.splice(out.begin(), out, ij);

    kernel_base<LA, 2, 1> *kern = 0;

    return new kern_dmul2_ij_i_j(zz);
}


} // namespace libtensor

#endif // LIBTENSOR_KERN_DMUL2_IJ_I_J_IMPL_H
