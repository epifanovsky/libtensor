#ifndef LIBTENSOR_KERN_MUL2_I_I_I_IMPL_H
#define LIBTENSOR_KERN_MUL2_I_I_I_IMPL_H

#include "kern_mul2_i_i_i.h"

namespace libtensor {


template<typename LA, typename T>
const char *kern_mul2_i_i_i<LA, T>::k_clazz = "kern_mul2_i_i_i";


template<typename LA, typename T>
void kern_mul2_i_i_i<LA, T>::run(
    device_context_ref ctx,
    const loop_registers_x<2, 1, T> &r) {

    LA::mul2_i_i_i_x(ctx, m_ni, r.m_ptra[0], m_sia, r.m_ptra[1], m_sib,
        r.m_ptrb[0], m_sic, m_d);
}


template<typename LA, typename T>
kernel_base<LA, 2, 1, T> *kern_mul2_i_i_i<LA, T>::match(const kern_mul2<LA, T> &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //    Minimize sic > 0:
    //    -------------
    //    w   a  b  c
    //    ni  1  1  sic  -->  c_i# = a_i b_i
    //    -------------       [i_i_i]
    //

    iterator_t ii = in.end();
    size_t sic_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) == 1 && i->stepa(1) == 1 && i->stepb(0) > 0) {
            if(sic_min == 0 || sic_min > i->stepb(0)) {
                ii = i; sic_min = i->stepb(0);
            }
        }
    }
    if(ii == in.end()) return 0;

    kern_mul2_i_i_i zz;
    zz.m_d = z.m_d;
    zz.m_ni = ii->weight();
    zz.m_sia = 1;
    zz.m_sib = 1;
    zz.m_sic = ii->stepb(0);
    in.splice(out.begin(), out, ii);

    return new kern_mul2_i_i_i(zz);
}


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL2_I_I_I_IMPL_H
