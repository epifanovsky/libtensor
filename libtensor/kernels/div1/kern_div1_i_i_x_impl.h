#ifndef LIBTENSOR_KERN_DIV1_I_I_X_IMPL_H
#define LIBTENSOR_KERN_DIV1_I_I_X_IMPL_H

#include "kern_div1_i_i_x.h"

namespace libtensor {


template<typename LA, typename T>
const char *kern_div1_i_i_x<LA, T>::k_clazz = "kern_div1_i_i_x";


template<typename LA, typename T>
void kern_div1_i_i_x<LA, T>::run(
    device_context_ref ctx,
    const loop_registers_x<1, 1, T> &r) {

    LA::div1_i_i_x(ctx, m_ni, r.m_ptra[0], m_sia, r.m_ptrb[0], 1, m_d);
}


template<typename LA, typename T>
kernel_base<LA, 1, 1, T> *kern_div1_i_i_x<LA, T>::match(const kern_div1<LA, T> &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //    Minimize sia > 0:
    //    ----------
    //    w   a   b
    //    ni  sia 1   -->  b_i = a_i# d
    //    ----------       [div2_i_i_x]

    iterator_t ii = in.end();
    size_t sia_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) > 0 && i->stepb(0) == 1) {
            if(sia_min == 0 || sia_min > i->stepa(0)) {
                ii = i; sia_min = i->stepa(0);
            }
        }
    }
    if(ii == in.end()) return 0;

    kern_div1_i_i_x<LA, T> zz;
    zz.m_d = z.m_d;
    zz.m_ni = ii->weight();
    zz.m_sia = ii->stepa(0);
    zz.m_sib = 1;
    in.splice(out.begin(), out, ii);

    return new kern_div1_i_i_x(zz);
}


} // namespace libtensor

#endif // LIBTENSOR_KERN_DIV1_I_I_X_IMPL_H
