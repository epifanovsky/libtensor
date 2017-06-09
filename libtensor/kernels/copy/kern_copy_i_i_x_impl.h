#ifndef LIBTENSOR_KERN_COPY_I_I_X_IMPL_H
#define LIBTENSOR_KERN_COPY_I_I_X_IMPL_H

#include "kern_copy_i_i_x.h"
#include "kern_copy_ij_ij_x.h"
#include "kern_copy_ij_ji_x.h"

namespace libtensor {


template<typename LA, typename T>
const char *kern_copy_i_i_x<LA, T>::k_clazz = "kern_copy_i_i_x";


template<typename LA, typename T>
void kern_copy_i_i_x<LA, T>::run(
    device_context_ref ctx,
    const loop_registers_x<1, 1, T> &r) {

    LA::copy_i_i(ctx, m_ni, r.m_ptra[0], m_sia, r.m_ptrb[0], 1); // FIXME need to generalize this in linalg to make sure that it works for floats
    if(m_d != 1.0) {
        LA::mul1_i_x(ctx, m_ni, m_d, r.m_ptrb[0], 1);
    }
}


template<typename LA, typename T>
kernel_base<LA, 1, 1> *kern_copy_i_i_x<LA, T>::match(const kern_copy<LA, T> &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //    Minimize sia > 0:
    //    ----------
    //    w   a   b
    //    ni  sia 1   -->  b_i = a_i# d
    //    ----------       [mul2_i_i_x]

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

    kern_copy_i_i_x zz;
    zz.m_d = z.m_d;
    zz.m_ni = ii->weight();
    zz.m_sia = ii->stepa(0);
    zz.m_sib = 1;
    in.splice(out.begin(), out, ii);

    kernel_base<LA, 1, 1> *kern = 0;

    if((kern = kern_copy_ij_ij_x<LA, T>::match(zz, in, out))) return kern;
    if((kern = kern_copy_ij_ji_x<LA, T>::match(zz, in, out))) return kern;

    return new kern_copy_i_i_x(zz);
}


} // namespace libtensor

#endif // LIBTENSOR_KERN_COPY_I_I_X_IMPL_H
