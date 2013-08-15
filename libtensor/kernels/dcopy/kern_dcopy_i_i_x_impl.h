#ifndef LIBTENSOR_KERN_DCOPY_I_I_X_IMPL_H
#define LIBTENSOR_KERN_DCOPY_I_I_X_IMPL_H

#include "kern_dcopy_i_i_x.h"
#include "kern_dcopy_ij_ij_x.h"
#include "kern_dcopy_ij_ji_x.h"

namespace libtensor {


template<typename LA>
const char *kern_dcopy_i_i_x<LA>::k_clazz = "kern_dcopy_i_i_x";


template<typename LA>
void kern_dcopy_i_i_x<LA>::run(
    device_context_ref ctx,
    const loop_registers<1, 1> &r) {

    LA::copy_i_i(ctx, m_ni, r.m_ptra[0], m_sia, r.m_ptrb[0], 1);
    if(m_d != 1.0) {
        LA::mul1_i_x(ctx, m_ni, m_d, r.m_ptrb[0], 1);
    }
}


template<typename LA>
kernel_base<LA, 1, 1> *kern_dcopy_i_i_x<LA>::match(const kern_dcopy<LA> &z,
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

    kern_dcopy_i_i_x zz;
    zz.m_d = z.m_d;
    zz.m_ni = ii->weight();
    zz.m_sia = ii->stepa(0);
    zz.m_sib = 1;
    in.splice(out.begin(), out, ii);

    kernel_base<LA, 1, 1> *kern = 0;

    if((kern = kern_dcopy_ij_ij_x<LA>::match(zz, in, out))) return kern;
    if((kern = kern_dcopy_ij_ji_x<LA>::match(zz, in, out))) return kern;

    return new kern_dcopy_i_i_x(zz);
}


} // namespace libtensor

#endif // LIBTENSOR_KERN_DCOPY_I_I_X_IMPL_H
