#ifndef LIBTENSOR_KERN_ADD2_IMPL_H
#define LIBTENSOR_KERN_ADD2_IMPL_H

#include "../kern_add2.h"
#include "kern_add2_i_i_x_x.h"
#include "kern_add2_i_x_i_x.h"

namespace libtensor {


template<typename LA, typename T>
const char *kern_add2<LA, T>::k_clazz = "kern_add2";


template<typename LA, typename T>
void kern_add2<LA, T>::run(
    device_context_ref ctx,
    const loop_registers_x<2, 1, T> &r) {

    r.m_ptrb[0][0] += (r.m_ptra[0][0] * m_ka + r.m_ptra[1][0] * m_kb) * m_d;

}


template<typename LA, typename T>
kernel_base<LA, 2, 1, T> *kern_add2<LA, T>::match(T ka, T kb, T d,
    list_t &in, list_t &out) {

    kernel_base<LA, 2, 1, T> *kern = 0;

    kern_add2<LA, T> zz;
    zz.m_ka = ka;
    zz.m_kb = kb;
    zz.m_d = d;

    if((kern = kern_add2_i_i_x_x<LA, T>::match(zz, in, out))) return kern;
    if((kern = kern_add2_i_x_i_x<LA, T>::match(zz, in, out))) return kern;

    return new kern_add2(zz);
}


} // namespace libtensor

#endif // LIBTENSOR_KERN_ADD2_IMPL_H
