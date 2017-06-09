#ifndef LIBTENSOR_KERN_COPY_IMPL_H
#define LIBTENSOR_KERN_COPY_IMPL_H

#include "../kern_copy.h"
#include "kern_copy_i_i_x.h"

namespace libtensor {


template<typename LA, typename T>
const char *kern_copy<LA, T>::k_clazz = "kern_copy";


template<typename LA, typename T>
void kern_copy<LA, T>::run(
    device_context_ref ctx,
    const loop_registers_x<1, 1,T> &r) {

    r.m_ptrb[0][0] = r.m_ptra[0][0] * m_d;
}


template<typename LA, typename T>
kernel_base<LA, 1, 1> *kern_copy<LA,T>::match(T d, list_t &in,
    list_t &out) {

    kernel_base<LA, 1, 1> *kern = 0;

    kern_copy zz;
    zz.m_d = d;

    if((kern = kern_copy_i_i_x<LA,T>::match(zz, in, out))) return kern;

    return new kern_copy(zz);
}


} // namespace libtensor

#endif // LIBTENSOR_KERN_COPY_IMPL_H
