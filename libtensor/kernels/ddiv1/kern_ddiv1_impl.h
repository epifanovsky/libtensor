#ifndef LIBTENSOR_KERN_DDIV1_IMPL_H
#define LIBTENSOR_KERN_DDIV1_IMPL_H

#include "../kern_ddiv1.h"
#include "kern_ddiv1_i_i_x.h"

namespace libtensor {


template<typename LA>
const char *kern_ddiv1<LA>::k_clazz = "kern_ddiv1";


template<typename LA>
void kern_ddiv1<LA>::run(
    device_context_ref ctx,
    const loop_registers<1, 1> &r) {

    r.m_ptrb[0][0] = (r.m_ptrb[0][0] * m_d) / r.m_ptra[0][0];
}


template<typename LA>
kernel_base<LA, 1, 1> *kern_ddiv1<LA>::match(double d, list_t &in,
    list_t &out) {

    kernel_base<LA, 1, 1> *kern = 0;

    kern_ddiv1 zz;
    zz.m_d = d;

    if(kern = kern_ddiv1_i_i_x<LA>::match(zz, in, out)) return kern;

    return new kern_ddiv1(zz);
}


} // namespace libtensor

#endif // LIBTENSOR_KERN_DDIV1_IMPL_H
