#ifndef LIBTENSOR_KERN_DADD1_IMPL_H
#define LIBTENSOR_KERN_DADD1_IMPL_H

#include "../kern_dadd1.h"
#include "kern_dadd1_i_i_x.h"

namespace libtensor {


template<typename LA>
const char *kern_dadd1<LA>::k_clazz = "kern_dadd1";


template<typename LA>
void kern_dadd1<LA>::run(const loop_registers<1, 1> &r) {

    r.m_ptrb[0][0] += r.m_ptra[0][0] * m_d;
}


template<typename LA>
kernel_base<1, 1> *kern_dadd1<LA>::match(double d, list_t &in, list_t &out) {

    kernel_base<1, 1> *kern = 0;

    kern_dadd1 zz;
    zz.m_d = d;

    if(kern = kern_dadd1_i_i_x<LA>::match(zz, in, out)) return kern;

    return new kern_dadd1(zz);
}


} // namespace libtensor

#endif // LIBTENSOR_KERN_DADD1_IMPL_H
