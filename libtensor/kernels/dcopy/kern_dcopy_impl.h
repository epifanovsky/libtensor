#ifndef LIBTENSOR_KERN_DCOPY_IMPL_H
#define LIBTENSOR_KERN_DCOPY_IMPL_H

#include "../kern_dcopy.h"
#include "kern_dcopy_i_i_x.h"

namespace libtensor {


template<typename LA>
const char *kern_dcopy<LA>::k_clazz = "kern_dcopy";


template<typename LA>
void kern_dcopy<LA>::run(
    device_context_ref ctx,
    const loop_registers<1, 1> &r) {

    r.m_ptrb[0][0] = r.m_ptra[0][0] * m_d;
}


template<typename LA>
kernel_base<LA, 1, 1> *kern_dcopy<LA>::match(double d, list_t &in,
    list_t &out) {

    kernel_base<LA, 1, 1> *kern = 0;

    kern_dcopy zz;
    zz.m_d = d;

    if(kern = kern_dcopy_i_i_x<LA>::match(zz, in, out)) return kern;

    return new kern_dcopy(zz);
}


} // namespace libtensor

#endif // LIBTENSOR_KERN_DCOPY_IMPL_H
