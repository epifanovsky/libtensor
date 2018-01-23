#ifndef LIBTENSOR_KERN_DMUL2_IMPL_H
#define LIBTENSOR_KERN_DMUL2_IMPL_H

#include "../kern_dmul2.h"
#include "kern_dmul2_i_i_i.h"
#include "kern_dmul2_i_x_i.h"
#include "kern_dmul2_i_i_x.h"
#include "kern_dmul2_x_p_p.h"

namespace libtensor {


template<typename LA>
const char *kern_dmul2<LA>::k_clazz = "kern_dmul2";


template<typename LA>
void kern_dmul2<LA>::run(
    device_context_ref ctx,
    const loop_registers<2, 1> &r) {

    r.m_ptrb[0][0] += r.m_ptra[0][0] * r.m_ptra[1][0] * m_d;

}


template<typename LA>
kernel_base<LA, 2, 1, double> *kern_dmul2<LA>::match(double d, list_t &in,
    list_t &out) {

    kernel_base<LA, 2, 1, double> *kern = 0;

    kern_dmul2 zz;
    zz.m_d = d;

    if((kern = kern_dmul2_i_i_x<LA>::match(zz, in, out))) return kern;
    if((kern = kern_dmul2_i_x_i<LA>::match(zz, in, out))) return kern;
    if((kern = kern_dmul2_x_p_p<LA>::match(zz, in, out))) return kern;
    if((kern = kern_dmul2_i_i_i<LA>::match(zz, in, out))) return kern;

    return new kern_dmul2(zz);
}


} // namespace libtensor

#endif // LIBTENSOR_KERN_DMUL2_IMPL_H
