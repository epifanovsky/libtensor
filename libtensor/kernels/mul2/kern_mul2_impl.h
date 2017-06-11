#ifndef LIBTENSOR_KERN_MUL2_IMPL_H
#define LIBTENSOR_KERN_MUL2_IMPL_H

#include "../kern_mul2.h"
#include "kern_mul2_i_i_i.h"
#include "kern_mul2_i_x_i.h"
#include "kern_mul2_i_i_x.h"
#include "kern_mul2_x_p_p.h"

namespace libtensor {


template<typename LA, typename T>
const char *kern_mul2<LA, T>::k_clazz = "kern_mul2";


template<typename LA, typename T>
void kern_mul2<LA, T>::run(
    device_context_ref ctx,
    const loop_registers_x<2, 1, T> &r) {

    r.m_ptrb[0][0] += r.m_ptra[0][0] * r.m_ptra[1][0] * m_d;

}


template<typename LA, typename T>
kernel_base<LA, 2, 1, T> *kern_mul2<LA, T>::match(T d, list_t &in,
    list_t &out) {

    kernel_base<LA, 2, 1, T> *kern = 0;

    kern_mul2 zz;
    zz.m_d = d;

    if((kern = kern_mul2_i_i_x<LA, T>::match(zz, in, out))) return kern;
    if((kern = kern_mul2_i_x_i<LA, T>::match(zz, in, out))) return kern;
    if((kern = kern_mul2_x_p_p<LA, T>::match(zz, in, out))) return kern;
    if((kern = kern_mul2_i_i_i<LA, T>::match(zz, in, out))) return kern;

    return new kern_mul2(zz);
}


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL2_IMPL_H
