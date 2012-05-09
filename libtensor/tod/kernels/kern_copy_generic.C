#include "kern_copy_generic.h"

namespace libtensor {


const char *kern_copy_generic::k_clazz = "kern_copy_generic";


void kern_copy_generic::run(const loop_registers<1, 1> &r) {

    r.m_ptrb[0][0] = r.m_ptra[0][0] * m_d;
}


kernel_base<1, 1> *kern_copy_generic::match(double d, list_t &in, list_t &out) {

    kernel_base<1, 1> *kern = 0;

    kern_copy_generic zz;
    zz.m_d = d;

    return new kern_copy_generic(zz);
}


} // namespace libtensor
