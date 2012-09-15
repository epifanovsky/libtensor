#include "../kern_ddiv1.h"

namespace libtensor {


const char *kern_ddiv1::k_clazz = "kern_ddiv1";


void kern_ddiv1::run(void*, const loop_registers<1, 1> &r) {

    r.m_ptrb[0][0] = (r.m_ptrb[0][0] * m_d) / r.m_ptra[0][0];
}


kernel_base<linalg, 1, 1> *kern_ddiv1::match(double d, list_t &in,
    list_t &out) {

    kernel_base<linalg, 1, 1> *kern = 0;

    kern_ddiv1 zz;
    zz.m_d = d;

    return new kern_ddiv1(zz);
}


} // namespace libtensor
