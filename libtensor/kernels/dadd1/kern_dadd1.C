#include "../kern_dadd1.h"

namespace libtensor {


const char *kern_dadd1::k_clazz = "kern_dadd1";


void kern_dadd1::run(const loop_registers<1, 1> &r) {

    r.m_ptrb[0][0] += r.m_ptra[0][0] * m_d;
}


kernel_base<1, 1> *kern_dadd1::match(double d, list_t &in, list_t &out) {

    kernel_base<1, 1> *kern = 0;

    kern_dadd1 zz;
    zz.m_d = d;

    return new kern_dadd1(zz);
}


} // namespace libtensor
