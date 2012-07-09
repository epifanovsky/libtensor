#include "../kern_dmul1.h"

namespace libtensor {


const char *kern_dmul1::k_clazz = "kern_dmul1";


void kern_dmul1::run(const loop_registers<1, 1> &r) {

    r.m_ptrb[0][0] *= r.m_ptra[0][0] * m_d;
}


kernel_base<1, 1> *kern_dmul1::match(double d, list_t &in, list_t &out) {

    kernel_base<1, 1> *kern = 0;

    kern_dmul1 zz;
    zz.m_d = d;

    return new kern_dmul1(zz);
}


} // namespace libtensor
