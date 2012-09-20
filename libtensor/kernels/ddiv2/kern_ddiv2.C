#include "../kern_ddiv2.h"

namespace libtensor {


const char *kern_ddiv2::k_clazz = "kern_ddiv2";


void kern_ddiv2::run(void*, const loop_registers<2, 1> &r) {

    r.m_ptrb[0][0] += m_d * r.m_ptra[0][0] / r.m_ptra[1][0];

}


kernel_base<linalg, 2, 1> *kern_ddiv2::match(double d, list_t &in,
    list_t &out) {

    kernel_base<linalg, 2, 1> *kern = 0;

    kern_ddiv2 zz;
    zz.m_d = d;

    return new kern_ddiv2(zz);
}


} // namespace libtensor
