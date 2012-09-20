#include "../kern_dmuladd1.h"

namespace libtensor {


const char *kern_dmuladd1::k_clazz = "kern_dmuladd1";


void kern_dmuladd1::run(void*, const loop_registers<1, 1> &r) {

    r.m_ptrb[0][0] = r.m_ptrb[0][0] + r.m_ptra[0][0] * r.m_ptrb[0][0] * m_d;
}


kernel_base<linalg, 1, 1> *kern_dmuladd1::match(double d, list_t &in,
    list_t &out) {

    kernel_base<linalg, 1, 1> *kern = 0;

    kern_dmuladd1 zz;
    zz.m_d = d;

    return new kern_dmuladd1(zz);
}


} // namespace libtensor
