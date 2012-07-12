#include "../kern_dcopy.h"
#include "kern_dcopy_i_i_x.h"

namespace libtensor {


const char *kern_dcopy::k_clazz = "kern_dcopy";


void kern_dcopy::run(const loop_registers<1, 1> &r) {

    r.m_ptrb[0][0] = r.m_ptra[0][0] * m_d;
}


kernel_base<1, 1> *kern_dcopy::match(double d, list_t &in, list_t &out) {

    kernel_base<1, 1> *kern = 0;

    kern_dcopy zz;
    zz.m_d = d;

    if(kern = kern_dcopy_i_i_x::match(zz, in, out)) return kern;

    return new kern_dcopy(zz);
}


} // namespace libtensor
