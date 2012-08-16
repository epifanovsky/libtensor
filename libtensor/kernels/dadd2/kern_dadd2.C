#include "../kern_dadd2.h"
#include "kern_dadd2_i_i_x_x.h"
#include "kern_dadd2_i_x_i_x.h"

namespace libtensor {


const char *kern_dadd2::k_clazz = "kern_dadd2";


void kern_dadd2::run(const loop_registers<2, 1> &r) {

    r.m_ptrb[0][0] += (r.m_ptra[0][0] * m_ka + r.m_ptra[1][0] * m_kb) * m_d;

}


kernel_base<2, 1> *kern_dadd2::match(double ka, double kb, double d,
    list_t &in, list_t &out) {

    kernel_base<2, 1> *kern = 0;

    kern_dadd2 zz;
    zz.m_ka = ka;
    zz.m_kb = kb;
    zz.m_d = d;

    if(kern = kern_dadd2_i_i_x_x::match(zz, in, out)) return kern;
    if(kern = kern_dadd2_i_x_i_x::match(zz, in, out)) return kern;

    return new kern_dadd2(zz);
}


} // namespace libtensor
