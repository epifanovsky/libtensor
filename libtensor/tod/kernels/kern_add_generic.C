#include "kern_add_generic.h"
#include "kern_add_i_i_x_x.h"
#include "kern_add_i_x_i_x.h"

namespace libtensor {


const char *kern_add_generic::k_clazz = "kern_add_generic";


void kern_add_generic::run(const loop_registers<2, 1> &r) {

    r.m_ptrb[0][0] += (r.m_ptra[0][0] * m_ka + r.m_ptra[1][0] * m_kb) * m_d;

}


kernel_base<2, 1> *kern_add_generic::match(double ka, double kb, double d,
    list_t &in, list_t &out) {

    kernel_base<2, 1> *kern = 0;

    kern_add_generic zz;
    zz.m_ka = ka;
    zz.m_kb = kb;
    zz.m_d = d;

    if(kern = kern_add_i_i_x_x::match(zz, in, out)) return kern;
    if(kern = kern_add_i_x_i_x::match(zz, in, out)) return kern;

    return new kern_add_generic(zz);
}


} // namespace libtensor
