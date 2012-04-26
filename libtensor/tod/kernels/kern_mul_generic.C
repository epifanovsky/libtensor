#include "kern_mul_generic.h"
#include "kern_mul_i_i_i.h"
#include "kern_mul_i_x_i.h"
#include "kern_mul_i_i_x.h"
#include "kern_mul_x_p_p.h"

namespace libtensor {


const char *kern_mul_generic::k_clazz = "kern_mul_generic";


void kern_mul_generic::run(const loop_registers<2, 1> &r) {

    r.m_ptrb[0][0] += r.m_ptra[0][0] * r.m_ptra[1][0] * m_d;

}


kernel_base<2, 1> *kern_mul_generic::match(double d, list_t &in, list_t &out) {

    kernel_base<2, 1> *kern = 0;

    kern_mul_generic zz;
    zz.m_d = d;

    if(kern = kern_mul_i_i_x::match(zz, in, out)) return kern;
    if(kern = kern_mul_i_x_i::match(zz, in, out)) return kern;
    if(kern = kern_mul_x_p_p::match(zz, in, out)) return kern;
    if(kern = kern_mul_i_i_i::match(zz, in, out)) return kern;

    return new kern_mul_generic(zz);
}


} // namespace libtensor
