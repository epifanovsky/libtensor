#include "../kern_dmul2.h"
#include "kern_dmul2_i_i_i.h"
#include "kern_dmul2_i_x_i.h"
#include "kern_dmul2_i_i_x.h"
#include "kern_dmul2_x_p_p.h"

namespace libtensor {


const char *kern_dmul2::k_clazz = "kern_dmul2";


void kern_dmul2::run(const loop_registers<2, 1> &r) {

    r.m_ptrb[0][0] += r.m_ptra[0][0] * r.m_ptra[1][0] * m_d;

}


kernel_base<2, 1> *kern_dmul2::match(double d, list_t &in, list_t &out) {

    kernel_base<2, 1> *kern = 0;

    kern_dmul2 zz;
    zz.m_d = d;

    if(kern = kern_dmul2_i_i_x::match(zz, in, out)) return kern;
    if(kern = kern_dmul2_i_x_i::match(zz, in, out)) return kern;
    if(kern = kern_dmul2_x_p_p::match(zz, in, out)) return kern;
    if(kern = kern_dmul2_i_i_i::match(zz, in, out)) return kern;

    return new kern_dmul2(zz);
}


} // namespace libtensor
