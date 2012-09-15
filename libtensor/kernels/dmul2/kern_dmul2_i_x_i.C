#include <libtensor/linalg/linalg.h>
#include "kern_dmul2_i_x_i.h"
#include "kern_dmul2_i_p_pi.h"

namespace libtensor {


const char *kern_dmul2_i_x_i::k_clazz = "kern_dmul2_i_x_i";


void kern_dmul2_i_x_i::run(const loop_registers<2, 1> &r) {

    linalg::mul2_i_i_x(m_ni, r.m_ptra[1], m_sib, r.m_ptra[0][0] * m_d,
        r.m_ptrb[0], m_sic);
}


kernel_base<2, 1> *kern_dmul2_i_x_i::match(const kern_dmul2 &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //    Minimize sic > 0:
    //    -------------
    //    w   a  b  c
    //    ni  0  1  sic  -->  c_i# = a b_i
    //    -------------       [i_x_i]
    //

    iterator_t ii = in.end();
    size_t sic_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) == 0 && i->stepa(1) == 1 && i->stepb(0) > 0) {
            if(sic_min == 0 || sic_min > i->stepb(0)) {
                ii = i; sic_min = i->stepb(0);
            }
        }
    }
    if(ii == in.end()) return 0;

    kern_dmul2_i_x_i zz;
    zz.m_d = z.m_d;
    zz.m_ni = ii->weight();
    zz.m_sib = 1;
    zz.m_sic = ii->stepb(0);
    in.splice(out.begin(), out, ii);

    kernel_base<2, 1> *kern = 0;

    if(kern = kern_dmul2_i_p_pi::match(zz, in, out)) return kern;

    return new kern_dmul2_i_x_i(zz);
}


} // namespace libtensor
