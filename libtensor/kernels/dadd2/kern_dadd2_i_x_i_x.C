#include <libtensor/linalg/linalg.h>
#include "kern_dadd2_i_x_i_x.h"

namespace libtensor {


const char *kern_dadd2_i_x_i_x::k_clazz = "kern_dadd2_i_x_i_x";


void kern_dadd2_i_x_i_x::run(const loop_registers<2, 1> &r) {

    linalg::add_i_i_x_x(0, m_ni, r.m_ptra[1], m_sib, m_kb, r.m_ptra[0][0], m_ka,
        r.m_ptrb[0], m_sic, m_d);
}


kernel_base<2, 1> *kern_dadd2_i_x_i_x::match(const kern_dadd2 &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //    Minimize sic > 0:
    //    -------------
    //    w   a  b  c
    //    ni  0  1  sic  -->  c_i# = (a + b_i) d
    //    -------------       [i_x_i_x]
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

    kern_dadd2_i_x_i_x zz;
    zz.m_ka = z.m_ka;
    zz.m_kb = z.m_kb;
    zz.m_d = z.m_d;
    zz.m_ni = ii->weight();
    zz.m_sib = 1;
    zz.m_sic = ii->stepb(0);
    in.splice(out.begin(), out, ii);

    kernel_base<2, 1> *kern = 0;

    return new kern_dadd2_i_x_i_x(zz);
}


} // namespace libtensor
