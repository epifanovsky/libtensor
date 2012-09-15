#include <libtensor/linalg/linalg.h>
#include "kern_dmul2_ij_i_j.h"
#include "kern_dmul2_ij_ip_pj.h"

namespace libtensor {


const char *kern_dmul2_ij_i_j::k_clazz = "kern_dmul2_ij_i_j";


void kern_dmul2_ij_i_j::run(const loop_registers<2, 1> &r) {

    linalg::mul2_ij_i_j_x(0, m_ni, m_nj, r.m_ptra[0], m_sia, r.m_ptra[1], m_sjb,
        r.m_ptrb[0], m_sic, m_d);
}


kernel_base<2, 1> *kern_dmul2_ij_i_j::match(const kern_dmul2_i_i_x &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Minimize sjb > 0:
    //  ---------------
    //  w   a  b    c
    //  ni  1  0    sic
    //  nj  0  sjb  1    -->  c_i#j = a_i b_j#
    //  ---------------       [ij_i_j]
    //

    iterator_t ij = in.end();
    size_t sjb_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepb(0) == 1) {
            if(z.m_sic % (i->weight() * i->stepb(0))) continue;
            if(sjb_min == 0 || sjb_min > i->stepa(1)) {
                ij = i; sjb_min = i->stepa(1);
            }
        }
    }
    if(ij == in.end()) return 0;

    kern_dmul2_ij_i_j zz;
    zz.m_d = z.m_d;
    zz.m_ni = z.m_ni;
    zz.m_nj = ij->weight();
    zz.m_sia = z.m_sia;
    zz.m_sjb = ij->stepa(1);
    zz.m_sic = z.m_sic;
    in.splice(out.begin(), out, ij);

    kernel_base<2, 1> *kern = 0;

//  if(kern = kern_mul_ij_ip_pj::match(zz, in, out)) return kern;

    return new kern_dmul2_ij_i_j(zz);
}


} // namespace libtensor
