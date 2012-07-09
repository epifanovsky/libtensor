#include "../../linalg/linalg.h"
#include "kern_mul_ij_j_i.h"

namespace libtensor {


const char *kern_mul_ij_j_i::k_clazz = "kern_mul_ij_j_i";


void kern_mul_ij_j_i::run(const loop_registers<2, 1> &r) {

    linalg::ij_i_j_x(m_ni, m_nj, r.m_ptra[1], m_sib, r.m_ptra[0], m_sja,
        r.m_ptrb[0], m_sic, m_d);
}


kernel_base<2, 1> *kern_mul_ij_j_i::match(const kern_dmul2_i_i_x &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;
    if(z.m_sic != 1) return 0;

    //  Rename i -> j.

    //  Minimize sib > 0:
    //  ---------------
    //  w   a  b    c
    //  nj  1  0    1
    //  ni  0  sib  sic  -->  c_i#j = a_j b_i#
    //  ---------------       [ij_j_i]
    //

    iterator_t ii = in.end();
    size_t sib_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepb(0) > 0) {
            if(i->stepb(0) % z.m_ni) continue;
            if(sib_min == 0 || sib_min > i->stepa(1)) {
                ii = i; sib_min = i->stepa(1);
            }
        }
    }
    if(ii == in.end()) return 0;

    kern_mul_ij_j_i zz;
    zz.m_d = z.m_d;
    zz.m_ni = ii->weight();
    zz.m_nj = z.m_ni;
    zz.m_sja = z.m_sia;
    zz.m_sib = ii->stepa(1);
    zz.m_sic = ii->stepb(0);
    in.splice(out.begin(), out, ii);

    kernel_base<2, 1> *kern = 0;

    return new kern_mul_ij_j_i(zz);
}


} // namespace libtensor
