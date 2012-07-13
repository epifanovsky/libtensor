#include <libtensor/linalg/linalg.h>
#include "kern_dcopy_ij_ji_x.h"

namespace libtensor {


const char *kern_dcopy_ij_ji_x::k_clazz = "kern_dcopy_ij_ji_x";


void kern_dcopy_ij_ji_x::run(const loop_registers<1, 1> &r) {

    linalg::ij_ji_x(m_ni, m_nj, r.m_ptra[0], m_sja, m_d, r.m_ptrb[0], m_sib);
}


kernel_base<1, 1> *kern_dcopy_ij_ji_x::match(const kern_dcopy_i_i_x &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;
    if(z.m_sib != 1) return 0;

    //  Rename i->j

    //    Minimize sib > 0:
    //    ----------
    //    w   a   b
    //    nj  sja 1
    //    ni  1   sib  -->  b_i#j = a_j%i d
    //    ----------        [ij_ji_x]

    iterator_t ii = in.end();
    size_t sib_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) == 1 && i->stepb(0) > 0) {
            if(sib_min == 0 || sib_min > i->stepb(0)) {
                ii = i; sib_min = i->stepb(0);
            }
        }
    }
    if(ii == in.end()) return 0;

    kern_dcopy_ij_ji_x zz;
    zz.m_d = z.m_d;
    zz.m_ni = ii->weight();
    zz.m_nj = z.m_ni;
    zz.m_sja = z.m_sia;
    zz.m_sib = ii->stepb(0);
    in.splice(out.begin(), out, ii);

    kernel_base<1, 1> *kern = 0;

    return new kern_dcopy_ij_ji_x(zz);
}


} // namespace libtensor
