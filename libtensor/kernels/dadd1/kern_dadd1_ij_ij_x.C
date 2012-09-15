#include <libtensor/linalg/linalg.h>
#include "kern_dadd1_ij_ij_x.h"

namespace libtensor {


const char *kern_dadd1_ij_ij_x::k_clazz = "kern_dadd1_ij_ij_x";


void kern_dadd1_ij_ij_x::run(const loop_registers<1, 1> &r) {

    linalg::add1_ij_ij_x(0, m_ni, m_nj, r.m_ptra[0], m_sia, m_d,
        r.m_ptrb[0], m_sib);
}


kernel_base<1, 1> *kern_dadd1_ij_ij_x::match(const kern_dadd1_i_i_x &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;
    if(z.m_sia != 1 || z.m_sib != 1) return 0;

    //  Rename i->j

    //    Minimize sia > 0:
    //    ----------
    //    w   a   b
    //    nj  1   1
    //    ni  sia sib  -->  b_i#j = a_i%j d
    //    ----------        [copy_ij_ij_x]

    iterator_t ii = in.end();
    size_t sia_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) > 0 && i->stepb(0) > 0) {
            if(sia_min == 0 || sia_min > i->stepa(0)) {
                ii = i; sia_min = i->stepa(0);
            }
        }
    }
    if(ii == in.end()) return 0;

    kern_dadd1_ij_ij_x zz;
    zz.m_d = z.m_d;
    zz.m_ni = ii->weight();
    zz.m_nj = z.m_ni;
    zz.m_sia = ii->stepa(0);
    zz.m_sib = ii->stepb(0);
    in.splice(out.begin(), out, ii);

    kernel_base<1, 1> *kern = 0;

    return new kern_dadd1_ij_ij_x(zz);
}


} // namespace libtensor
