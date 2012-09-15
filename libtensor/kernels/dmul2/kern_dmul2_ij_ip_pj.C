#include <libtensor/linalg/linalg.h>
#include "kern_dmul2_ij_ip_pj.h"

namespace libtensor {


const char *kern_dmul2_ij_ip_pj::k_clazz = "kern_dmul2_ij_ip_pj";


void kern_dmul2_ij_ip_pj::run(const loop_registers<2, 1> &r) {

    linalg::mul2_ij_ip_pj_x(m_ni, m_nj, m_np, r.m_ptra[0], m_sia, r.m_ptra[1],
        m_spb, r.m_ptrb[0], m_sic, m_d);
}


kernel_base<2, 1> *kern_dmul2_ij_ip_pj::match(const kern_dmul2_i_p_pi &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;
    if(z.m_spa != 1 || z.m_sic != 1) return 0;

    //  Rename i -> j.

    //  1. Minimize sia > 0.
    //  -----------------
    //  w   a    b    c
    //  nj  0    1    1
    //  np  1    spb  0
    //  ni  sia  0    sic  --> c_j#i = a_j#p b_p#i
    //  -----------------      [ij_ip_pj]
    //

    iterator_t ii = in.end();
    size_t sia_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) > 0 && i->stepa(1) == 0 && i->stepb(0) > 0) {
            if(i->stepa(0) % z.m_np) continue;
            if(i->stepb(0) % z.m_ni) continue;
            if(sia_min == 0 || sia_min > i->stepa(0)) {
                ii = i; sia_min = i->stepa(0);
            }
        }
    }
    if(ii == in.end()) return 0;

    kern_dmul2_ij_ip_pj zz;
    zz.m_d = z.m_d;
    zz.m_ni = ii->weight();
    zz.m_nj = z.m_ni;
    zz.m_np = z.m_np;
    zz.m_sia = ii->stepa(0);
    zz.m_spb = z.m_spb;
    zz.m_sic = ii->stepb(0);
    in.splice(out.begin(), out, ii);

    kernel_base<2, 1> *kern = 0;

    return new kern_dmul2_ij_ip_pj(zz);
}


} // namespace libtensor
