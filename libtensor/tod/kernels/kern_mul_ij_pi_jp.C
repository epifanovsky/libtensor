#include "../../linalg/linalg.h"
#include "kern_mul_ij_pi_jp.h"

namespace libtensor {


const char *kern_mul_ij_pi_jp::k_clazz = "kern_mul_ij_pi_jp";


void kern_mul_ij_pi_jp::run(const loop_registers<2, 1> &r) {

    linalg::ij_pi_jp_x(m_ni, m_nj, m_np, r.m_ptra[0], m_spa, r.m_ptra[1],
        m_sjb, r.m_ptrb[0], m_sic, m_d);
}


kernel_base<2, 1> *kern_mul_ij_pi_jp::match(const kern_mul_i_pi_p &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;
    if(z.m_spb != 1) return 0;

    //  1. Minimize sjb > 0:
    //  -----------------
    //  w   a    b    c
    //  ni  1    0    sic
    //  np  spa  1    0
    //  nj  0    sjb  1    -->  c_i#j = a_p$i b_j%p
    //  -----------------       sz(i) = w1, sz(j) = w3,
    //                          sz(p) = w2
    //                          sz(#) = k1', sz($) = k2,
    //                          sz(%) = k4
    //                          [ij_pi_jp]

    iterator_t ij = in.end();
    size_t sjb_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepb(0) == 1) {
            if(i->stepa(1) % z.m_np) continue;
            if(z.m_sic % i->weight()) continue;
            if(sjb_min == 0 || sjb_min > i->stepa(1)) {
                ij = i; sjb_min = i->stepa(1);
            }
        }
    }
    if(ij == in.end()) return 0;

    kern_mul_ij_pi_jp zz;
    zz.m_d = z.m_d;
    zz.m_ni = z.m_ni;
    zz.m_nj = ij->weight();
    zz.m_np = z.m_np;
    zz.m_spa = z.m_spa;
    zz.m_sjb = ij->stepa(1);
    zz.m_sic = z.m_sic;
    in.splice(out.begin(), out, ij);

    kernel_base<2, 1> *kern = 0;

    return new kern_mul_ij_pi_jp(zz);
}


} // namespace libtensor
