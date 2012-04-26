#include "../../linalg/linalg.h"
#include "kern_mul_ijkl_jpl_ipk.h"

namespace libtensor {


const char *kern_mul_ijkl_jpl_ipk::k_clazz = "kern_mul_ijkl_jpl_ipk";


void kern_mul_ijkl_jpl_ipk::run(const loop_registers<2, 1> &r) {

//  if(m_skc == m_nl && m_sjc == m_skc * m_nk && m_sic == m_sjc * m_nj) {
//
//      linalg::ijkl_ipl_jpk_x(m_ni, m_nj, m_nk, m_nl, m_np, r.m_ptra[0],
//          m_spa, m_sia, r.m_ptra[1], m_spb, m_sjb, r.m_ptrb[0],
//          m_d);
//      return;
//  }

    for(size_t i = 0; i < m_ni; i++)
    for(size_t j = 0; j < m_nj; j++) {
        linalg::ij_pi_pj_x(m_nk, m_nl, m_np,
            r.m_ptra[1] + i * m_sib, m_spb,
            r.m_ptra[0] + j * m_sja, m_spa,
            r.m_ptrb[0] + i * m_sic + j * m_sjc, m_skc, m_d);
    }
}


kernel_base<2, 1> *kern_mul_ijkl_jpl_ipk::match(const kern_mul_ijk_ipk_pj &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Rename i -> j, j -> k, k -> l

    //  Minimize sib > 0:
    //  ------------------
    //  w   a    b    c
    //  nl  1    0    1
    //  np  spa  spb  0
    //  nk  0    1    skc
    //  nj  sja  0    sjc
    //  ni  0    sib  sic  --> c_i#j#k#l = a_j#p#l b_i#p#k
    //  -----------------      [ijkl_jpl_ipk]
    //

    iterator_t ii = in.end();
    size_t sib_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepb(0) > 0) {
            if(i->stepa(1) % (z.m_spb * z.m_np)) continue;
            if(i->stepb(0) % (z.m_sic * z.m_ni)) continue;
            if(sib_min == 0 || sib_min > i->stepa(1)) {
                ii = i; sib_min = i->stepa(1);
            }
        }
    }
    if(ii == in.end()) return 0;

    kern_mul_ijkl_jpl_ipk zz;
    zz.m_d = z.m_d;
    zz.m_ni = ii->weight();
    zz.m_nj = z.m_ni;
    zz.m_nk = z.m_nj;
    zz.m_nl = z.m_nk;
    zz.m_np = z.m_np;
    zz.m_sja = z.m_sia;
    zz.m_spa = z.m_spa;
    zz.m_sib = ii->stepa(1);
    zz.m_spb = z.m_spb;
    zz.m_sic = ii->stepb(0);
    zz.m_sjc = z.m_sic;
    zz.m_skc = z.m_sjc;
    in.splice(out.begin(), out, ii);

    kernel_base<2, 1> *kern = 0;

    return new kern_mul_ijkl_jpl_ipk(zz);
}


} // namespace libtensor
