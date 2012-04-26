#include "../../linalg/linalg.h"
#include "kern_mul_ijklm_ipkm_jpl.h"

namespace libtensor {


const char *kern_mul_ijklm_ipkm_jpl::k_clazz = "kern_mul_ijklm_ipkm_jpl";


void kern_mul_ijklm_ipkm_jpl::run(const loop_registers<2, 1> &r) {

//  if(m_skc == m_nl && m_sjc == m_skc * m_nk && m_sic == m_sjc * m_nj) {
//
//      linalg::ijkl_ipl_jpk_x(m_ni, m_nj, m_nk, m_nl, m_np, r.m_ptra[0],
//          m_spa, m_sia, r.m_ptra[1], m_spb, m_sjb, r.m_ptrb[0],
//          m_d);
//      return;
//  }

    for(size_t i = 0; i < m_ni; i++)
    for(size_t j = 0; j < m_nj; j++)
    for(size_t k = 0; k < m_nk; k++) {
        linalg::ij_pi_pj_x(m_nl, m_nm, m_np,
            r.m_ptra[1] + j * m_sjb, m_spb,
            r.m_ptra[0] + i * m_sia + k * m_ska, m_spa,
            r.m_ptrb[0] + i * m_sic + j * m_sjc + k * m_skc, m_slc,
            m_d);
    }
}


kernel_base<2, 1> *kern_mul_ijklm_ipkm_jpl::match(
    const kern_mul_ijkl_ipl_jpk &z, list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Rename k -> l, l -> m

    //  Minimize ska > 0:
    //  ------------------
    //  w   a    b    c
    //  nm  1    0    1
    //  np  spa  spb  0
    //  nl  0    1    slc
    //  ni  sia  0    sic
    //  nj  0    sjb  sjc
    //  nk  ska  0    skc  --> c_i#j#k#l#m = a_i#p#k#m b_j#p#l
    //  -----------------      [ijklm_ipkm_jpl]
    //

    iterator_t ik = in.end();
    size_t ska_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) > 0 && i->stepa(1) == 0 && i->stepb(0) > 0) {
            if(i->stepa(0) % (z.m_nl)) continue;
            if(z.m_spa % (i->weight() * i->stepa(0))) continue;
            if(i->stepb(0) % (z.m_skc * z.m_nk)) continue;
            if(z.m_sjc % (i->weight() * i->stepb(0))) continue;
            if(ska_min == 0 || ska_min > i->stepa(0)) {
                ik = i; ska_min = i->stepa(0);
            }
        }
    }
    if(ik == in.end()) return 0;

    kern_mul_ijklm_ipkm_jpl zz;
    zz.m_d = z.m_d;
    zz.m_ni = z.m_ni;
    zz.m_nj = z.m_nj;
    zz.m_nk = ik->weight();
    zz.m_nl = z.m_nk;
    zz.m_nm = z.m_nl;
    zz.m_np = z.m_np;
    zz.m_sia = z.m_sia;
    zz.m_spa = z.m_spa;
    zz.m_ska = ik->stepa(0);
    zz.m_sjb = z.m_sjb;
    zz.m_spb = z.m_spb;
    zz.m_sic = z.m_sic;
    zz.m_sjc = z.m_sjc;
    zz.m_skc = ik->stepb(0);
    zz.m_slc = z.m_skc;
    in.splice(out.begin(), out, ik);

    kernel_base<2, 1> *kern = 0;

    return new kern_mul_ijklm_ipkm_jpl(zz);
}


} // namespace libtensor
