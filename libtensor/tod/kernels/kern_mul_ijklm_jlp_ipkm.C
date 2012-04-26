#include "../../linalg/linalg.h"
#include "kern_mul_ijklm_jlp_ipkm.h"
#include "kern_mul_ijklmn_kjmp_ipln.h"

namespace libtensor {


const char *kern_mul_ijklm_jlp_ipkm::k_clazz = "kern_mul_ijklm_jlp_ipkm";


void kern_mul_ijklm_jlp_ipkm::run(const loop_registers<2, 1> &r) {

    if(m_sic == m_nj * m_sjc && m_sjc == m_nk * m_skc &&
        m_skc == m_nl * m_slc && m_slc == m_nm &&
        m_sja == m_nl * m_sla && m_sla == m_np &&
        m_sib == m_np * m_spb && m_spb == m_nk * m_skb &&
        m_skb == m_nm) {

        linalg::ijklm_ipkm_jlp_x(m_ni, m_nj, m_nk, m_nl, m_nm, m_np,
            r.m_ptra[1], r.m_ptra[0], r.m_ptrb[0], m_d);
        return;
    }

    for(size_t i = 0; i < m_ni; i++)
    for(size_t j = 0; j < m_nj; j++)
    for(size_t k = 0; k < m_nk; k++) {
        linalg::ij_ip_pj_x(m_nl, m_nm, m_np,
            r.m_ptra[0] + j * m_sja, m_sla,
            r.m_ptra[1] + i * m_sib + k * m_skb, m_spb,
            r.m_ptrb[0] + i * m_sic + j * m_sjc + k * m_skc, m_slc,
            m_d);
    }
}


kernel_base<2, 1> *kern_mul_ijklm_jlp_ipkm::match(const kern_mul_ijkl_jkp_ipl &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Rename k -> l, l -> m.

    //  Minimize skb > 0.
    //  -----------------
    //  w   a    b    c
    //  nm  0    1    1
    //  np  1    spb  0
    //  nl  sla  0    slc
    //  ni  0    sib  sic
    //  nj  sja  0    sjc
    //  nk  0    skb  skc  --> c_i#j#k#l#m = a_j#l#p b_i#p#k#m
    //  -----------------      [ijklm_jlp_ipkm]
    //

    iterator_t ik = in.end();
    size_t skb_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepb(0) > 0) {
            if(i->stepa(1) % z.m_nl) continue;
            if(z.m_spb % (i->weight() * i->stepa(1))) continue;
            if(i->stepb(0) % (z.m_skc * z.m_nk)) continue;
            if(z.m_sjc % (i->weight() * i->stepb(0))) continue;
            if(skb_min == 0 || skb_min > i->stepa(1)) {
                ik = i; skb_min = i->stepa(1);
            }
        }
    }
    if(ik == in.end()) return 0;

    kern_mul_ijklm_jlp_ipkm zz;
    zz.m_d = z.m_d;
    zz.m_ni = z.m_ni;
    zz.m_nj = z.m_nj;
    zz.m_nk = ik->weight();
    zz.m_nl = z.m_nk;
    zz.m_nm = z.m_nl;
    zz.m_np = z.m_np;
    zz.m_sja = z.m_sja;
    zz.m_sla = z.m_ska;
    zz.m_sib = z.m_sib;
    zz.m_spb = z.m_spb;
    zz.m_skb = ik->stepa(1);
    zz.m_sic = z.m_sic;
    zz.m_sjc = z.m_sjc;
    zz.m_skc = ik->stepb(0);
    zz.m_slc = z.m_skc;
    in.splice(out.begin(), out, ik);

    kernel_base<2, 1> *kern = 0;

    if(kern = kern_mul_ijklmn_kjmp_ipln::match(zz, in, out)) return kern;

    return new kern_mul_ijklm_jlp_ipkm(zz);
}


} // namespace libtensor
