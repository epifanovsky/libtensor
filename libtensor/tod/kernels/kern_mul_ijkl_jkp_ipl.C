#include "../../linalg/linalg.h"
#include "kern_mul_ijkl_jkp_ipl.h"
#include "kern_mul_ijklm_jlp_ipkm.h"

namespace libtensor {


const char *kern_mul_ijkl_jkp_ipl::k_clazz = "kern_mul_ijkl_jkp_ipl";


void kern_mul_ijkl_jkp_ipl::run(const loop_registers<2, 1> &r) {

    for(size_t i = 0; i < m_ni; i++)
    for(size_t j = 0; j < m_nj; j++) {
        linalg::mul2_ij_ip_pj_x(m_nk, m_nl, m_np,
            r.m_ptra[0] + j * m_sja, m_ska,
            r.m_ptra[1] + i * m_sib, m_spb,
            r.m_ptrb[0] + i * m_sic + j * m_sjc, m_skc, m_d);
    }
}


kernel_base<2, 1> *kern_mul_ijkl_jkp_ipl::match(const kern_mul_ijk_jp_ipk &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Rename j -> k, k -> l.

    //  Minimize sja > 0.
    //  -----------------
    //  w   a    b    c
    //  nl  0    1    1
    //  np  1    spb  0
    //  nk  ska  0    skc
    //  ni  0    sib  sic
    //  nj  sja  0    sjc  --> c_i#j#k#l = a_j#k#p b_i#p#l
    //  -----------------      [ijkl_jkp_ipl]
    //

    iterator_t ij = in.end();
    size_t sja_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) > 0 && i->stepa(1) == 0 && i->stepb(0) > 0) {
            if(i->stepa(0) % (z.m_sja * z.m_nj)) continue;
            if(i->stepb(0) % (z.m_sjc * z.m_nj)) continue;
            if(z.m_sic % (i->weight() * i->stepb(0))) continue;
            if(sja_min == 0 || sja_min > i->stepa(0)) {
                ij = i; sja_min = i->stepa(0);
            }
        }
    }
    if(ij == in.end()) return 0;

    kern_mul_ijkl_jkp_ipl zz;
    zz.m_d = z.m_d;
    zz.m_ni = z.m_ni;
    zz.m_nj = ij->weight();
    zz.m_nk = z.m_nj;
    zz.m_nl = z.m_nk;
    zz.m_np = z.m_np;
    zz.m_sja = ij->stepa(0);
    zz.m_ska = z.m_sja;
    zz.m_sib = z.m_sib;
    zz.m_spb = z.m_spb;
    zz.m_sic = z.m_sic;
    zz.m_sjc = ij->stepb(0);
    zz.m_skc = z.m_sjc;
    in.splice(out.begin(), out, ij);

    kernel_base<2, 1> *kern = 0;

    if(kern = kern_mul_ijklm_jlp_ipkm::match(zz, in, out)) return kern;

    return new kern_mul_ijkl_jkp_ipl(zz);
}


} // namespace libtensor
