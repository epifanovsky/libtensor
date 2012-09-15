#include "../../linalg/linalg.h"
#include "kern_mul_ijkl_pliq_pjqk.h"


namespace libtensor {


const char *kern_mul_ijkl_pliq_pjqk::k_clazz = "kern_mul_ijkl_pliq_pjqk";


void kern_mul_ijkl_pliq_pjqk::run(const loop_registers<2, 1> &r) {

    if(m_sia == m_nq && m_sla == m_sia * m_ni && m_spa == m_sla * m_nl &&
        m_sqb == m_nk && m_sjb == m_sqb * m_nq &&
        m_spb == m_sjb * m_nj && m_skc == m_nl &&
        m_sjc == m_skc * m_nk && m_sic == m_sjc * m_nj) {

        linalg::ijkl_pliq_pjqk_x(m_ni, m_nj, m_nk, m_nl, m_np, m_nq,
            r.m_ptra[0], r.m_ptra[1], r.m_ptrb[0], m_d);
        return;
    }

    const double *pa = r.m_ptra[0];
    double *pc = r.m_ptrb[0];

    for(size_t i = 0; i < m_ni; i++) {

        const double *pb1 = r.m_ptra[1];
        double *pc1 = pc;

        for(size_t j = 0; j < m_nj; j++) {

            const double *pa2 = pa, *pb2 = pb1;

            for(size_t p = 0; p < m_np; p++) {
                linalg::mul2_ij_pi_jp_x(m_nk, m_nl, m_nq, pb2, m_sqb,
                    pa2, m_sla, pc1, m_skc, m_d);
                pa2 += m_spa;
                pb2 += m_spb;
            }

            pb1 += m_sjb;
            pc1 += m_sjc;

        }

        pa += m_sia;
        pc += m_sic;
    }
}


kernel_base<2, 1> *kern_mul_ijkl_pliq_pjqk::match(
    const kern_mul_ijk_pkq_piqj &z, list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Rename i -> j, j -> k, k -> l.

    //  Minimize sia > 0:
    //  -----------------
    //  w   a    b    c
    //  nk  0    1    skc
    //  nq  1    sqb  0
    //  nl  sla  0    1
    //  nj  0    sjb  sjc
    //  np  spa  spb  0
    //  ni  sia  0    sic  -->  c_i#j#k#l = a_p#l#i#q b_p#j#q#k
    //  -----------------       [ijkl_pliq_pjqk]
    //

    iterator_t ii = in.end();
    size_t sia_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) > 0 && i->stepa(1) == 0 && i->stepb(0) > 0) {
            if(i->stepa(0) % z.m_nq ||
                z.m_ska % (i->weight() * i->stepa(0)))
                continue;
            if(i->stepb(0) % (z.m_sic * z.m_ni)) continue;
            if(sia_min == 0 || sia_min > i->stepa(0)) {
                ii = i; sia_min = i->stepa(0);
            }
        }
    }
    if(ii == in.end()) return 0;

    kern_mul_ijkl_pliq_pjqk zz;
    zz.m_d = z.m_d;
    zz.m_ni = ii->weight();
    zz.m_nj = z.m_ni;
    zz.m_nk = z.m_nj;
    zz.m_nl = z.m_nk;
    zz.m_np = z.m_np;
    zz.m_nq = z.m_nq;
    zz.m_spa = z.m_spa;
    zz.m_sla = z.m_ska;
    zz.m_sia = ii->stepa(0);
    zz.m_spb = z.m_spb;
    zz.m_sjb = z.m_sib;
    zz.m_sqb = z.m_sqb;
    zz.m_sic = ii->stepb(0);
    zz.m_sjc = z.m_sic;
    zz.m_skc = z.m_sjc;
    in.splice(out.begin(), out, ii);

    kernel_base<2, 1> *kern = 0;

    return new kern_mul_ijkl_pliq_pjqk(zz);
}


} // namespace libtensor
