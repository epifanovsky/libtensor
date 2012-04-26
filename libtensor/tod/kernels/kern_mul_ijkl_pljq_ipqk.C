#include "../../linalg/linalg.h"
#include "kern_mul_ijkl_pljq_ipqk.h"


namespace libtensor {


const char *kern_mul_ijkl_pljq_ipqk::k_clazz = "kern_mul_ijkl_pljq_ipqk";


void kern_mul_ijkl_pljq_ipqk::run(const loop_registers<2, 1> &r) {

    if(m_sja == m_nq && m_sqb == m_nk && m_skc == m_nl &&
        m_sla == m_sja * m_nj && m_spb == m_sqb * m_nq &&
        m_sjc == m_skc * m_nk && m_spa == m_sla * m_nl &&
        m_sib == m_spb * m_np && m_sic == m_sjc * m_nj) {

        linalg::ijkl_ipqk_pljq_x(m_ni, m_nj, m_nk, m_nl, m_np, m_nq,
            r.m_ptra[1], r.m_ptra[0], r.m_ptrb[0], m_d);
        return;
    }

    const double *pb = r.m_ptra[1];
    double *pc = r.m_ptrb[0];

    for(size_t i = 0; i < m_ni; i++) {

        const double *pa1 = r.m_ptra[0];
        double *pc1 = pc;

        for(size_t j = 0; j < m_nj; j++) {

            const double *pa2 = pa1, *pb2 = pb;

            for(size_t p = 0; p < m_np; p++) {
                linalg::ij_pi_jp_x(m_nk, m_nl, m_nq, pb2, m_sqb,
                    pa2, m_sla, pc1, m_skc, m_d);
                pa2 += m_spa;
                pb2 += m_spb;
            }

            pa1 += m_sja;
            pc1 += m_sjc;
        }

        pb += m_sib;
        pc += m_sic;
    }
}


kernel_base<2, 1> *kern_mul_ijkl_pljq_ipqk::match(
    const kern_mul_ijk_pkq_ipqj &z, list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Rename j -> k, k -> l.

    //  Minimize sja > 0:
    //  -----------------
    //  w   a    b    c
    //  nk  0    1    skc
    //  nq  1    sqb  0
    //  nl  sla  0    1
    //  ni  0    sib  sic
    //  np  spa  spb  0
    //  nj  sja  0    sjc  -->  c_i#j#k#l = a_p#l#j#q b_i#p#q#k
    //  -----------------       [ijkl_pljq_ipqk]
    //

    iterator_t ij = in.end();
    size_t sja_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) > 0 && i->stepa(1) == 0 && i->stepb(0) > 0) {
            if(i->stepa(0) % z.m_nq ||
                z.m_ska % (i->weight() * i->stepa(0)))
                continue;
            if(i->stepb(0) % (z.m_sjc * z.m_nj) ||
                z.m_sic % (i->weight() * i->stepb(0)))
                continue;
            if(sja_min == 0 || sja_min > i->stepa(0)) {
                ij = i; sja_min = i->stepa(0);
            }
        }
    }
    if(ij == in.end()) return 0;

    kern_mul_ijkl_pljq_ipqk zz;
    zz.m_d = z.m_d;
    zz.m_ni = z.m_ni;
    zz.m_nj = ij->weight();
    zz.m_nk = z.m_nj;
    zz.m_nl = z.m_nk;
    zz.m_np = z.m_np;
    zz.m_nq = z.m_nq;
    zz.m_spa = z.m_spa;
    zz.m_sla = z.m_ska;
    zz.m_sja = ij->stepa(0);
    zz.m_sib = z.m_sib;
    zz.m_spb = z.m_spb;
    zz.m_sqb = z.m_sqb;
    zz.m_sic = z.m_sic;
    zz.m_sjc = ij->stepb(0);
    zz.m_skc = z.m_sjc;
    in.splice(out.begin(), out, ij);

    kernel_base<2, 1> *kern = 0;

    return new kern_mul_ijkl_pljq_ipqk(zz);
}


} // namespace libtensor
