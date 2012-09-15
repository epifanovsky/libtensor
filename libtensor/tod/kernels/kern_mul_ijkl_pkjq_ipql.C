#include "../../linalg/linalg.h"
#include "kern_mul_ijkl_pkjq_ipql.h"

namespace libtensor {


const char *kern_mul_ijkl_pkjq_ipql::k_clazz = "kern_mul_ijkl_pkjq_ipql";


void kern_mul_ijkl_pkjq_ipql::run(const loop_registers<2, 1> &r) {

    if(m_sja == m_nq && m_sqb == m_nl && m_skc == m_nl &&
        m_ska == m_sja * m_nj && m_spb == m_sqb * m_nq &&
        m_sjc == m_skc * m_nk && m_spa == m_ska * m_nk &&
        m_sib == m_spb * m_np && m_sic == m_sjc * m_nj) {

        linalg::ijkl_ipql_pkjq_x(m_ni, m_nj, m_nk, m_nl, m_np, m_nq,
            r.m_ptra[1], r.m_ptra[0], r.m_ptrb[0], m_d);
        return;
    }

    const double *pa = r.m_ptra[0], *pb = r.m_ptra[1];
    double *pc = r.m_ptrb[0];

    for(size_t p = 0; p < m_np; p++) {
        const double *pa1 = pa, *pb1 = pb;
        double *pc1 = pc;
        for(size_t k = 0; k < m_nk; k++) {
            const double *pa2 = pa1, *pb2 = pb1;
            double *pc2 = pc1;
            for(size_t i = 0; i < m_ni; i++) {
                linalg::mul2_ij_ip_pj_x(m_nj, m_nl, m_nq, pa2, m_sja,
                    pb2, m_sqb, pc2, m_sjc, m_d);
                pb2 += m_sib;
                pc2 += m_sic;
            }
            pa1 += m_ska;
            pc1 += m_skc;
        }
        pa += m_spa;
        pb += m_spb;
    }
}


kernel_base<2, 1> *kern_mul_ijkl_pkjq_ipql::match(
    const kern_mul_ijk_pjq_ipqk &z, list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Rename k -> l.

    //  Minimize ska > 0.
    //  -----------------
    //  w   a    b    c
    //  nl  0    1    1
    //  nq  1    sqb  0
    //  nj  sja  0    sjc
    //  ni  0    sib  sic
    //  np  spa  spb  0
    //  nk  ska  0    skc  --> c_i#j#k#l = a_p#k#j#q b_i#p#q#l
    //  -----------------      [ijkl_pkjq_ipql]
    //

    iterator_t ik = in.end();
    size_t ska_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) > 0 && i->stepa(1) == 0 && i->stepb(0) > 0) {
            if(i->stepa(0) % (z.m_sja * z.m_nj) ||
                z.m_spa % (i->weight() * i->stepa(0)))
                continue;
            if(i->stepb(0) % z.m_nk ||
                z.m_sjc % (i->weight() * i->stepb(0)))
                continue;
            if(ska_min == 0 || ska_min > i->stepa(0)) {
                ik = i; ska_min = i->stepa(0);
            }
        }
    }
    if(ik == in.end()) return 0;

    kern_mul_ijkl_pkjq_ipql zz;
    zz.m_d = z.m_d;
    zz.m_ni = z.m_ni;
    zz.m_nj = z.m_nj;
    zz.m_nk = ik->weight();
    zz.m_nl = z.m_nk;
    zz.m_np = z.m_np;
    zz.m_nq = z.m_nq;
    zz.m_spa = z.m_spa;
    zz.m_ska = ik->stepa(0);
    zz.m_sja = z.m_sja;
    zz.m_sib = z.m_sib;
    zz.m_spb = z.m_spb;
    zz.m_sqb = z.m_sqb;
    zz.m_sic = z.m_sic;
    zz.m_sjc = z.m_sjc;
    zz.m_skc = ik->stepb(0);
    in.splice(out.begin(), out, ik);

    kernel_base<2, 1> *kern = 0;

    return new kern_mul_ijkl_pkjq_ipql(zz);
}


} // namespace libtensor
