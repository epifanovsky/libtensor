#include "../../linalg/linalg.h"
#include "kern_mul_ijkl_pliq_jpkq.h"

namespace libtensor {


const char *kern_mul_ijkl_pliq_jpkq::k_clazz = "kern_mul_ijkl_pliq_jpkq";


void kern_mul_ijkl_pliq_jpkq::run(const loop_registers<2, 1> &r) {

    if(m_sia == m_nq && m_skb == m_nq && m_skc == m_nl &&
        m_sla == m_sia * m_ni && m_spb == m_skb * m_nk &&
        m_sjc == m_skc * m_nk && m_spa == m_sla * m_nl &&
        m_sjb == m_spb * m_np && m_sic == m_sjc * m_nj) {

        linalg::ijkl_pliq_jpkq_x(m_ni, m_nj, m_nk, m_nl, m_np, m_nq,
            r.m_ptra[0], r.m_ptra[1], r.m_ptrb[0], m_d);
        return;
    }

    const double *pa = r.m_ptra[0], *pb = r.m_ptra[1];
    double *pc = r.m_ptrb[0];
    for(size_t i = 0; i < m_ni; i++) {
        const double *pa1 = pa, *pb1 = pb;
        double *pc1 = pc;
        for(size_t j = 0; j < m_nj; j++) {
            const double *pa2 = pa1, *pb2 = pb1;
            for(size_t p = 0; p < m_np; p++) {
                linalg::mul2_ij_ip_jp_x(m_nk, m_nl, m_nq, pb2, m_skb,
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


kernel_base<2, 1> *kern_mul_ijkl_pliq_jpkq::match(
    const kern_mul_ijk_pkiq_pjq &z, list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Rename j -> k, k -> l.

    //  Minimize sjc > 0:
    //  -----------------
    //  w   a    b    c
    //  nq  1    1    0
    //  nl  sla  0    1
    //  nk  0    skb  skc
    //  np  spa  spb  0
    //  ni  sia  0    sic
    //  nj  0    sjb  sjc  -->  c_i#j#k#l = a_p#l#i#q b_j#p#k#q
    //  -----------------       [ijkl_pliq_jpkq]

    iterator_t ij = in.end();
    size_t sjc_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepb(0) > 0) {
            if(i->stepa(1) % (z.m_spb * z.m_np)) continue;
            if(i->stepb(0) % (z.m_sjc * z.m_nj) ||
                z.m_sic % (i->weight() * i->stepb(0)))
                continue;
            if(sjc_min == 0 || sjc_min > i->stepb(0)) {
                ij = i; sjc_min = i->stepb(0);
            }
        }
    }
    if(ij == in.end()) return 0;

    kern_mul_ijkl_pliq_jpkq zz;
    zz.m_d = z.m_d;
    zz.m_ni = z.m_ni;
    zz.m_nj = ij->weight();
    zz.m_nk = z.m_nj;
    zz.m_nl = z.m_nk;
    zz.m_np = z.m_np;
    zz.m_nq = z.m_nq;
    zz.m_spa = z.m_spa;
    zz.m_sla = z.m_ska;
    zz.m_sia = z.m_sia;
    zz.m_sjb = ij->stepa(1);
    zz.m_spb = z.m_spb;
    zz.m_skb = z.m_sjb;
    zz.m_sic = z.m_sic;
    zz.m_sjc = ij->stepb(0);
    zz.m_skc = z.m_sjc;
    in.splice(out.begin(), out, ij);

    kernel_base<2, 1> *kern = 0;

    return new kern_mul_ijkl_pliq_jpkq(zz);
}


} // namespace libtensor
