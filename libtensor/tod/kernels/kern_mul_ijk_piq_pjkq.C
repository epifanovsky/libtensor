#include "../../linalg/linalg.h"
#include "kern_mul_ijk_piq_pjkq.h"
#include "kern_mul_ijkl_pkiq_pjlq.h"

namespace libtensor {


const char *kern_mul_ijk_piq_pjkq::k_clazz = "kern_mul_ijk_piq_pjkq";


void kern_mul_ijk_piq_pjkq::run(const loop_registers<2, 1> &r) {

    const double *pa = r.m_ptra[0], *pb = r.m_ptra[1];
    double *pc = r.m_ptrb[0];

    for(size_t j = 0; j < m_nj; j++) {
        const double *pa1 = pa, *pb1 = pb;
        for(size_t p = 0; p < m_np; p++) {
            linalg::mul2_ij_ip_jp_x(m_ni, m_nk, m_nq, pa1, m_sia,
                pb1, m_skb, pc, m_sic, m_d);
            pa1 += m_spa;
            pb1 += m_spb;
        }
        pb += m_sjb;
        pc += m_sjc;
    }
}


kernel_base<2, 1> *kern_mul_ijk_piq_pjkq::match(const kern_mul_ij_piq_pjq &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Rename j -> k.

    //  Minimize sjc > 0:
    //  -----------------
    //  w   a    b    c
    //  nq  1    1    0
    //  nk  0    skb  1
    //  ni  sia  0    sic
    //  np  spa  spb  0
    //  nj  0    sjb  sjc  -->  c_i#j#k = a_p#i#q b_p#j#k#q
    //  -----------------       [ijk_piq_pjkq]
    //

    iterator_t ij = in.end();
    size_t sjc_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepb(0) > 0) {
            if(i->stepa(1) % (z.m_sjb * z.m_nj) ||
                z.m_spb % (i->weight() * i->stepa(1)))
                continue;
            if(i->stepb(0) % z.m_nj ||
                z.m_sic % (i->weight() * i->stepb(0)))
                continue;
            if(sjc_min == 0 || sjc_min > i->stepb(0)) {
                ij = i; sjc_min = i->stepb(0);
            }
        }
    }
    if(ij == in.end()) return 0;

    kern_mul_ijk_piq_pjkq zz;
    zz.m_d = z.m_d;
    zz.m_ni = z.m_ni;
    zz.m_nj = ij->weight();
    zz.m_nk = z.m_nj;
    zz.m_np = z.m_np;
    zz.m_nq = z.m_nq;
    zz.m_spa = z.m_spa;
    zz.m_sia = z.m_sia;
    zz.m_spb = z.m_spb;
    zz.m_sjb = ij->stepa(1);
    zz.m_skb = z.m_sjb;
    zz.m_sic = z.m_sic;
    zz.m_sjc = ij->stepb(0);
    in.splice(out.begin(), out, ij);

    kernel_base<2, 1> *kern = 0;

    if(kern = kern_mul_ijkl_pkiq_pjlq::match(zz, in, out)) return kern;

    return new kern_mul_ijk_piq_pjkq(zz);
}


} // namespace libtensor
