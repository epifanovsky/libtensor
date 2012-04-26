#include "../../linalg/linalg.h"
#include "kern_mul_ijkl_pkqj_iqpl.h"

namespace libtensor {


const char *kern_mul_ijkl_pkqj_iqpl::k_clazz = "kern_mul_ijkl_pkqj_iqpl";


void kern_mul_ijkl_pkqj_iqpl::run(const loop_registers<2, 1> &r) {

    if(m_sqa == m_nj && m_spb == m_nl && m_skc == m_nl &&
        m_ska == m_sqa * m_nq && m_sqb == m_spb * m_np &&
        m_sjc == m_skc * m_nk && m_spa == m_ska * m_nk &&
        m_sib == m_sqb * m_nq && m_sic == m_sjc * m_nj) {

        linalg::ijkl_ipql_qkpj_x(m_ni, m_nj, m_nk, m_nl, m_nq, m_np,
            r.m_ptra[1], r.m_ptra[0], r.m_ptrb[0], m_d);
        return;
    }

    const double *pa = r.m_ptra[0], *pb = r.m_ptra[1];
    double *pc = r.m_ptrb[0];

    for(size_t i = 0; i < m_ni; i++) {
        const double *pa1 = pa, *pb1 = pb;
        for(size_t q = 0; q < m_nq; q++) {
            const double *pa2 = pa1;
            double *pc2 = pc;
            for(size_t k = 0; k < m_nk; k++) {
                linalg::ij_pi_pj_x(m_nj, m_nl, m_np, pa2, m_spa,
                    pb1, m_spb, pc2, m_sjc, m_d);
                pa2 += m_ska;
                pc2 += m_skc;
            }
            pa1 += m_sqa;
            pb1 += m_sqb;
        }
        pb += m_sib;
        pc += m_sic;
    }
}


kernel_base<2, 1> *kern_mul_ijkl_pkqj_iqpl::match(
    const kern_mul_ijk_pjqi_qpk &z, list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Rename i -> j, j -> k, k -> l.

    //  Minimize sib > 0:
    //  ------------------
    //  w   a    b     c
    //  nj  1    0     sjc
    //  np  spa  spb   0
    //  nl  0    1     1
    //  nk  ska  0     skc
    //  nq  sqa  sqb   0
    //  ni  0    sib   sic  -->  c_i#j#k#l = a_p#k#q#j b_i#q#p#l
    //  ------------------       [ijkl_pkqj_iqpl]
    //

    iterator_t ii = in.end();
    size_t sib_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepb(0) > 0) {
            if(i->stepa(1) % (z.m_sqb * z.m_nq)) continue;
            if(i->stepb(0) % (z.m_sic * z.m_ni)) continue;
            if(sib_min == 0 || sib_min > i->stepa(1)) {
                ii = i; sib_min = i->stepa(1);
            }
        }
    }
    if(ii == in.end()) return 0;

    kern_mul_ijkl_pkqj_iqpl zz;
    zz.m_d = z.m_d;
    zz.m_ni = ii->weight();
    zz.m_nj = z.m_ni;
    zz.m_nk = z.m_nj;
    zz.m_nl = z.m_nk;
    zz.m_np = z.m_np;
    zz.m_nq = z.m_nq;
    zz.m_spa = z.m_spa;
    zz.m_ska = z.m_sja;
    zz.m_sqa = z.m_sqa;
    zz.m_sib = ii->stepa(1);
    zz.m_sqb = z.m_sqb;
    zz.m_spb = z.m_spb;
    zz.m_sic = ii->stepb(0);
    zz.m_sjc = z.m_sic;
    zz.m_skc = z.m_sjc;
    in.splice(out.begin(), out, ii);

    kernel_base<2, 1> *kern = 0;

    return new kern_mul_ijkl_pkqj_iqpl(zz);
}


} // namespace libtensor
