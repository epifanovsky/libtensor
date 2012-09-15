#include "../../linalg/linalg.h"
#include "kern_mul_ijk_pjqi_qpk.h"
#include "kern_mul_ijkl_pkqj_iqpl.h"
#include "kern_mul_ijkl_pkqj_qipl.h"

namespace libtensor {


const char *kern_mul_ijk_pjqi_qpk::k_clazz = "kern_mul_ijk_pjqi_qpk";


void kern_mul_ijk_pjqi_qpk::run(const loop_registers<2, 1> &r) {

    const double *pa = r.m_ptra[0], *pb = r.m_ptra[1];
    double *pc = r.m_ptrb[0];

    for(size_t q = 0; q < m_nq; q++) {
        const double *pa1 = pa;
        double *pc1 = pc;
        for(size_t j = 0; j < m_nj; j++) {
            linalg::mul2_ij_pi_pj_x(m_ni, m_nk, m_np, pa1, m_spa, pb,
                m_spb, pc1, m_sic, m_d);
            pa1 += m_sja;
            pc1 += m_sjc;
        }
        pa += m_sqa;
        pb += m_sqb;
    }
}


kernel_base<2, 1> *kern_mul_ijk_pjqi_qpk::match(const kern_mul_ijk_pji_pk &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Minimize sqa > 0:
    //  ------------------
    //  w   a    b     c
    //  ni  1    0     sic
    //  np  spa  spb   0
    //  nk  0    1     1
    //  nj  sja  0     sjc
    //  nq  sqa  sqb   0    -->  c_i#j#k = a_p#j#q#i b_q#p#k
    //  ------------------       [ijk_pjqi_qpk]
    //

    iterator_t iq = in.end();
    size_t sqa_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) > 0 && i->stepa(1) > 0 && i->stepb(0) == 0) {
            if(i->stepa(0) % z.m_ni ||
                z.m_sja % (i->weight() * i->stepa(0)))
                continue;
            if(i->stepa(1) % (z.m_spb * z.m_np)) continue;
            if(sqa_min == 0 || sqa_min > i->stepa(0)) {
                iq = i; sqa_min = i->stepa(0);
            }
        }
    }
    if(iq == in.end()) return 0;

    kern_mul_ijk_pjqi_qpk zz;
    zz.m_d = z.m_d;
    zz.m_ni = z.m_ni;
    zz.m_nj = z.m_nj;
    zz.m_nk = z.m_nk;
    zz.m_np = z.m_np;
    zz.m_nq = iq->weight();
    zz.m_spa = z.m_spa;
    zz.m_sja = z.m_sja;
    zz.m_sqa = iq->stepa(0);
    zz.m_sqb = iq->stepa(1);
    zz.m_spb = z.m_spb;
    zz.m_sic = z.m_sic;
    zz.m_sjc = z.m_sjc;
    in.splice(out.begin(), out, iq);

    kernel_base<2, 1> *kern = 0;

    if(kern = kern_mul_ijkl_pkqj_iqpl::match(zz, in, out)) return kern;
    if(kern = kern_mul_ijkl_pkqj_qipl::match(zz, in, out)) return kern;

    return new kern_mul_ijk_pjqi_qpk(zz);
}


} // namespace libtensor
