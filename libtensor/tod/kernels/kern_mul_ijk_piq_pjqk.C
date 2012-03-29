#include "../../linalg/linalg.h"
#include "kern_mul_ijk_piq_pjqk.h"
#include "kern_mul_ijkl_pkiq_pjql.h"

namespace libtensor {


const char *kern_mul_ijk_piq_pjqk::k_clazz = "kern_mul_ijk_piq_pjqk";


void kern_mul_ijk_piq_pjqk::run(const loop_registers<2, 1> &r) {

    const double *pa = r.m_ptra[0], *pb = r.m_ptra[1];
    double *pc = r.m_ptrb[0];

    for(size_t p = 0; p < m_np; p++) {
        const double *pb1 = pb;
        double *pc1 = pc;
        for(size_t j = 0; j < m_nj; j++) {
            linalg::ij_ip_pj_x(m_ni, m_nk, m_nq, pa, m_sia,
                pb1, m_sqb, pc1, m_sic, m_d);
            pb1 += m_sjb;
            pc1 += m_sjc;
        }
        pa += m_spa;
        pb += m_spb;
    }
}


kernel_base<2, 1> *kern_mul_ijk_piq_pjqk::match(const kern_mul_ijk_ip_jpk &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Rename p -> q.

    //  Minimize spa > 0.
    //  -----------------
    //  w   a    b    c
    //  nk  0    1    1
    //  nq  1    sqb  0
    //  ni  sia  0    sic
    //  nj  0    sjb  sjc
    //  np  spa  spb  0    --> c_i#j#k = a_p#i#q b_p#j#q#k
    //  -----------------      [ijk_piq_pjqk]
    //

    iterator_t ip = in.end();
    size_t spa_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) > 0 && i->stepa(1) > 0 && i->stepb(0) == 0) {
            if(i->stepa(0) % (z.m_sia * z.m_ni)) continue;
            if(i->stepa(1) % (z.m_sjb * z.m_nj)) continue;
            if(spa_min == 0 || spa_min > i->stepa(0)) {
                ip = i; spa_min = i->stepa(0);
            }
        }
    }
    if(ip == in.end()) return 0;

    kern_mul_ijk_piq_pjqk zz;
    zz.m_d = z.m_d;
    zz.m_ni = z.m_ni;
    zz.m_nj = z.m_nj;
    zz.m_nk = z.m_nk;
    zz.m_np = ip->weight();
    zz.m_nq = z.m_np;
    zz.m_spa = ip->stepa(0);
    zz.m_sia = z.m_sia;
    zz.m_spb = ip->stepa(1);
    zz.m_sjb = z.m_sjb;
    zz.m_sqb = z.m_spb;
    zz.m_sic = z.m_sic;
    zz.m_sjc = z.m_sjc;
    in.splice(out.begin(), out, ip);

    kernel_base<2, 1> *kern = 0;

    if(kern = kern_mul_ijkl_pkiq_pjql::match(zz, in, out)) return kern;

    return new kern_mul_ijk_piq_pjqk(zz);
}


} // namespace libtensor
