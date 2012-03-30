#include "../../linalg/linalg.h"
#include "kern_mul_ij_jipq_qp.h"
#include "kern_mul_ijk_kjpq_iqp.h"

namespace libtensor {


const char *kern_mul_ij_jipq_qp::k_clazz = "kern_mul_ij_jipq_qp";


void kern_mul_ij_jipq_qp::run(const loop_registers<2, 1> &r) {

    for(size_t i = 0; i < m_ni; i++) {
        linalg::i_ipq_qp_x(m_nj, m_np, m_nq, r.m_ptra[0] + i * m_sia,
            m_spa, m_sja, r.m_ptra[1], m_sqb,
            r.m_ptrb[0] + i * m_sic, 1, m_d);
    }
}


kernel_base<2, 1> *kern_mul_ij_jipq_qp::match(const kern_mul_i_ipq_qp &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Minimize sja > 0:
    //  -----------------
    //  w   a    b    c
    //  np  spa  1    0
    //  nq  1    spb  0
    //  ni  sia  0    sic
    //  nj  sja  0    1    -->  c_i#j = a_j#i#p#q b_q#p
    //  -----------------       [ij_jipq_qp]
    //

    iterator_t ij = in.end();
    size_t sja_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) > 0 && i->stepa(1) == 0 && i->stepb(0) == 1) {
            if(i->stepa(0) % (z.m_sia * z.m_ni)) continue;
            if(z.m_sic % i->weight()) continue;
            if(sja_min == 0 || sja_min > i->stepa(0)) {
                ij = i; sja_min = i->stepa(0);
            }
        }
    }
    if(ij == in.end()) return 0;

    kern_mul_ij_jipq_qp zz;
    zz.m_d = z.m_d;
    zz.m_ni = z.m_ni;
    zz.m_nj = ij->weight();
    zz.m_np = z.m_np;
    zz.m_nq = z.m_nq;
    zz.m_sja = ij->stepa(0);
    zz.m_sia = z.m_sia;
    zz.m_spa = z.m_spa;
    zz.m_sqb = z.m_sqb;
    zz.m_sic = z.m_sic;
    in.splice(out.begin(), out, ij);

    kernel_base<2, 1> *kern = 0;

    if(kern = kern_mul_ijk_kjpq_iqp::match(zz, in, out)) return kern;

    return new kern_mul_ij_jipq_qp(zz);
}


} // namespace libtensor
