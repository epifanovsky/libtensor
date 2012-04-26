#include "../../linalg/linalg.h"
#include "kern_mul_ijk_kjpq_iqp.h"

namespace libtensor {


const char *kern_mul_ijk_kjpq_iqp::k_clazz = "kern_mul_ijk_kjpq_iqp";


void kern_mul_ijk_kjpq_iqp::run(const loop_registers<2, 1> &r) {

    if(m_spa == m_nq && m_sja == m_np * m_spa && m_ska == m_nj * m_sja &&
        m_sqb == m_np && m_sib == m_nq * m_sqb && m_sjc == m_nk &&
        m_sic == m_nj * m_sjc) {

        linalg::ijk_ipq_kjqp_x(m_ni, m_nj, m_nk, m_nq, m_np,
            r.m_ptra[1], r.m_ptra[0], r.m_ptrb[0], m_d);
        return;
    }

    for(size_t j = 0; j < m_nj; j++) {
        linalg::ij_ipq_jqp_x(m_ni, m_nk, m_nq, m_np,
            r.m_ptra[1], m_sqb, m_sib,
            r.m_ptra[0] + j * m_sja, m_spa, m_ska,
            r.m_ptrb[0] + j * m_sjc, m_sic, m_d);
    }
}


kernel_base<2, 1> *kern_mul_ijk_kjpq_iqp::match(const kern_mul_ij_jipq_qp &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Rename i -> j, j -> k

    //  Minimize sib > 0:
    //  -----------------
    //  w   a    b    c
    //  np  spa  1    0
    //  nq  1    spb  0
    //  nj  sja  0    sjc
    //  nk  ska  0    1
    //  ni  0    sib  sic  -->  c_i#j#k = a_k#j#p#q b_i#q#p
    //  -----------------       [ijk_kjpq_iqp]
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

    kern_mul_ijk_kjpq_iqp zz;
    zz.m_d = z.m_d;
    zz.m_ni = ii->weight();
    zz.m_nj = z.m_ni;
    zz.m_nk = z.m_nj;
    zz.m_np = z.m_np;
    zz.m_nq = z.m_nq;
    zz.m_ska = z.m_sja;
    zz.m_sja = z.m_sia;
    zz.m_spa = z.m_spa;
    zz.m_sib = ii->stepa(1);
    zz.m_sqb = z.m_sqb;
    zz.m_sic = ii->stepb(0);
    zz.m_sjc = z.m_sic;
    in.splice(out.begin(), out, ii);

    kernel_base<2, 1> *kern = 0;

    return new kern_mul_ijk_kjpq_iqp(zz);
}


} // namespace libtensor
