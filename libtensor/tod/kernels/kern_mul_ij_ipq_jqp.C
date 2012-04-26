#include "../../linalg/linalg.h"
#include "kern_mul_ij_ipq_jqp.h"

namespace libtensor {


const char *kern_mul_ij_ipq_jqp::k_clazz = "kern_mul_ij_ipq_jqp";


void kern_mul_ij_ipq_jqp::run(const loop_registers<2, 1> &r) {

    linalg::ij_ipq_jqp_x(m_ni, m_nj, m_np, m_nq, r.m_ptra[0], m_spa, m_sia,
        r.m_ptra[1], m_sqb, m_sjb, r.m_ptrb[0], m_sic, m_d);
}


kernel_base<2, 1> *kern_mul_ij_ipq_jqp::match(const kern_mul_i_ipq_qp &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Minimize sjb > 0:
    //  -----------------
    //  w   a    b    c
    //  np  spa  1    0
    //  nq  1    spb  0
    //  ni  sia  0    sic
    //  nj  0    sjb  1    -->  c_i#j = a_i#p#q b_j#q#p
    //  -----------------       [ij_ipq_jqp]
    //

    iterator_t ij = in.end();
    size_t sjb_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepb(0) == 1) {
            if(i->stepa(1) % (z.m_sqb * z.m_nq)) continue;
            if(z.m_sic % i->weight()) continue;
            if(sjb_min == 0 || sjb_min > i->stepa(1)) {
                ij = i; sjb_min = i->stepa(1);
            }
        }
    }
    if(ij == in.end()) return 0;

    kern_mul_ij_ipq_jqp zz;
    zz.m_d = z.m_d;
    zz.m_ni = z.m_ni;
    zz.m_nj = ij->weight();
    zz.m_np = z.m_np;
    zz.m_nq = z.m_nq;
    zz.m_sia = z.m_sia;
    zz.m_spa = z.m_spa;
    zz.m_sjb = ij->stepa(1);
    zz.m_sqb = z.m_sqb;
    zz.m_sic = z.m_sic;
    in.splice(out.begin(), out, ij);

    kernel_base<2, 1> *kern = 0;

    return new kern_mul_ij_ipq_jqp(zz);
}


} // namespace libtensor
