#ifndef LIBTENSOR_KERN_MUL2_X_PQ_QP_IMPL_H
#define LIBTENSOR_KERN_MUL2_X_PQ_QP_IMPL_H

#include "kern_mul2_x_pq_qp.h"

namespace libtensor {


template<typename LA, typename T>
const char *kern_mul2_x_pq_qp<LA, T>::k_clazz = "kern_mul2_x_pq_qp";


template<typename LA, typename T>
void kern_mul2_x_pq_qp<LA, T>::run(
    device_context_ref ctx,
    const loop_registers_x<2, 1, T> &r) {

    r.m_ptrb[0][0] += LA::mul2_x_pq_qp(ctx, m_np, m_nq, r.m_ptra[0], m_spa,
        r.m_ptra[1], m_sqb) * m_d;
}


template<typename LA, typename T>
kernel_base<LA, 2, 1, T> *kern_mul2_x_pq_qp<LA, T>::match(
    const kern_mul2_x_p_p<LA, T> &z, list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Minimize spb > 0:
    //  ---------------
    //  w   a    b    c
    //  np  spa  1    0
    //  nq  1    spb  0  -->  c_# = a_p#q b_q#p
    //  ---------------       [mul2_x_pq_qp]
    //

    iterator_t iq = in.end();
    size_t spb_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) == 1 && i->stepa(1) > 0 && i->stepb(0) == 0) {
            if(i->stepa(1) % z.m_np) continue;
            if(spb_min == 0 || spb_min > i->stepa(1)) {
                iq = i; spb_min = i->stepa(1);
            }
        }
    }
    if(iq == in.end()) return 0;

    kern_mul2_x_pq_qp zz;
    zz.m_d = z.m_d;
    zz.m_np = z.m_np;
    zz.m_nq = iq->weight();
    zz.m_spa = z.m_spa;
    zz.m_sqb = iq->stepa(1);
    in.splice(out.begin(), out, iq);

    return new kern_mul2_x_pq_qp(zz);
}


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL2_X_PQ_QP_IMPL_H
