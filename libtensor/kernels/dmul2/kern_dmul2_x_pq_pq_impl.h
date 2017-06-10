#ifndef LIBTENSOR_KERN_DMUL2_X_PQ_PQ_IMPL_H
#define LIBTENSOR_KERN_DMUL2_X_PQ_PQ_IMPL_H

#include "kern_dmul2_x_pq_pq.h"

namespace libtensor {


template<typename LA>
const char *kern_dmul2_x_pq_pq<LA>::k_clazz = "kern_dmul2_x_pq_pq";


template<typename LA>
void kern_dmul2_x_pq_pq<LA>::run(
    device_context_ref ctx,
    const loop_registers<2, 1> &r) {

    r.m_ptrb[0][0] += LA::mul2_x_pq_pq(ctx, m_np, m_nq, r.m_ptra[0], m_spa,
        r.m_ptra[1], m_spb) * m_d;
}


template<typename LA>
kernel_base<LA, 2, 1, double> *kern_dmul2_x_pq_pq<LA>::match(
    const kern_dmul2_x_p_p<LA> &z, list_t &in, list_t &out) {

    if(in.empty()) return 0;
    if(z.m_spa != 1) return 0;

    //    1. Minimize spa > 0:
    //    -------------
    //    w   a   b   c
    //    nq  1   1   0
    //    np  spa spb 0  -->  c_# = a_p#q b_p$q
    //    -------------
    //                        [mul2_x_pq_pq]

    iterator_t ip = in.end();
    size_t spa_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); ++i) {
        if(i->stepa(0) > 0 && i->stepa(1) > 0 && i->stepb(0) == 0) {
            if(i->stepa(0) % z.m_np || i->stepa(1) % z.m_np) continue;
            if(spa_min == 0 || spa_min > i->stepa(0)) {
                ip = i; spa_min = i->stepa(0);
            }
        }
    }
    if(ip == in.end()) return 0;

    kern_dmul2_x_pq_pq zz;
    zz.m_d = z.m_d;
    zz.m_np = ip->weight();
    zz.m_nq = z.m_np;
    zz.m_spa = ip->stepa(0);
    zz.m_spb = ip->stepa(1);
    in.splice(out.begin(), out, ip);

    return new kern_dmul2_x_pq_pq(zz);
}


} // namespace libtensor

#endif // LIBTENSOR_KERN_DMUL2_X_PQ_PQ_IMPL_H
