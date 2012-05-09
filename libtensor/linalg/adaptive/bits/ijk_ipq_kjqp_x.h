#ifndef LIBTENSOR_LINALG_ADAPTIVE_IJK_IPQ_KJQP_X_H
#define LIBTENSOR_LINALG_ADAPTIVE_IJK_IPQ_KJQP_X_H

#include "trp_ijk_jik.h"

namespace libtensor {


template<typename M, typename L1, typename L2, typename L3>
void linalg_base_level5_adaptive<M, L1, L2, L3>::ijk_ipq_kjqp_x(
    size_t ni, size_t nj, size_t nk, size_t np, size_t nq,
    const double *a, const double *b, double *c, double d) {

    size_t npq = np * nq;
    size_t njk = nj * nk;

    double *a1 = M::allocate(ni * npq);
    double *b1 = M::allocate(njk * npq);

    //  a1_iqp <- a_ipq
    for(size_t i = 0; i < ni; i++) {
        L2::ij_ji(nq, np, a + i * npq, nq, a1 + i * npq, np);
    }

    //  b1_jkqp <- b_kjqp
    trp_ijk_jik::transpose(nj, nk, npq, b, nj * npq, b1, nk * npq);

    //  c_ijk += d * a1_iqp b1_jkqp
    L3::ij_ip_jp_x(ni, njk, npq, a1, npq, b1, npq, c, njk, d);

    M::deallocate(b1);
    M::deallocate(a1);
}


} // namespace libtensor

#endif // LIBTENSOR_LINALG_ADAPTIVE_IJK_IPQ_KJQP_X_H
