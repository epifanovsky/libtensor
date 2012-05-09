#ifndef LIBTENSOR_LINALG_ADAPTIVE_IJKL_PLIQ_JPKQ_X_H
#define LIBTENSOR_LINALG_ADAPTIVE_IJKL_PLIQ_JPKQ_X_H

#include "trp_ijkl_ikjl.h"
#include "trp_ijkl_jkil.h"
#include "trp_ijkl_kijl.h"
#include "trp_ijkl_kjil.h"

namespace libtensor {


template<typename M, typename L1, typename L2, typename L3>
void linalg_base_level6_adaptive<M, L1, L2, L3>::ijkl_pliq_jpkq_x(
    size_t ni, size_t nj, size_t nk, size_t nl, size_t np, size_t nq,
    const double *a, const double *b, double *c, double d) {

    size_t npq = np * nq;
    size_t nil = ni * nl;
    size_t njk = nj * nk;
    size_t njkl = njk * nl;
    size_t npq1 = (npq % 4 == 0) ? npq : npq + 4 - npq % 4;
    size_t nil1 = (nil % 4 == 0) ? nil : nil + 4 - nil % 4;
    size_t njk1 = (njk % 4 == 0) ? njk : njk + 4 - njk % 4;

    double *a1 = M::allocate(nil * npq1);
    double *b1 = M::allocate(njk * npq1);
    double *c1 = M::allocate(njk * nil1);

    //  a1_ilpq <- a_pliq
    trp_ijkl_kjil::transpose(ni, nl, np, nq, a, ni * nq, a1, npq1);

    //  b1_jkpq <- b_jpkq
    trp_ijkl_ikjl::transpose(nj, nk, np, nq, b, nk * nq, b1, npq1);

    //  c1_jkil <- c_ijkl
    trp_ijkl_kijl::transpose(nj, nk, ni, nl, c, nk * nl, c1, nil1);

    //  c1_jkil += d * b1_jkpq a1_ilpq
    L3::ij_ip_jp_x(njk, nil, npq, b1, npq1, a1, npq1, c1, nil1, d);

    //  c_ijkl <- c1_jkil
    trp_ijkl_jkil::transpose(ni, nj, nk, nl, c1, nil1, c, nk * nl);

    M::deallocate(c1);
    M::deallocate(b1);
    M::deallocate(a1);
}


} // namespace libtensor

#endif // LIBTENSOR_LINALG_ADAPTIVE_IJKL_PLIQ_JPKQ_X_H
