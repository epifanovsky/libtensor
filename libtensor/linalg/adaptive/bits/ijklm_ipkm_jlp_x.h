#ifndef LIBTENSOR_LINALG_ADAPTIVE_IJKLM_IPKM_JLP_X_H
#define LIBTENSOR_LINALG_ADAPTIVE_IJKLM_IPKM_JLP_X_H

#include "trp_ijk_jik.h"

namespace libtensor {


template<typename M, typename L1, typename L2, typename L3>
void linalg_base_level6_adaptive<M, L1, L2, L3>::ijklm_ipkm_jlp_x(
    size_t ni, size_t nj, size_t nk, size_t nl, size_t nm, size_t np,
    const double *a, const double *b, double *c, double d) {

    size_t nikm = ni * nk * nm;
    size_t nikm1 = (nikm % 4 == 0) ? nikm : nikm + 4 - nikm % 4;
    size_t njl = nj * nl;

    double *a1 = M::allocate(np * nikm1);
    double *c1 = M::allocate(njl * nikm1);
    memset(c1, 0, sizeof(double) * njl * nikm1);

    //  a1_pikm <- a_ipkm
    trp_ijk_jik::transpose(np, ni, nk * nm, a, np * nk * nm, a1, nikm1);

    //  c1_jlikm = d b_jlp a1_pikm
    L3::ij_ip_pj_x(njl, nikm, np, b, np, a1, nikm1, c1, nikm1, d);

    for(size_t j = 0; j < nj; j++)
    for(size_t l = 0; l < nl; l++) {
        size_t jl = (j * nl + l) * nikm1;
        for(size_t i = 0; i < ni; i++)
        for(size_t k = 0; k < nk; k++) {
            L1::i_i_x(nm, c1 + jl + (i * nk + k) * nm, 1, 1.0,
                c + (((i * nj + j) * nk + k) * nl + l) * nm, 1);
        }
    }

    M::deallocate(c1);
    M::deallocate(a1);
}


} // namespace libtensor

#endif // LIBTENSOR_LINALG_ADAPTIVE_IJKLM_IPKM_JLP_X_H
