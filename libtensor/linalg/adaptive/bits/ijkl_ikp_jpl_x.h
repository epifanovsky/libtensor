#ifndef LIBTENSOR_LINALG_ADAPTIVE_IJKL_IKP_JPL_X_H
#define LIBTENSOR_LINALG_ADAPTIVE_IJKL_IKP_JPL_X_H

#include "trp_ijk_jik.h"
#include "trp_ijkl_ikjl.h"
#include "trp_ijkl_jkil.h"
#include "trp_ijkl_kijl.h"

namespace libtensor {


template<typename M, typename L1, typename L2, typename L3>
void linalg_base_level6_adaptive<M, L1, L2, L3>::ijkl_ikp_jpl_x(
    size_t ni, size_t nj, size_t nk, size_t nl, size_t np,
    const double *a, size_t ska, size_t sia,
    const double *b, size_t spb, size_t sjb,
    double *c, double d) {

    size_t nik = ni * nk;
    size_t njl = nj * nl;
    size_t njl1 = (njl % 4 == 0) ? njl : njl + 4 - njl % 4;

    if(ska != np) {

        double *a1 = M::allocate(ni * nk * np);
        double *b1 = M::allocate(np * njl1);
        double *c1 = M::allocate(nik * njl1);

        //  a1_kip <- a_ikp
        trp_ijk_jik::transpose(nk, ni, np, a, ska, sia, a1, ni * np);

        //  b1_pjl <- b_jpl
        trp_ijk_jik::transpose(np, nj, nl, b, spb, sjb, b1, njl1);

        //  c1_kijl <- c_ijkl
        trp_ijkl_jkil::transpose(nk, ni, nj, nl, c, nk * nl, c1, njl1);

        //  c1_kijl += d * a_kip b1_pjl
        L3::ij_ip_pj_x(nik, njl, np, a1, np, b1, njl1, c1, njl1, d);

        //  c_ijkl <- c1_kijl
        trp_ijkl_kijl::transpose(ni, nj, nk, nl, c1, njl1, c, nk * nl);

        M::deallocate(c1);
        M::deallocate(b1);
        M::deallocate(a1);

    } else {

        double *b1 = M::allocate(np * njl1);
        double *c1 = M::allocate(nik * njl1);

        //  b1_pjl <- b_jpl
        trp_ijk_jik::transpose(np, nj, nl, b, spb, sjb, b1, njl1);

        //  c1_ikjl <- c_ijkl
        trp_ijkl_ikjl::transpose(ni, nk, nj, nl, c, nk * nl, c1, njl1);

        //  c1_ikjl += d * a_ikp b1_pjl
        L3::ij_ip_pj_x(nik, njl, np, a, np, b1, njl1, c1, njl1, d);

        //  c_ijkl <- c1_ikjl
        trp_ijkl_ikjl::transpose(ni, nj, nk, nl, c1, njl1, c, nk * nl);

        M::deallocate(c1);
        M::deallocate(b1);
    }
}


} // namespace libtensor

#endif // LIBTENSOR_LINALG_ADAPTIVE_IJKL_IKP_JPL_X_H
