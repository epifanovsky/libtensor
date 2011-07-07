#ifndef LIBTENSOR_LINALG_ADAPTIVE_IJKL_IPJ_PLK_X_H
#define LIBTENSOR_LINALG_ADAPTIVE_IJKL_IPJ_PLK_X_H

#include "trp_ijk_jik.h"

namespace libtensor {


template<typename M, typename L1, typename L2, typename L3>
void linalg_base_level5_adaptive<M, L1, L2, L3>::ijkl_ipj_plk_x(
	size_t ni, size_t nj, size_t nk, size_t nl, size_t np,
	const double *a, const double *b, double *c, double d) {

	size_t nij = ni * nj;
	size_t nij1 = (nij % 4 == 0) ? nij : nij + 4 - nij % 4;
	size_t nkl = nk * nl;
	size_t nkl1 = (nkl % 4 == 0) ? nkl : nkl + 4 - nkl % 4;

	double *a1 = M::allocate(np * nij1);
	double *b1 = M::allocate(np * nkl1);

	//	a1_pij <- a_ipj
	trp_ijk_jik::transpose(np, ni, nj, a, np * nj, a1, nij1);

	//	b1_pkl <- b_plk
	for(size_t p = 0; p < np; p++) {
		L2::ij_ji(nk, nl, b + p * nkl, nk, b1 + p * nkl1, nl);
	}

	//	c_ijkl += d * a1_pij b1_pkl
	L3::ij_pi_pj_x(nij, nkl, np, a1, nij1, b1, nkl1, c, nkl, d);

	M::deallocate(b1);
	M::deallocate(a1);
}


} // namespace libtensor

#endif // LIBTENSOR_LINALG_ADAPTIVE_IJKL_IPJ_PLK_X_H
