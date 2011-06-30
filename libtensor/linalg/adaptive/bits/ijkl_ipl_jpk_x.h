#ifndef LIBTENSOR_LINALG_ADAPTIVE_IJKL_IPL_JPK_X_H
#define LIBTENSOR_LINALG_ADAPTIVE_IJKL_IPL_JPK_X_H

#include "trp_ijk_jik.h"
#include "trp_ijkl_kijl.h"
#include "trp_ijkl_jkil.h"

namespace libtensor {


template<typename M, typename L1, typename L2, typename L3>
void linalg_base_level6_adaptive<M, L1, L2, L3>::ijkl_ipl_jpk_x(
	size_t ni, size_t nj, size_t nk, size_t nl, size_t np,
	const double *a, size_t spa, size_t sia,
	const double *b, size_t spb, size_t sjb,
	double *c, double d) {

	size_t nil = ni * nl;
	size_t nil1 = (nil % 4 == 0) ? nil : nil + 4 - nil % 4;
	size_t njk = nj * nk;
	size_t njk1 = (njk % 4 == 0) ? njk : njk + 4 - njk % 4;

	double *a1 = M::allocate(np * nil1);
	double *b1 = M::allocate(np * njk1);
	double *c1 = M::allocate(njk * nil1);

	//	a1_pil <- a_ipl
	trp_ijk_jik::transpose(np, ni, nl, a, spa, sia, a1, nil1);

	//	b1_pjk <- b_jpk
	trp_ijk_jik::transpose(np, nj, nk, b, spb, sjb, b1, njk1);

	//	c1_jkil <- c_ijkl
	trp_ijkl_kijl::transpose(nj, nk, ni, nl, c, nk * nl, c1, nil1);

	//	c1_jkil += d * a1_pil b1_pjk
	L3::ij_pi_pj_x(njk, nil, np, b1, njk1, a1, nil1, c1, nil1, d);

	//	c_ijkl <- c1_jkil
	trp_ijkl_jkil::transpose(ni, nj, nk, nl, c1, nil1, c, nk * nl);

	M::deallocate(c1);
	M::deallocate(b1);
	M::deallocate(a1);
}


} // namespace libtensor

#endif // LIBTENSOR_LINALG_ADAPTIVE_IJKL_IPL_JPK_X_H
