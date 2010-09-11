#ifndef LIBTENSOR_LINALG_ADAPTIVE_IJKL_PKIQ_PJQL_X_H
#define LIBTENSOR_LINALG_ADAPTIVE_IJKL_PKIQ_PJQL_X_H

#include "trp_ijkl_ikjl.h"
#include "trp_ijkl_kjil.h"

namespace libtensor {


template<typename M, typename L1, typename L2, typename L3>
void linalg_base_level6_adaptive<M, L1, L2, L3>::ijkl_pkiq_pjql_x(
	size_t ni, size_t nj, size_t nk, size_t nl, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	size_t npq = np * nq;
	size_t nik = ni * nk;
	size_t njl = nj * nl;
	size_t njkl = njl * nk;
	size_t npq1 = (npq % 4 == 0) ? npq : npq + 4 - npq % 4;
	size_t nik1 = (nik % 4 == 0) ? nik : nik + 4 - nik % 4;
	size_t njl1 = (njl % 4 == 0) ? njl : njl + 4 - njl % 4;

	double *a1 = M::allocate(nik * npq1);
	double *b1 = M::allocate(npq * njl1);
	double *c1 = M::allocate(nik * njl1);

	//	a1_ikpq <- a_pkiq
	trp_ijkl_kjil::transpose(ni, nk, np, nq, a, ni * nq, a1, npq1);

	//	b1_pqjl <- b_pjql
	trp_ijkl_ikjl::transpose(np, nq, nj, nl, b, nq * nl, b1, njl1);

	//	c1_ikjl <- c_ijkl
	trp_ijkl_ikjl::transpose(ni, nk, nj, nl, c, nk * nl, c1, njl1);

	//	c1_ikjl += d * a1_ikpq b1_pqjl
	L3::ij_ip_pj_x(nik, njl, npq, a1, npq1, b1, njl1, c1, njl1, d);

	//	c_ijkl <- c1_ikjl
	trp_ijkl_ikjl::transpose(ni, nj, nk, nl, c1, njl1, c, nk * nl);

	M::deallocate(c1);
	M::deallocate(b1);
	M::deallocate(a1);
}


} // namespace libtensor

#endif // LIBTENSOR_LINALG_ADAPTIVE_IJKL_PKIQ_PJQL_X_H
