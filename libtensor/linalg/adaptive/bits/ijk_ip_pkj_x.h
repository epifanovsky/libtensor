#ifndef LIBTENSOR_LINALG_ADAPTIVE_IJK_IP_PKJ_X_H
#define LIBTENSOR_LINALG_ADAPTIVE_IJK_IP_PKJ_X_H

namespace libtensor {


template<typename M, typename L1, typename L2, typename L3>
void linalg_base_level4_adaptive<M, L1, L2, L3>::ijk_ip_pkj_x(
	size_t ni, size_t nj, size_t nk, size_t np,
	const double *a, size_t sia,
	const double *b, size_t skb, size_t spb,
	double *c, size_t sjc, size_t sic,
	double d) {

	size_t njk = nj * nk;
	size_t njk1 = (njk % 4 == 0) ? njk : njk + 4 - njk % 4;

	double *b1 = M::allocate(np * njk1);

	for(size_t p = 0; p < np; p++) {
		L2::ij_ji(nj, nk, b + p * spb, skb, b1 + p * njk1, nk);
	}

	if(sjc == nk) {
		L3::ij_ip_pj_x(ni, njk, np, a, sia, b1, njk1, c, sic, d);
	} else {
		double *c1 = M::allocate(ni * njk);
		for(size_t ijk = 0; ijk < ni * njk; ijk++) c1[ijk] = 0.0;
		L3::ij_ip_pj_x(ni, njk, np, a, sia, b1, njk1, c1, njk, 1.0);
		for(size_t i = 0; i < ni; i++) {
			double *c2 = c1 + i * njk;
			double *c3 = c + i * sic;
			for(size_t j = 0; j < nj; j++) {
				L1::i_i_x(nk, c2 + j * nk, 1, d,
					c3 + j * sjc, 1);
			}
		}
		M::deallocate(c1);
	}

	M::deallocate(b1);
}


} // namespace libtensor

#endif // LIBTENSOR_LINALG_ADAPTIVE_IJK_IP_PKJ_X_H
