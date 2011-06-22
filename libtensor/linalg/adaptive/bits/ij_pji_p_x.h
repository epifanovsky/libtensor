#ifndef LIBTENSOR_LINALG_ADAPTIVE_IJ_PJI_P_X_H
#define LIBTENSOR_LINALG_ADAPTIVE_IJ_PJI_P_X_H

namespace libtensor {


template<typename M, typename L1, typename L2>
void linalg_base_level3x_adaptive<M, L1, L2>::ij_pji_p_x(
	size_t ni, size_t nj, size_t np,
	const double *a, size_t sja, size_t spa,
	const double *b, size_t spb,
	double *c, size_t sic,
	double d) {

	size_t nij = ni * nj;
	size_t nij1 = (nij % 4 == 0) ? nij : nij + 4 - nij % 4;

	double *a1 = M::allocate(np * nij1);

	for(size_t p = 0; p < np; p++) {
		L2::ij_ji(ni, nj, a + p * spa, sja, a1 + p * nij1, nj);
	}

	if(sic == nj) {
		L2::i_pi_p_x(nij, np, a1, nij1, b, spb, c, 1, d);
	} else {
		double *c1 = M::allocate(nij);
		for(size_t i = 0; i < nij; i++) c1[i] = 0.0;
		L2::i_pi_p_x(nij, np, a1, nij1, b, spb, c1, 1, d);
		for(size_t i = 0; i < ni; i++) {
			L1::i_i_x(nj, c1 + i * nj, 1, 1.0, c + i * sic, 1);
		}
		M::deallocate(c1);
	}

	M::deallocate(a1);
}


} // namespace libtensor

#endif // LIBTENSOR_LINALG_ADAPTIVE_IJ_PJI_P_X_H
