#ifndef LIBTENSOR_ALGO_IJKL_PLIQ_PJQK_H
#define LIBTENSOR_ALGO_IJKL_PLIQ_PJQK_H

#include <cstring>

namespace libtensor {


template<typename Impl>
void algo_ijkl_pliq_pjqk(const double *a, const double *b, double *c, double d,
	size_t ni, size_t nj, size_t nk, size_t nl, size_t np, size_t nq) {

	size_t npq = np * nq;
	size_t nil = ni * nl;
	size_t njk = nj * nk;
	size_t njkl = njk * nl;
	size_t npq1 = (npq % 4 == 0) ? npq : npq + 4 - npq % 4;
	size_t nil1 = (nil % 4 == 0) ? nil : nil + 4 - nil % 4;
	size_t njk1 = (njk % 4 == 0) ? njk : njk + 4 - njk % 4;

	double *a1 = Impl::allocate(nil * npq1);
	double *b1 = Impl::allocate(npq * njk1);
	double *c1 = Impl::allocate(njk * nil1);

	//	a_pliq -> a_ilpq

	const double *aa = a;
	for(size_t p = 0; p < np; p++) {
	for(size_t l = 0; l < nl; l++) {
	for(size_t i = 0; i < ni; i++, aa += nq) {
		memcpy(a1 + (i * nl + l) * npq1 + p * nq, aa,
			sizeof(double) * nq);
	}
	}
	}

	//	b_pjqk -> b_pqjk

	const double *bb = b;
	for(size_t p = 0; p < np; p++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t q = 0; q < nq; q++, bb += nk) {
		memcpy(b1 + (p * nq + q) * njk1 + j * nk, bb,
			sizeof(double) * nk);
	}
	}
	}

	//	c_ijkl -> c_jkil

	double *cc = c;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t k = 0; k < nk; k++, cc += nl) {
		memcpy(c1 + (j * nk + k) * nil1 + i * nl, cc,
			sizeof(double) * nl);
	}
	}
	}

	//	c_jkil += d * b_pqjk a_ilpq

	Impl::ij_pi_jp(b1, a1, c1, d, njk, nil, npq, nil1, npq1, njk1);

	//	c_jkil -> c_ijkl

	for(size_t j = 0; j < nj; j++) {
	for(size_t k = 0; k < nk; k++) {
		cc = c1 + (j * nk + k) * nil1;
		for(size_t i = 0; i < ni; i++, cc += nl) {
			memcpy(c + ((i * nj + j) * nk + k) * nl, cc,
				sizeof(double) * nl);
		}
	}
	}

	Impl::deallocate(c1);
	Impl::deallocate(b1);
	Impl::deallocate(a1);
}


} // namespace libtensor

#endif // LIBTENSOR_ALGO_IJKL_PLIQ_PJQK_H
