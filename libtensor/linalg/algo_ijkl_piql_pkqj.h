#ifndef LIBTENSOR_ALGO_IJKL_PIQL_PKQJ_H
#define LIBTENSOR_ALGO_IJKL_PIQL_PKQJ_H

#include <cstring>

namespace libtensor {


template<typename Impl>
void algo_ijkl_piql_pkqj(const double *a, const double *b, double *c, double d,
	size_t ni, size_t nj, size_t nk, size_t nl, size_t np, size_t nq) {

	size_t npq = np * nq;
	size_t nil = ni * nl;
	size_t nkj = nk * nj;
	size_t njkl = nkj * nl;
	size_t npq1 = (npq % 4 == 0) ? npq : npq + 4 - npq % 4;
	size_t nil1 = (nil % 4 == 0) ? nil : nil + 4 - nil % 4;
	size_t nkj1 = (nkj % 4 == 0) ? nkj : nkj + 4 - nkj % 4;

	double *a1 = Impl::allocate(npq * nil1);
	double *b1 = Impl::allocate(npq * nkj1);
	double *c1 = Impl::allocate(nkj * nil1);

	//	a_piql -> a_pqil

	const double *aa = a;
	for(size_t p = 0; p < np; p++) {
	for(size_t i = 0; i < ni; i++) {
	for(size_t q = 0; q < nq; q++, aa += nl) {
		memcpy(a1 + (p * nq + q) * nil1 + i * nl, aa,
			sizeof(double) * nl);
	}
	}
	}

	//	b_pkqj -> b_pqkj

	const double *bb = b;
	for(size_t p = 0; p < np; p++) {
	for(size_t k = 0; k < nk; k++) {
	for(size_t q = 0; q < nq; q++, bb += nj) {
		memcpy(b1 + (p * nq + q) * nkj1 + k * nj, bb,
			sizeof(double) * nj);
	}
	}
	}

	//	c_ijkl -> c_kjil

	double *cc = c;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t k = 0; k < nk; k++, cc += nl) {
		memcpy(c1 + (k * nj + j) * nil1 + i * nl, cc,
			sizeof(double) * nl);
	}
	}
	}

	//	c_kjil += d * b_pqkj a_pqil

	Impl::ij_pi_pj(b1, a1, c1, d, nkj, nil, npq, nil1, nkj1, nil1);

	//	c_kjil -> c_ijkl

	for(size_t k = 0; k < nk; k++) {
	for(size_t j = 0; j < nj; j++) {
		cc = c + (j * nk + k) * nl;
		for(size_t i = 0; i < ni; i++, cc += njkl) {
			memcpy(cc, c1 + (k * nj + j) * nil1 + i * nl,
				sizeof(double) * nl);
		}
	}
	}

	Impl::deallocate(c1);
	Impl::deallocate(b1);
	Impl::deallocate(a1);
}


} // namespace libtensor

#endif // LIBTENSOR_ALGO_IJKL_PIQL_PKQJ_H
