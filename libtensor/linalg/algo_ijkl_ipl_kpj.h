#ifndef LIBTENSOR_ALGO_IJKL_IPL_KPJ_H
#define LIBTENSOR_ALGO_IJKL_IPL_KPJ_H

#include <cstring>

namespace libtensor {


template<typename Impl>
void algo_ijkl_ipl_kpj(const double *a, const double *b, double *c, double d,
	size_t ni, size_t nj, size_t nk, size_t nl, size_t np) {

	size_t nil = ni * nl;
	size_t nkj = nk * nj;
	size_t njkl = nkj * nl;
	size_t np1 = (np % 4 == 0) ? np : np + 4 - np % 4;
	size_t nil1 = (nil % 4 == 0) ? nil : nil + 4 - nil % 4;
	size_t nkj1 = (nkj % 4 == 0) ? nkj : nkj + 4 - nkj % 4;

	double *a1 = Impl::allocate(np * nil1);
	double *b1 = Impl::allocate(np * nkj1);
	double *c1 = Impl::allocate(nkj * nil1);

	//	a_ipl -> a_pil

	const double *aa = a;
	for(size_t i = 0; i < ni; i++) {
	for(size_t p = 0; p < np; p++, aa += nl) {
		memcpy(a1 + p * nil1 + i * nl, aa, sizeof(double) * nl);
	}
	}

	//	b_kpj -> b_pkj

	const double *bb = b;
	for(size_t k = 0; k < nk; k++) {
	for(size_t p = 0; p < np; p++, bb += nj) {
		memcpy(b1 + p * nkj1 + k * nj, bb, sizeof(double) * nj);
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

	//	c_kjil += d * b_pkj a_pil

	Impl::ij_pi_pj(b1, a1, c1, d, nkj, nil, np, nil1, nkj1, nil1);

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

#endif // LIBTENSOR_ALGO_IJKL_IPL_KPJ_H
