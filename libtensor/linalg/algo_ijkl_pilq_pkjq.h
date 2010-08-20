#ifndef LIBTENSOR_ALGO_IJKL_PILQ_PKJQ_H
#define LIBTENSOR_ALGO_IJKL_PILQ_PKJQ_H

#include <cstring>

namespace libtensor {


template<typename Impl>
void algo_ijkl_pilq_pkjq(const double *a, const double *b, double *c, double d,
	size_t ni, size_t nj, size_t nk, size_t nl, size_t np, size_t nq) {

	size_t npq = np * nq;
	size_t nil = ni * nl;
	size_t nkj = nk * nj;
	size_t njkl = nkj * nl;
	size_t npq1 = (npq % 4 == 0) ? npq : npq + 4 - npq % 4;
	size_t nil1 = (nil % 4 == 0) ? nil : nil + 4 - nil % 4;
	size_t nkj1 = (nkj % 4 == 0) ? nkj : nkj + 4 - nkj % 4;

	double *a1 = Impl::allocate(nil * npq1);
	double *b1 = Impl::allocate(nkj * npq1);
	double *c1 = Impl::allocate(nkj * nil1);

	//	a_pilq -> a_ilpq

	const double *aa = a;
	for(size_t p = 0; p < np; p++) {
	for(size_t i = 0; i < ni; i++) {
	for(size_t l = 0; l < nl; l++, aa += nq) {
		memcpy(a1 + (i * nl + l) * npq1 + p * nq, aa,
			sizeof(double) * nq);
	}
	}
	}

	//	b_pkjq -> b_kjpq

	const double *bb = b;
	for(size_t p = 0; p < np; p++) {
	for(size_t k = 0; k < nk; k++) {
	for(size_t j = 0; j < nj; j++, bb += nq) {
		memcpy(b1 + (k * nj + j) * npq1 + p * nq, bb,
			sizeof(double) * nq);
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

	//	c_kjil += d * b_kjpq a_ilpq

	Impl::ij_ip_jp(b1, a1, c1, d, nkj, nil, npq, npq1, nil1, npq1);

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

#endif // LIBTENSOR_ALGO_IJKL_PILQ_PKJQ_H
