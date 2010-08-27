#ifndef LIBTENSOR_ALGO_IJKL_PKIQ_JPLQ_H
#define LIBTENSOR_ALGO_IJKL_PKIQ_JPLQ_H

#include <cstring>

namespace libtensor {


template<typename Impl>
void algo_ijkl_pkiq_jplq(const double *a, const double *b, double *c, double d,
	size_t ni, size_t nj, size_t nk, size_t nl, size_t np, size_t nq) {

	size_t npq = np * nq;
	size_t nik = ni * nk;
	size_t njl = nj * nl;
	size_t njkl = njl * nk;
	size_t npq1 = (npq % 4 == 0) ? npq : npq + 4 - npq % 4;
	size_t nik1 = (nik % 4 == 0) ? nik : nik + 4 - nik % 4;
	size_t njl1 = (njl % 4 == 0) ? njl : njl + 4 - njl % 4;

	double *a1 = Impl::allocate(nik * npq1);
	double *b1 = Impl::allocate(njl * npq1);
	double *c1 = Impl::allocate(nik * njl1);

	//	a_pkiq -> a_ikpq

	const double *aa = a;
	for(size_t p = 0; p < np; p++) {
	for(size_t k = 0; k < nk; k++) {
	for(size_t i = 0; i < ni; i++, aa += nq) {
		memcpy(a1 + (i * nk + k) * npq1 + p * nq, aa,
			sizeof(double) * nq);
	}
	}
	}

	//	b_jplq -> b_jlpq

	const double *bb = b;
	for(size_t j = 0; j < nj; j++) {
	for(size_t p = 0; p < np; p++) {
	for(size_t l = 0; l < nl; l++, bb += nq) {
		memcpy(b1 + (j * nl + l) * npq1 + p * nq, bb,
			sizeof(double) * nq);
	}
	}
	}

	//	c_ijkl -> c_ikjl

	double *cc = c;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t k = 0; k < nk; k++, cc += nl) {
		memcpy(c1 + (i * nk + k) * njl1 + j * nl, cc,
			sizeof(double) * nl);
	}
	}
	}

	//	c_ikjl += d * a_ikpq b_jlpq

	Impl::ij_ip_jp(a1, b1, c1, d, nik, njl, npq, npq1, njl1, npq1);

	//	c_ikjl -> c_ijkl

	for(size_t i = 0; i < ni; i++) {
	for(size_t k = 0; k < nk; k++) {
		cc = c1 + (i * nk + k) * njl1;
		for(size_t j = 0; j < nj; j++, cc += nl) {
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

#endif // LIBTENSOR_ALGO_IJKL_PKIQ_JPLQ_H
