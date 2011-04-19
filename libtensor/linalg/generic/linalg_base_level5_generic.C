#include "linalg_base_level5_generic.h"

namespace libtensor {


void linalg_base_level5_generic::ijk_ipkq_jpq_x(
	size_t ni, size_t nj, size_t nk, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t p = 0; p < np; p++) {

		const double *a1 = a + (i * np + p) * nk * nq;
		const double *b1 = b + (j * np + p) * nq;
		double *c1 = c + (i * nj + j) * nk;

		for(size_t k = 0; k < nk; k++) {

			size_t kq0 = k * nq;

			for(size_t q = 0; q < nq; q++) {
				c1[k] += d * a1[kq0 + q] * b1[q];
			}
		}
	}
	}
	}
}


void linalg_base_level5_generic::ijk_pikq_jpq_x(
	size_t ni, size_t nj, size_t nk, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t p = 0; p < np; p++) {

		const double *a1 = a + (p * ni + i) * nk * nq;
		const double *b1 = b + (j * np + p) * nq;
		double *c1 = c + (i * nj + j) * nk;

		for(size_t k = 0; k < nk; k++) {

			size_t kq0 = k * nq;

			for(size_t q = 0; q < nq; q++) {
				c1[k] += d * a1[kq0 + q] * b1[q];
			}
		}
	}
	}
	}
}


void linalg_base_level5_generic::ijk_piqk_jpq_x(
	size_t ni, size_t nj, size_t nk, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t p = 0; p < np; p++) {

		const double *a1 = a + (p * ni + i) * nq * nk;
		const double *b1 = b + (j * np + p) * nq;
		double *c1 = c + (i * nj + j) * nk;

		for(size_t q = 0; q < nq; q++) {

			size_t qk0 = q * nk;

			for(size_t k = 0; k < nk; k++) {
				c1[k] += d * a1[qk0 + k] * b1[q];
			}
		}
	}
	}
	}
}


void linalg_base_level5_generic::ijkl_ipl_kpj_x(
	size_t ni, size_t nj, size_t nk, size_t nl, size_t np,
	const double *a, const double *b, double *c, double d) {

	for(size_t i = 0; i < ni; i++) {
		for(size_t k = 0; k < nk; k++) {

			const double *a1 = a + i * np * nl;
			const double *b1 = b + k * np * nj;

			for(size_t p = 0; p < np; p++) {
			for(size_t j = 0; j < nj; j++) {

				size_t ijk = ((i * nj + j) * nk + k) * nl;
				size_t pj = p * nj + j;
				size_t pl0 = p * nl;

				for(size_t l = 0; l < nl; l++) {
					c[ijk + l] += d * a1[pl0 + l] * b1[pj];
				}
			}
			}
		}
		}
}


} // namespace libtensor
