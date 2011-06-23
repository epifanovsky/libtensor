#include "linalg_base_level6_generic.h"

namespace libtensor {


void linalg_base_level6_generic::ijkl_ipl_jpk_x(
	size_t ni, size_t nj, size_t nk, size_t nl, size_t np,
	const double *a, size_t spa, size_t sia,
	const double *b, size_t spb, size_t sjb,
	double *c, double d) {

	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < nj; j++) {

		double *c1 = c + (i * nj + j) * nk * nl;

		for(size_t p = 0; p < np; p++) {

			const double *a1 = a + i * sia + p * spa;
			const double *b1 = b + j * sjb + p * spb;

			for(size_t k = 0; k < nk; k++) {
			for(size_t l = 0; l < nl; l++) {
				c1[k * nl + l] += d * a1[l] * b1[k];
			}
			}
		}
	}
	}
}


void linalg_base_level6_generic::ijkl_ipkq_pljq_x(
	size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	for(size_t i = 0; i < ni; i++) {
	for(size_t l = 0; l < nl; l++) {
	for(size_t p = 0; p < np; p++) {

		const double *a1 = a + (i * np + p) * nk * nq;
		const double *b1 = b + (p * nl + l) * nj * nq;

		for(size_t j = 0; j < nj; j++) {
		for(size_t k = 0; k < nk; k++) {

			size_t ijk = ((i * nj + j) * nk + k) * nl;
			for(size_t q = 0; q < nq; q++) {
				size_t kq = k * nq + q;
				size_t jq = j * nq + q;
				c[ijk + l] += d * a1[kq] * b1[jq];
			}
		}
		}
	}
	}
	}
}


void linalg_base_level6_generic::ijkl_iplq_kpjq_x(
	size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	for(size_t i = 0; i < ni; i++) {
	for(size_t k = 0; k < nk; k++) {
	for(size_t p = 0; p < np; p++) {

		const double *a1 = a + (i * np + p) * nl * nq;
		const double *b1 = b + (k * np + p) * nj * nq;

		for(size_t j = 0; j < nj; j++) {
		for(size_t l = 0; l < nl; l++) {

			size_t ijk = ((i * nj + j) * nk + k) * nl;
			for(size_t q = 0; q < nq; q++) {
				size_t lq = l * nq + q;
				size_t jq = j * nq + q;
				c[ijk + l] += d * a1[lq] * b1[jq];
			}
		}
		}
	}
	}
	}
}


void linalg_base_level6_generic::ijkl_iplq_pkjq_x(
	size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	for(size_t i = 0; i < ni; i++) {
	for(size_t k = 0; k < nk; k++) {
	for(size_t p = 0; p < np; p++) {

		const double *a1 = a + (i * np + p) * nl * nq;
		const double *b1 = b + (p * nk + k) * nj * nq;

		for(size_t j = 0; j < nj; j++) {
		for(size_t l = 0; l < nl; l++) {

			size_t ijk = ((i * nj + j) * nk + k) * nl;
			for(size_t q = 0; q < nq; q++) {
				size_t lq = l * nq + q;
				size_t jq = j * nq + q;
				c[ijk + l] += d * a1[lq] * b1[jq];
			}
		}
		}
	}
	}
	}
}


void linalg_base_level6_generic::ijkl_iplq_pkqj_x(
	size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	for(size_t i = 0; i < ni; i++) {
	for(size_t k = 0; k < nk; k++) {
	for(size_t p = 0; p < np; p++) {

		const double *a1 = a + (i * np + p) * nl * nq;
		const double *b1 = b + (p * nk + k) * nq * nj;

		for(size_t j = 0; j < nj; j++) {
		for(size_t l = 0; l < nl; l++) {

			size_t ijk = ((i * nj + j) * nk + k) * nl;
			for(size_t q = 0; q < nq; q++) {
				size_t lq = l * nq + q;
				size_t qj = q * nj + j;
				c[ijk + l] += d * a1[lq] * b1[qj];
			}
		}
		}
	}
	}
	}
}


void linalg_base_level6_generic::ijkl_ipqk_pljq_x(
	size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	for(size_t i = 0; i < ni; i++) {
	for(size_t l = 0; l < nl; l++) {
	for(size_t p = 0; p < np; p++) {

		const double *a1 = a + (i * np + p) * nq * nk;
		const double *b1 = b + (p * nl + l) * nj * nq;

		for(size_t j = 0; j < nj; j++) {
		for(size_t k = 0; k < nk; k++) {

			size_t ijk = ((i * nj + j) * nk + k) * nl;
			for(size_t q = 0; q < nq; q++) {
				size_t qk = q * nk + k;
				size_t jq = j * nq + q;
				c[ijk + l] += d * a1[qk] * b1[jq];
			}
		}
		}
	}
	}
	}
}


void linalg_base_level6_generic::ijkl_ipql_pkjq_x(
	size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	for(size_t i = 0; i < ni; i++) {
	for(size_t k = 0; k < nk; k++) {
	for(size_t p = 0; p < np; p++) {

		const double *a1 = a + (i * np + p) * nq * nl;
		const double *b1 = b + (p * nk + k) * nj * nq;

		for(size_t j = 0; j < nj; j++) {
		for(size_t q = 0; q < nq; q++) {

			size_t ijk = ((i * nj + j) * nk + k) * nl;
			size_t jq = j * nq + q;
			size_t ql0 = q * nl;

			for(size_t l = 0; l < nl; l++) {
				c[ijk + l] += d * a1[ql0 + l] * b1[jq];
			}
		}
		}
	}
	}
	}
}


void linalg_base_level6_generic::ijkl_ipql_pkqj_x(
	size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	for(size_t i = 0; i < ni; i++) {
	for(size_t k = 0; k < nk; k++) {
	for(size_t p = 0; p < np; p++) {

		const double *a1 = a + (i * np + p) * nq * nl;
		const double *b1 = b + (p * nk + k) * nq * nj;

		for(size_t q = 0; q < nq; q++) {
		for(size_t j = 0; j < nj; j++) {

			size_t ijk = ((i * nj + j) * nk + k) * nl;
			size_t qj = q * nj + j;
			size_t ql0 = q * nl;

			for(size_t l = 0; l < nl; l++) {
				c[ijk + l] += d * a1[ql0 + l] * b1[qj];
			}
		}
		}
	}
	}
	}
}


void linalg_base_level6_generic::ijkl_ipql_qkpj_x(
	size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	for(size_t i = 0; i < ni; i++) {
	for(size_t k = 0; k < nk; k++) {
	for(size_t p = 0; p < np; p++) {
	for(size_t q = 0; q < nq; q++) {

		const double *a1 = a + ((i * np + p) * nq + q) * nl;
		const double *b1 = b + (q * nk + k) * np * nj;

		for(size_t j = 0; j < nj; j++) {

			size_t ijk = ((i * nj + j) * nk + k) * nl;
			size_t pj = p * nj + j;

			for(size_t l = 0; l < nl; l++) {
				c[ijk + l] += d * a1[l] * b1[pj];
			}
		}
	}
	}
	}
	}
}


void linalg_base_level6_generic::ijkl_pilq_kpjq_x(
	size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	for(size_t p = 0; p < np; p++) {
	for(size_t i = 0; i < ni; i++) {
	for(size_t k = 0; k < nk; k++) {

		const double *a1 = a + (p * ni + i) * nl * nq;
		const double *b1 = b + (k * np + p) * nj * nq;

		for(size_t j = 0; j < nj; j++) {
		for(size_t l = 0; l < nl; l++) {

			size_t ijk = ((i * nj + j) * nk + k) * nl;
			for(size_t q = 0; q < nq; q++) {
				size_t lq = l * nq + q;
				size_t jq = j * nq + q;
				c[ijk + l] += d * a1[lq] * b1[jq];
			}
		}
		}
	}
	}
	}
}


void linalg_base_level6_generic::ijkl_pilq_pkjq_x(
	size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	for(size_t p = 0; p < np; p++) {
	for(size_t i = 0; i < ni; i++) {
	for(size_t k = 0; k < nk; k++) {

		const double *a1 = a + (p * ni + i) * nl * nq;
		const double *b1 = b + (p * nk + k) * nj * nq;

		for(size_t j = 0; j < nj; j++) {
		for(size_t l = 0; l < nl; l++) {

			size_t ijk = ((i * nj + j) * nk + k) * nl;
			for(size_t q = 0; q < nq; q++) {
				size_t lq = l * nq + q;
				size_t jq = j * nq + q;
				c[ijk + l] += d * a1[lq] * b1[jq];
			}
		}
		}
	}
	}
	}
}


void linalg_base_level6_generic::ijkl_pikq_pljq_x(
	size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	for(size_t i = 0; i < ni; i++) {
	for(size_t l = 0; l < nl; l++) {
	for(size_t p = 0; p < np; p++) {

		const double *a1 = a + (p * ni + i) * nk * nq;
		const double *b1 = b + (p * nl + l) * nj * nq;

		for(size_t j = 0; j < nj; j++) {
		for(size_t k = 0; k < nk; k++) {

			size_t ijk = ((i * nj + j) * nk + k) * nl;
			for(size_t q = 0; q < nq; q++) {
				size_t kq = k * nq + q;
				size_t jq = j * nq + q;
				c[ijk + l] += d * a1[kq] * b1[jq];
			}
		}
		}
	}
	}
	}
}


void linalg_base_level6_generic::ijkl_piqk_pljq_x(
	size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	for(size_t i = 0; i < ni; i++) {
	for(size_t l = 0; l < nl; l++) {
	for(size_t p = 0; p < np; p++) {

		const double *a1 = a + (p * ni + i) * nq * nk;
		const double *b1 = b + (p * nl + l) * nj * nq;

		for(size_t j = 0; j < nj; j++) {
		for(size_t k = 0; k < nk; k++) {

			size_t ijk = ((i * nj + j) * nk + k) * nl;
			for(size_t q = 0; q < nq; q++) {
				size_t qk = q * nk + k;
				size_t jq = j * nq + q;
				c[ijk + l] += d * a1[qk] * b1[jq];
			}
		}
		}
	}
	}
	}
}


void linalg_base_level6_generic::ijkl_piql_kpqj_x(
	size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	for(size_t p = 0; p < np; p++) {
	for(size_t i = 0; i < ni; i++) {
	for(size_t k = 0; k < nk; k++) {

		const double *a1 = a + (p * ni + i) * nq * nl;
		const double *b1 = b + (k * np + p) * nq * nj;

		for(size_t q = 0; q < nq; q++) {
		for(size_t j = 0; j < nj; j++) {

			size_t ijk = ((i * nj + j) * nk + k) * nl;
			size_t qj = q * nj + j;
			size_t ql0 = q * nl;

			for(size_t l = 0; l < nl; l++) {
				c[ijk + l] += d * a1[ql0 + l] * b1[qj];
			}
		}
		}
	}
	}
	}
}


void linalg_base_level6_generic::ijkl_piql_pkjq_x(
	size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	for(size_t p = 0; p < np; p++) {
	for(size_t i = 0; i < ni; i++) {
	for(size_t k = 0; k < nk; k++) {

		const double *a1 = a + (p * ni + i) * nq * nl;
		const double *b1 = b + (p * nk + k) * nq * nj;

		for(size_t q = 0; q < nq; q++) {
		for(size_t j = 0; j < nj; j++) {

			size_t ijk = ((i * nj + j) * nk + k) * nl;
			size_t jq = j * nq + q;
			size_t ql0 = q * nl;

			for(size_t l = 0; l < nl; l++) {
				c[ijk + l] += d * a1[ql0 + l] * b1[jq];
			}
		}
		}
	}
	}
	}
}


void linalg_base_level6_generic::ijkl_piql_pkqj_x(
	size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	for(size_t p = 0; p < np; p++) {
	for(size_t i = 0; i < ni; i++) {
	for(size_t k = 0; k < nk; k++) {

		const double *a1 = a + (p * ni + i) * nq * nl;
		const double *b1 = b + (p * nk + k) * nq * nj;

		for(size_t q = 0; q < nq; q++) {
		for(size_t j = 0; j < nj; j++) {

			size_t ijk = ((i * nj + j) * nk + k) * nl;
			size_t qj = q * nj + j;
			size_t ql0 = q * nl;

			for(size_t l = 0; l < nl; l++) {
				c[ijk + l] += d * a1[ql0 + l] * b1[qj];
			}
		}
		}
	}
	}
	}
}


void linalg_base_level6_generic::ijkl_piql_qkpj_x(
	size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	for(size_t i = 0; i < ni; i++) {
	for(size_t k = 0; k < nk; k++) {
	for(size_t p = 0; p < np; p++) {
	for(size_t q = 0; q < nq; q++) {

		const double *a1 = a + ((p * ni + i) * nq + q) * nl;
		const double *b1 = b + (q * nk + k) * np * nj;

		for(size_t j = 0; j < nj; j++) {

			size_t ijk = ((i * nj + j) * nk + k) * nl;
			size_t pj = p * nj + j;

			for(size_t l = 0; l < nl; l++) {
				c[ijk + l] += d * a1[l] * b1[pj];
			}
		}
	}
	}
	}
	}
}


void linalg_base_level6_generic::ijkl_pkiq_jplq_x(
	size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	for(size_t p = 0; p < np; p++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t k = 0; k < nk; k++) {

		const double *a1 = a + (p * nk + k) * ni * nq;
		const double *b1 = b + (j * np + p) * nl * nq;

		for(size_t i = 0; i < ni; i++) {
		for(size_t l = 0; l < nl; l++) {

			size_t ijk = ((i * nj + j) * nk + k) * nl;
			size_t iq0 = i * nq;
			size_t lq0 = l * nq;

			for(size_t q = 0; q < nq; q++) {
				c[ijk + l] += d * a1[iq0 + q] * b1[lq0 + q];
			}
		}
		}
	}
	}
	}
}


void linalg_base_level6_generic::ijkl_pkiq_jpql_x(
	size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	for(size_t p = 0; p < np; p++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t k = 0; k < nk; k++) {

		const double *a1 = a + (p * nk + k) * ni * nq;
		const double *b1 = b + (j * np + p) * nq * nl;

		for(size_t i = 0; i < ni; i++) {
		for(size_t q = 0; q < nq; q++) {

			size_t ijk = ((i * nj + j) * nk + k) * nl;
			size_t iq0 = i * nq;
			size_t ql0 = q * nl;

			for(size_t l = 0; l < nl; l++) {
				c[ijk + l] += d * a1[iq0 + q] * b1[ql0 + l];
			}
		}
		}
	}
	}
	}
}


void linalg_base_level6_generic::ijkl_pkiq_pjlq_x(
	size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	for(size_t p = 0; p < np; p++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t k = 0; k < nk; k++) {

		const double *a1 = a + (p * nk + k) * ni * nq;
		const double *b1 = b + (p * nj + j) * nl * nq;

		for(size_t i = 0; i < ni; i++) {
		for(size_t l = 0; l < nl; l++) {

			size_t ijk = ((i * nj + j) * nk + k) * nl;
			size_t iq0 = i * nq;
			size_t lq0 = l * nq;

			for(size_t q = 0; q < nq; q++) {
				c[ijk + l] += d * a1[iq0 + q] * b1[lq0 + q];
			}
		}
		}
	}
	}
	}
}


void linalg_base_level6_generic::ijkl_pkiq_pjql_x(
	size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	for(size_t p = 0; p < np; p++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t k = 0; k < nk; k++) {

		const double *a1 = a + (p * nk + k) * ni * nq;
		const double *b1 = b + (p * nj + j) * nl * nq;

		for(size_t i = 0; i < ni; i++) {
		for(size_t q = 0; q < nq; q++) {

			size_t ijk = ((i * nj + j) * nk + k) * nl;
			size_t iq0 = i * nq;
			size_t ql0 = q * nl;

			for(size_t l = 0; l < nl; l++) {
				c[ijk + l] += d * a1[iq0 + q] * b1[ql0 + l];
			}
		}
		}
	}
	}
	}
}


void linalg_base_level6_generic::ijkl_pliq_jpkq_x(
	size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	for(size_t p = 0; p < np; p++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t l = 0; l < nl; l++) {

		const double *a1 = a + (p * nl + l) * ni * nq;
		const double *b1 = b + (j * np + p) * nk * nq;

		for(size_t i = 0; i < ni; i++) {
		for(size_t k = 0; k < nk; k++) {

			size_t ijk = ((i * nj + j) * nk + k) * nl;
			size_t iq0 = i * nq;
			size_t kq0 = k * nq;

			for(size_t q = 0; q < nq; q++) {
				c[ijk + l] += d * a1[iq0 + q] * b1[kq0 + q];
			}
		}
		}
	}
	}
	}
}


void linalg_base_level6_generic::ijkl_pliq_jpqk_x(
	size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	for(size_t p = 0; p < np; p++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t l = 0; l < nl; l++) {

		const double *a1 = a + (p * nl + l) * ni * nq;
		const double *b1 = b + (j * np + p) * nk * nq;

		for(size_t i = 0; i < ni; i++) {
		for(size_t q = 0; q < nq; q++) {

			size_t ijk0 = (i * nj + j) * nk * nl;
			size_t iq0 = i * nq;
			size_t qk0 = q * nk;

			for(size_t k = 0; k < nk; k++) {
				c[ijk0 + k * nl + l] +=
					d * a1[iq0 + q] * b1[qk0 + k];
			}
		}
		}
	}
	}
	}
}


void linalg_base_level6_generic::ijkl_pliq_pjqk_x(
	size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq,
	const double *a, const double *b, double *c, double d) {

	for(size_t p = 0; p < np; p++) {
	for(size_t j = 0; j < nj; j++) {
	for(size_t l = 0; l < nl; l++) {

		const double *a1 = a + (p * nl + l) * ni * nq;
		const double *b1 = b + (p * nj + j) * nk * nq;

		for(size_t i = 0; i < ni; i++) {
		for(size_t q = 0; q < nq; q++) {

			size_t ijk0 = (i * nj + j) * nk * nl;
			size_t iq0 = i * nq;
			size_t qk0 = q * nk;

			for(size_t k = 0; k < nk; k++) {
				c[ijk0 + k * nl + l] +=
					d * a1[iq0 + q] * b1[qk0 + k];
			}
		}
		}
	}
	}
	}
}


} // namespace libtensor
