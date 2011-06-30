#ifndef LIBTENSOR_LINALG_ADAPTIVE_TRP_IJK_JIK_H
#define LIBTENSOR_LINALG_ADAPTIVE_TRP_IJK_JIK_H

#include <cstring> // for memcpy

namespace libtensor {


/**	\brief Transpose \f$ b_{ijk} = a_{jik} \f$

	\ingroup libtensor_linalg
 **/
struct trp_ijk_jik {

	static void transpose(
		size_t ni, size_t nj, size_t nk,
		const double *a, size_t sja,
		double *b, size_t sib) {

		const double *aa = a;
		for(size_t j = 0; j < nj; j++, aa += sja) {
		for(size_t i = 0; i < ni; i++) {
			memcpy(b + i * sib + j * nk, aa + i * nk,
				sizeof(double) * nk);
		}
		}
	}

	static void transpose(
		size_t ni, size_t nj, size_t nk,
		const double *a, size_t sia, size_t sja,
		double *b, size_t sib) {

		const double *aa = a;
		for(size_t j = 0; j < nj; j++, aa += sja) {
		for(size_t i = 0; i < ni; i++) {
			memcpy(b + i * sib + j * nk, aa + i * sia,
				sizeof(double) * nk);
		}
		}
	}
};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_ADAPTIVE_TRP_IJK_JIK_H
