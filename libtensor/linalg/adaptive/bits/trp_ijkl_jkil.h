#ifndef LIBTENSOR_LINALG_ADAPTIVE_TRP_IJKL_JKIL_H
#define LIBTENSOR_LINALG_ADAPTIVE_TRP_IJKL_JKIL_H

#include <cstring> // for memcpy

namespace libtensor {


/**	\brief Transpose \f$ b_{ijkl} = a_{jkil} \f$

	\ingroup libtensor_linalg
 **/
struct trp_ijkl_jkil {

	static void transpose(
		size_t ni, size_t nj, size_t nk, size_t nl,
		const double *a, size_t ska,
		double *b, size_t sjb) {

		const double *aa = a;
		for(size_t j = 0; j < nj; j++) {
		for(size_t k = 0; k < nk; k++, aa += ska) {
		for(size_t i = 0; i < ni; i++) {
			memcpy(b + (i * nj + j) * sjb + k * nl, aa + j * nl,
				sizeof(double) * nl);
		}
		}
		}
	}
};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_ADAPTIVE_TRP_IJKL_JKIL_H
