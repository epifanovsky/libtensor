#ifndef LIBTENSOR_LINALG_BASE_LEVEL1_QCHEM_H
#define LIBTENSOR_LINALG_BASE_LEVEL1_QCHEM_H

#include "../generic/linalg_base_level1_generic.h"

namespace libtensor {


/**	\brief Level-1 linear algebra operations (Q-Chem)

	\ingroup libtensor_linalg
 **/
struct linalg_base_level1_qchem : public linalg_base_level1_generic {


	static void add_i_i_x_x(
		size_t ni,
		const double *a, size_t sia,
		double b,
		double *c, size_t sic,
		double d) {

		CL_DAXPY(ni, d, (double*)a, sia, c, sic);
		double db = d * b;
		if(sic == 1) {
			for(size_t i = 0; i < ni; i++) c[i] += db;
		} else {
			for(size_t i = 0; i < ni; i++) c[i * sic] += db;
		}
	}


	static void i_x(
		size_t ni,
		double a,
		double *c, size_t sic) {

		CL_DSCAL(ni, a, c, sic);
	}


	static double x_p_p(
		size_t np,
		const double *a, size_t spa,
		const double *b, size_t spb) {

		return CL_DDOT(np, (double*)a, spa, (double*)b, spb);
	}


	static void i_i_x(
		size_t ni,
		const double *a, size_t sia,
		double b,
		double *c, size_t sic) {

		CL_DAXPY(ni, b, (double*)a, sia, c, sic);
	}

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_LEVEL1_QCHEM_H
