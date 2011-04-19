#ifndef LIBTENSOR_LINALG_BASE_LEVEL1_ACML_H
#define LIBTENSOR_LINALG_BASE_LEVEL1_ACML_H

#include "../generic/linalg_base_level1_generic.h"

namespace libtensor {


/**	\brief Level-1 linear algebra operations (ACML)

	\ingroup libtensor_linalg
 **/
struct linalg_base_level1_acml : public linalg_base_level1_generic {


	static void i_x(
		size_t ni,
		double a,
		double *c, size_t sic) {

		dscal(ni, a, c, sic);
	}


	static double x_p_p(
		size_t np,
		const double *a, size_t spa,
		const double *b, size_t spb) {

		return ddot(np, (double*)a, spa, (double*)b, spb);
	}


	static void i_i_x(
		size_t ni,
		const double *a, size_t sia,
		double b,
		double *c, size_t sic) {

		daxpy(ni, b, (double*)a, sia, c, sic);
	}

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_LEVEL1_ACML_H
