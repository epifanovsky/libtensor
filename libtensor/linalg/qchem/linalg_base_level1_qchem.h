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

		if(b == 1.0) {
			if(sia == 1) {
				if(sic == 1) {
					mul_i_i_p11(ni, a, b, c);
				} else {
					mul_i_i_pxx(ni, a, sia, b, c, sic);
				}
			} else {
				mul_i_i_pxx(ni, a, sia, b, c, sic);
			}
		} else if(b == -1.0) {
			if(sia == 1) {
				if(sic == 1) {
					mul_i_i_m11(ni, a, b, c);
				} else {
					mul_i_i_mxx(ni, a, sia, b, c, sic);
				}
			} else {
				mul_i_i_mxx(ni, a, sia, b, c, sic);
			}
		} else {
			CL_DAXPY(ni, b, (double*)a, sia, c, sic);
		}
	}

private:
	static void mul_i_i_x_p11(size_t ni,
		const double *a, double b, double *c) {

		for(size_t i = 0; i < ni; i++) {
			c[i] += a[i];
		}
	}

	static void mul_i_i_x_pxx(size_t ni,
		const double *a, size_t sia, double b, double *c, size_t sic) {

		for(size_t i = 0; i < ni; i++) {
			c[sic * i] += a[sia * i];
		}
	}

	static void mul_i_i_x_m11(size_t ni,
		const double *a, double b, double *c) {

		for(size_t i = 0; i < ni; i++) {
			c[i] -= a[i];
		}
	}

	static void mul_i_i_x_mxx(size_t ni,
		const double *a, size_t sia, double b, double *c, size_t sic) {

		for(size_t i = 0; i < ni; i++) {
			c[sic * i] -= a[sia * i];
		}
	}

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_LEVEL1_QCHEM_H
