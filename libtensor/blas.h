#ifndef LIBTENSOR_BLAS_H
#define LIBTENSOR_BLAS_H

#include "defs.h"
#include "exception.h"

namespace libtensor {

class blas {
public:
	/**	\brief B = ca*A' + cb*B
	 **/
	static void daxpby_trp(const double *a, double *b, size_t ni, size_t nj,
		size_t stepi, size_t stepj, double ca, double cb);

private:
	// a = any, p = +1.0, m = -1.0, z = 0.0
	static void daxpby_trp_aa(const double *a, double *b, size_t ni,
		size_t nj, size_t stepi, size_t stepj, double ca, double cb);
	static void daxpby_trp_pa(const double *a, double *b, size_t ni,
		size_t nj, size_t stepi, size_t stepj, double ca);
	static void daxpby_trp_pp(const double *a, double *b, size_t ni,
		size_t nj, size_t stepi, size_t stepj);
	static void daxpby_trp_pm(const double *a, double *b, size_t ni,
		size_t nj, size_t stepi, size_t stepj);
	static void daxpby_trp_za(const double *a, double *b, size_t ni,
		size_t nj, size_t stepi, size_t stepj, double ca);
	static void daxpby_trp_zp(const double *a, double *b, size_t ni,
		size_t nj, size_t stepi, size_t stepj);
};

} // namespace libtensor

#endif // LIBTENSOR_BLAS_H
