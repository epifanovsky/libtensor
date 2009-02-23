#include "permutator.h"

namespace libtensor {

template<>
void permutator<double>::permute2(const double *src, double *dst,
	const dimensions &d) {

	const double *psrc = src;
	for(size_t i=0; i<d[0]; i++) {
		cblas_dcopy(d[1], psrc, 1, dst+i, d[0]);
		psrc += d[1];
	}
}

} // namespace libtensor

