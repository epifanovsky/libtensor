#include "permutator.h"

namespace libtensor {

template<>
void permutator<2,double>::permute(const double *src, double *dst,
	const dimensions<2> &d, const permutation<2> &p) {

	const double *psrc = src;
	for(size_t i=0; i<d[0]; i++) {
		cblas_dcopy(d[1], psrc, 1, dst+i, d[0]);
		psrc += d[1];
	}
}

} // namespace libtensor

