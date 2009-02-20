#ifndef __LIBTENSOR_PERMUTATOR_H
#define __LIBTENSOR_PERMUTATOR_H

#include <mkl.h>
#include "defs.h"
#include "exception.h"
#include "dimensions.h"
#include "permutation.h"

namespace libtensor {

/**	\brief Permutation engine

	This class does not perform any checks, only does the job.
	Make sure that the permutation is (1) necessary, (2) correct
	before calling.

	\ingroup libtensor
**/
template<typename T>
class permutator {
public:
	/**	\brief Permutes raw data
		\param src Data source.
		\param dst Data destination.
		\param d Dimensions of the source data.
		\param p Permutation to be applied
	**/
	static void permute(const T *src, T *dst, const dimensions &d,
		const permutation &p);

private:
	static void permute2(const T *src, T *dst, const dimensions &d);
	static void permute4(const T *src, T *dst, const dimensions &d,
		const permutation &p);
};

template<typename T>
void permutator<T>::permute(const T *src, T *dst, const dimensions &d,
	const permutation &p) {

	if(d.get_order() == 2) permute2(src, dst, d);
	//if(p.get_order() == 4) permute4(src, dst, d, p);
}

template<typename T>
void permutator<T>::permute2(const T *src, T *dst, const dimensions &d) {
	const T *psrc = src;
	T *pdst = NULL;
	for(size_t i=0; i<d[0]; i++) {
		pdst = dst+i;
		for(size_t j=0; j<d[1]; j++) {
			*pdst = *psrc;
			psrc++; pdst+=d[0];
		}
	}
}

template<>
void permutator<double>::permute2(const double *src, double *dst,
	const dimensions &d) {

	const double *psrc = src;
	for(size_t i=0; i<d[0]; i++) {
		cblas_dcopy(d[1], psrc, 1, dst+i, d[0]);
		psrc += d[1];
	}
}

template<typename T>
inline void permutator<T>::permute4(const T *src, T *dst,
	const dimensions &d, const permutation &p) {

	dimensions dperm(d); dperm.permute(p);

	register const T *psrc = src;
	size_t imax = d[0], jmax = d[1], kmax = d[2], lmax = d[3];
	size_t iinc = dperm.get_increment(p[0]),
		jinc = dperm.get_increment(p[1]),
		kinc = dperm.get_increment(p[2]),
		linc = dperm.get_increment(p[3]);

	T *pdsti = dst, *pdstj, *pdstk;
	for(size_t i=0; i<imax; i++) {
		pdstj = pdsti;
		for(size_t j=0; j<jmax; j++) {
			pdstk = pdstj;
			for(size_t k=0; k<kmax; k++) {
				register T *pdstl = pdstk;
				#pragma loop count(16)
				for(register size_t l=0; l<lmax; l++) {
					*pdstl = *psrc;
					pdstl += linc; psrc++;
				}
			}
			pdstj += jinc;
		}
		pdsti += iinc;
	}
}

} // namespace libtensor

#endif // __LIBTENSOR_PERMUTATOR_H

