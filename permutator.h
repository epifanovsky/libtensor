#ifndef LIBTENSOR_PERMUTATOR_H
#define LIBTENSOR_PERMUTATOR_H

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
template<size_t N, typename T>
class permutator {
public:
	/**	\brief Permutes raw data
		\param src Data source.
		\param dst Data destination.
		\param d Dimensions of the source data.
		\param p Permutation to be applied
	**/
	static void permute(const T *src, T *dst, const dimensions<N> &d,
		const permutation<N> &p);

};

template<typename T>
class permutator<2,T> {
public:
	static void permute(const T *src, T *dst, const dimensions<2> &d,
		const permutation<2> &p);
};

template<typename T>
class permutator<4,T> {
public:
	static void permute(const T *src, T *dst, const dimensions<4> &d,
		const permutation<4> &p);
};

template<size_t N, typename T>
void permutator<N,T>::permute(const T *src, T *dst, const dimensions<N> &d,
	const permutation<N> &p) {

}

template<typename T>
void permutator<2,T>::permute(const T *src, T *dst, const dimensions<2> &d,
	const permutation<2> &p) {
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
void permutator<2,double>::permute(const double *src, double *dst,
	const dimensions<2> &d, const permutation<2> &p);

template<typename T>
void permutator<4,T>::permute(const T *src, T *dst,
	const dimensions<4> &d, const permutation<4> &p) {

/*
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
*/
}

} // namespace libtensor

#endif // LIBTENSOR_PERMUTATOR_H

