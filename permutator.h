#ifndef __LIBTENSOR_PERMUTATOR_H
#define __LIBTENSOR_PERMUTATOR_H

#include "defs.h"
#include "exception.h"
#include "dimensions.h"
#include "permutation.h"

namespace libtensor {

template<typename _T>
class permutator {
public:
	static void permute(const _T *src, _T *dst, const dimensions &d,
		const permutation &p);

private:
	static void permute4(const _T *src, _T *dst, const dimensions &d,
		const permutation &p);
};

template<typename _T>
void permutator<_T>::permute(const _T *src, _T *dst, const dimensions &d,
	const permutation &p) {

	if(p.get_order() == 4) permute4(src, dst, d, p);
}

template<typename _T>
inline void permutator<_T>::permute4(const _T *src, _T *dst,
	const dimensions &d, const permutation &p) {

	dimensions dperm(d); dperm.permute(p);

	register const _T *psrc = src;
	size_t imax = d[0], jmax = d[1], kmax = d[2], lmax = d[3];
	size_t iinc = dperm.get_increment(p[0]), jinc = dperm.get_increment(p[1]),
		kinc = dperm.get_increment(p[2]), linc = dperm.get_increment(p[3]);

	_T *pdsti = dst, *pdstj, *pdstk;
	for(size_t i=0; i<imax; i++) {
		pdstj = pdsti;
		for(size_t j=0; j<jmax; j++) {
			pdstk = pdstj;
			for(size_t k=0; k<kmax; k++) {
				register _T *pdstl = pdstk;
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

