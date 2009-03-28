#ifndef LIBTENSOR_TOD_CONTRACT2_IMPL_H
#define LIBTENSOR_TOD_CONTRACT2_IMPL_H

#include "defs.h"
#include "exception.h"
#include "dimensions.h"
#include "permutation.h"

namespace libtensor {

/**	\brief Actual implementation of the contraction of two tensors

	\ingroup libtensor_tod
**/
template<size_t N, size_t M, size_t K>
class tod_contract2_impl {
public:

	static void contract(double *c, const dimensions<N+M> &dc,
		const permutation<N+M> &pc, const double *a,
		const dimensions<N+K> &da, const permutation<N+K> &pca,
		const double *b, const dimensions<M+K> &db,
		const permutation<M+K> &pcb) throw(exception);

	static void contract(double *c, const dimensions<N+M> &dc,
		const permutation<N+M> &pc, const double *a,
		const dimensions<N+K> &da, const permutation<N+K> &pca,
		const double *b, const dimensions<M+K> &db,
		const permutation<M+K> &pcb, double d) throw(exception);

};

} // namespace libtensor

#include "tod_contract2_impl_022.h"
#include "tod_contract2_impl_113.h"
#include "tod_contract2_impl_131.h"

#endif // LIBTENSOR_TOD_CONTRACT2_IMPL_H

