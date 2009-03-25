#ifndef LIBTENSOR_CONTRACT_2_0_4I_H
#define LIBTENSOR_CONTRACT_2_0_4I_H


#include "defs.h"
#include "exception.h"
#include "dimensions.h"
#include "permutation.h"

namespace libtensor {

/**	\brief Contracts two fourth-order %tensors over four indexes to yield
		a scalar

	Performs contraction:
	\f[ c = \mathcal{P}_c \sum_{ikjl}
		\mathcal{P}_a a_{ijkl} \mathcal{P}_b b_{ijkl} \f]

	\ingroup libtensor_tod
**/
class contract2_0_4i {
public:
	/**	\brief \f[ c = \mathcal{P}_c \sum_{ikjl}
			\mathcal{P}_a a_{ijkl} \mathcal{P}_b b_{ijkl} \f]
	**/
	static void contract(
		double *c,
		const double *a, const dimensions<4> &da, const permutation<4> &pca,
		const double *b, const dimensions<4> &db, const permutation<4> &pcb)
		throw(exception);

private:
	/**	\brief \f$ c = \sum_{ikjl} a_{ikjl} b_{ijkl} \f$
	**/
	static void c_0_0123_0123(double *c,
		const double *a, const dimensions<4> &da,
		const double *b, const dimensions<4> &db) throw(exception);
};

} // namespace libtensor

#endif //LIBTENSOR_CONTRACT_2_0_4I_H
