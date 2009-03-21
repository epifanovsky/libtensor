#ifndef LIBTENSOR_CONTRACT2_2_2I_H
#define LIBTENSOR_CONTRACT2_2_2I_H

#include "defs.h"
#include "exception.h"
#include "dimensions.h"
#include "permutation.h"

namespace libtensor {

/**	\brief Contracts a second-order %tensor with a fourth-order %tensor
		over two indexes.

	Performs contraction:
	\f[ c_{ij} = \mathcal{P}_c \sum_{kl}
		\mathcal{P}_a a_{kl} \mathcal{P}_b b_{ijkl} \f]

	\ingroup libtensor_tod
**/
class contract2_2_2i {
public:
	/**	\brief \f[ c_{ij} = \mathcal{P}_c \sum_{kl}
			\mathcal{P}_a a_{kl} \mathcal{P}_b b_{ijkl} \f]
	**/
	static void contract(double *c, const dimensions<2> &dc,
		const permutation<2> &pcc, const double *a,
		const dimensions<2> &da, const permutation<2> &pca,
		const double *b, const dimensions<4> &db,
		const permutation<4> &pcb) throw(exception);

private:
	/**	\brief \f$ c_{ij} = \sum_{kl} a_{kl} b_{ijkl} \f$
	**/
	static void c_01_01_0123(double *c, const dimensions<2> &dc,
		const double *a, const dimensions<2> &da,
		const double *b, const dimensions<4> &db) throw(exception);
};

} // namespace libtensor

#endif // LIBTENSOR_CONTRACT2_2_2I_H

