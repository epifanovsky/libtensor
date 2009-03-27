#ifndef LIBTENSOR_TOD_CONTRACT2_IMPL_022_H
#define LIBTENSOR_TOD_CONTRACT2_IMPL_022_H

#include "defs.h"
#include "exception.h"
#include "tod_contract2_impl.h"

namespace libtensor {

/**	\brief Contracts a second-order %tensor with a fourth-order %tensor
		over two indexes.

	Performs contractions:
	\f[ c_{ij} = \mathcal{P}_c \sum_{pq}
		\mathcal{P}_a a_{pq} \mathcal{P}_b b_{ijpq} \f]
	\f[ c_{ij} = c_{ij} + d \cdot \mathcal{P}_c \sum_{pq}
		\mathcal{P}_a a_{pq} \mathcal{P}_b b_{ijpq} \f]

	\ingroup libtensor_tod
**/
template<>
class tod_contract2_impl<0,2,2> {
public:
	/**	\brief \f$ c_{ij} = \mathcal{P}_c \sum_{pq}
			\mathcal{P}_a a_{pq} \mathcal{P}_b b_{ijpq} \f$
	**/
	static void contract(double *c, const dimensions<2> &dc,
		const permutation<2> &pcc, const double *a,
		const dimensions<2> &da, const permutation<2> &pca,
		const double *b, const dimensions<4> &db,
		const permutation<4> &pcb) throw(exception);

	/**	\brief \f$ c_{ij} = c_{ij} + d \cdot \mathcal{P}_c \sum_{pq}
			\mathcal{P}_a a_{pq} \mathcal{P}_b b_{ijpq} \f$
	**/
	static void contract(double *c, const dimensions<2> &dc,
		const permutation<2> &pcc, const double *a,
		const dimensions<2> &da, const permutation<2> &pca,
		const double *b, const dimensions<4> &db,
		const permutation<4> &pcb, double d) throw(exception);

private:
	/**	\brief \f$ c_{ij} = \sum_{pq} a_{pq} b_{ijpq} \f$
	**/
	static void c_01_01_0123(double *c, const dimensions<2> &dc,
		const double *a, const dimensions<2> &da,
		const double *b, const dimensions<4> &db) throw(exception);

	/**	\brief \f$ c_{ij} = c_{ij} + d \sum_{pq} a_{pq} b_{ijpq} \f$
	**/
	static void c_01_01_0123(double *c, const dimensions<2> &dc,
		const double *a, const dimensions<2> &da,
		const double *b, const dimensions<4> &db, double d)
			throw(exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_CONTRACT2_IMPL_022_H

