#ifndef LIBTENSOR_CONTRACT2_2_3I_H
#define LIBTENSOR_CONTRACT2_2_3I_H

#include "defs.h"
#include "exception.h"
#include "dimensions.h"
#include "permutation.h"

namespace libtensor {

/**	\brief Contracts two fourth-order tensors over three indexes.

	Performs contraction:
	\f[ c_{ij} = \mathcal{P}_c \sum_{klm}
		\mathcal{P}_a a_{iklm} \mathcal{P}_b b_{jklm} \f]

	\ingroup libtensor_tod
**/
class contract2_2_3i {
public:
	static void contract(double *c, const dimensions<2> &dc,
		const permutation<2> &pc, const double *a,
		const dimensions<4> &da, const permutation<4> &pa,
		const double *b, const dimensions<4> &db,
		const permutation<4> &pb) throw(exception);

	static void contract(double *c, const dimensions<2> &dc,
		const permutation<2> &pc, const double *a,
		const dimensions<4> &da, const permutation<4> &pa,
		const double *b, const dimensions<4> &db,
		const permutation<4> &pb, double x) throw(exception);

private:
	/**	\brief \f$ c_{ij} = \sum_{klm} a_{iklm} b_{jklm} \f$
	**/
	static void c_01_0123_0123(double *c, const dimensions<2> &dc,
		const double *a, const dimensions<4> &da,
		const double *b, const dimensions<4> &db);

	/**	\brief \f$ c_{ij} = \sum_{klm} a_{klim} b_{kljm} \f$
	**/
	static void c_01_2013_2013(double *c, const dimensions<2> &dc,
		const double *a, const dimensions<4> &da,
		const double *b, const dimensions<4> &db);

	/**	\brief \f$ c_{ij} = c_{ij} + x\sum_{klm} a_{klim} b_{kljm} \f$
	**/
	static void c_01_2013_2013a(double *c, const dimensions<2> &dc,
		const double *a, const dimensions<4> &da,
		const double *b, const dimensions<4> &db, double x);
};

} // namespace libtensor

#endif // LIBTENSOR_CONTRACT2_2_3I_H

