#ifndef LIBTENSOR_TOD_CONTRACT2_IMPL_113_H
#define LIBTENSOR_TOD_CONTRACT2_IMPL_113_H

#include "defs.h"
#include "exception.h"
#include "tod_contract2_impl.h"

namespace libtensor {

/**	\brief Contracts two fourth-order tensors over three indexes.

	Performs contractions:
	\f[ c_{ij} = \mathcal{P}_c \sum_{pqr}
		\mathcal{P}_a a_{ipqr} \mathcal{P}_b b_{jpqr} \f]
	\f[ c_{ij} = c_{ij} + d \cdot \mathcal{P}_c \sum_{pqr}
		\mathcal{P}_a a_{ipqr} \mathcal{P}_b b_{jpqr} \f]

	\ingroup libtensor_tod
**/
template<>
class tod_contract2_impl<1,1,3> {
public:
	/**	\brief \f$ c_{ij} = \mathcal{P}_c \sum_{pqr}
		\mathcal{P}_a a_{ipqr} \mathcal{P}_b b_{jpqr} \f$
	**/
	static void contract(double *c, const dimensions<2> &dc,
		const permutation<2> &pc, const double *a,
		const dimensions<4> &da, const permutation<4> &pa,
		const double *b, const dimensions<4> &db,
		const permutation<4> &pb) throw(exception);

	/**	\brief \f$ c_{ij} = c_{ij} + d \cdot \mathcal{P}_c \sum_{pqr}
		\mathcal{P}_a a_{ipqr} \mathcal{P}_b b_{jpqr} \f$
	**/
	static void contract(double *c, const dimensions<2> &dc,
		const permutation<2> &pc, const double *a,
		const dimensions<4> &da, const permutation<4> &pa,
		const double *b, const dimensions<4> &db,
		const permutation<4> &pb, double d) throw(exception);

private:
	/**	\brief \f$ c_{ij} = \sum_{pqr} a_{ipqr} b_{jpqr} \f$
	**/
	static void c_01_0123_0123(double *c, const dimensions<2> &dc,
		const double *a, const dimensions<4> &da,
		const double *b, const dimensions<4> &db);

	/**	\brief \f$ c_{ij} = c_{ij} + d \sum_{pqr} a_{ipqr} b_{jpqr} \f$
	**/
	static void c_01_0123_0123(double *c, const dimensions<2> &dc,
		const double *a, const dimensions<4> &da,
		const double *b, const dimensions<4> &db, double d);

	/**	\brief \f$ c_{ij} = \sum_{pqr} a_{pqir} b_{pqjr} \f$
	**/
	static void c_01_2013_2013(double *c, const dimensions<2> &dc,
		const double *a, const dimensions<4> &da,
		const double *b, const dimensions<4> &db);

	/**	\brief \f$ c_{ij} = c_{ij} + d \sum_{pqr} a_{pqir} b_{pqjr} \f$
	**/
	static void c_01_2013_2013(double *c, const dimensions<2> &dc,
		const double *a, const dimensions<4> &da,
		const double *b, const dimensions<4> &db, double d);
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_CONTRACT2_IMPL_113_H

