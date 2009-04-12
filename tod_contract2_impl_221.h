#ifndef LIBTENSOR_TOD_CONTRACT2_IMPL_221_H
#define LIBTENSOR_TOD_CONTRACT2_IMPL_221_H

#include "defs.h"
#include "exception.h"
#include "tod_contract2_impl.h"

namespace libtensor {

/**	\brief Contracts two third-order tensors over one %index.

	Performs contractions:
	\f[ c_{ijkl} = \mathcal{P}_c \sum_{p}
		\mathcal{P}_a a_{ijp} \mathcal{P}_b b_{klp} \f]
	\f[ c_{ijkl} = c_{ijkl} + d \cdot \mathcal{P}_c \sum_{p}
		\mathcal{P}_a a_{ijp} \mathcal{P}_b b_{klp} \f]

	\ingroup libtensor_tod
**/
template<>
class tod_contract2_impl<2,2,1> {
public:
	/**	\brief \f$ c_{ijkl} = \mathcal{P}_c \sum_{p}
			\mathcal{P}_a a_{ijp} \mathcal{P}_b b_{klp} \f$
	**/
	static void contract(double *c, const dimensions<4> &dc,
		const permutation<4> &pc, const double *a,
		const dimensions<3> &da, const permutation<3> &pa,
		const double *b, const dimensions<3> &db,
		const permutation<3> &pb) throw(exception);

	/**	\brief \f$ c_{ijkl} = c_{ijkl} + d \cdot \mathcal{P}_c \sum_{p}
			\mathcal{P}_a a_{ijp} \mathcal{P}_b b_{klp} \f$
	**/
	static void contract(double *c, const dimensions<4> &dc,
		const permutation<4> &pc, const double *a,
		const dimensions<3> &da, const permutation<3> &pa,
		const double *b, const dimensions<3> &db,
		const permutation<3> &pb, double d) throw(exception);

private:
	/**	\brief \f$ c_{ijkl} = \sum_{p} a_{ijp} b_{klp} \f$
	**/
	static void c_0123_012_012(double *c, const dimensions<4> &dc,
		const double *a, const dimensions<3> &da,
		const double *b, const dimensions<3> &db);

	/**	\brief \f$ c_{ijkl} = c_{ijkl} + d \sum_{p} a_{ijp} b_{klp} \f$
	**/
	static void c_0123_012_012(double *c, const dimensions<4> &dc,
		const double *a, const dimensions<3> &da,
		const double *b, const dimensions<3> &db, double d);
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_CONTRACT2_IMPL_221_H

