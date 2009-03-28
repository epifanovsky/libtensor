#ifndef LIBTENSOR_TOD_CONTRACT2_IMPL_131_H
#define LIBTENSOR_TOD_CONTRACT2_IMPL_131_H

#include "defs.h"
#include "exception.h"
#include "tod_contract2_impl.h"

namespace libtensor {

/**	\brief Contracts a second-order %tensor with a fourth-order %tensor
		over one %index.

	Performs contractions:
	\f[ c_{ijkl} = \mathcal{P}_c \sum_{p}
		\mathcal{P}_a a_{ip} \mathcal{P}_b b_{jklp} \f]
	\f[ c_{ijkl} = c_{ijkl} + d \cdot \mathcal{P}_c \sum_{p}
		\mathcal{P}_a a_{ip} \mathcal{P}_b b_{jklp} \f]

	\ingroup libtensor_tod
**/
template<>
class tod_contract2_impl<1,3,1> {
public:
	/**	\brief \f$ c_{ijkl} = \mathcal{P}_c \sum_{p}
			\mathcal{P}_a a_{ip} \mathcal{P}_b b_{jklp} \f$
	**/
	static void contract(double *c, const dimensions<4> &dc,
		const permutation<4> &pc, const double *a,
		const dimensions<2> &da, const permutation<2> &pa,
		const double *b, const dimensions<4> &db,
		const permutation<4> &pb) throw(exception);

	/**	\brief \f$ c_{ijkl} = c_{ijkl} + d \cdot \mathcal{P}_c \sum_{p}
			\mathcal{P}_a a_{ip} \mathcal{P}_b b_{jklp} \f$
	**/
	static void contract(double *c, const dimensions<4> &dc,
		const permutation<4> &pc, const double *a,
		const dimensions<2> &da, const permutation<2> &pa,
		const double *b, const dimensions<4> &db,
		const permutation<4> &pb, double d) throw(exception);

private:
	/**	\brief \f$ c_{jikl} = \sum_{p} a_{pi} b_{jpkl} \f$
	**/
	static void c_1023_10_0231(double *c, const dimensions<4> &dc,
		const double *a, const dimensions<2> &da,
		const double *b, const dimensions<4> &db);

	/**	\brief \f$ c_{jikl} = c_{jikl} + d \sum_{p} a_{pi} b_{jpkl} \f$
	**/
	static void c_1023_10_0231(double *c, const dimensions<4> &dc,
		const double *a, const dimensions<2> &da,
		const double *b, const dimensions<4> &db, double d);
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_CONTRACT2_IMPL_131_H

