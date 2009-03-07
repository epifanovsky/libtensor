#ifndef LIBTENSOR_CONTRACT2_4_2I_H
#define LIBTENSOR_CONTRACT2_4_2I_H

#include "defs.h"
#include "exception.h"

namespace libtensor {

/**	\brief Contracts two fourth-order tensors over two indexes.

	Performs contraction:
	\f[ c_{ijkl} = \mathcal{P}_c \sum_{mn}
		\mathcal{P}_a a_{ijmn} \mathcal{P}_b b_{klmn} \f]

	\ingroup libtensor_tod
**/
class contract2_4_2i {
public:
	static void contract(double *c, const dimensions &dc, size_t pcc,
		const double *a, const dimensions &da, size_t pca,
		const double *b, const dimensions &db, size_t pcb);

private:
	/**	\brief \f$ c_{ijkl} = \sum_{mn} a_{ijmn} b_{klmn} \f$
	**/
	static void c_0123_0123_0123(double *c, const dimensions &dc,
		const double *a, const dimensions &da,
		const double *b, const dimensions &db);
};

} // namespace libtensor

#endif // LIBTENSOR_CONTRACT2_4_2I_H

