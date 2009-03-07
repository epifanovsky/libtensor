#ifndef LIBTENSOR_CONTRACT2_2_3I_H
#define LIBTENSOR_CONTRACT2_2_3I_H

#include "defs.h"
#include "exception.h"

namespace libtensor {

/**	\brief Contracts two fourth-order tensors over three indexes.

	Performs contraction:
	\f[ c_{ij} = \mathcal{P}_c \sum_{klm}
		\mathcal{P}_a a_{iklm} \mathcal{P}_b b_{jklm} \f]

	\ingroup libtensor_tod
**/
class contract2_2_3i {
public:
	static void contract(double *c, const dimensions &dc, size_t pcc,
		const double *a, const dimensions &da, size_t pca,
		const double *b, const dimensions &db, size_t pcb);

private:
	/**	\brief \f$ c_{ij} = \sum_{klm} a_{iklm} b_{jklm} \f$
	**/
	static void c_01_0123_0123(double *c, const dimensions &dc,
		const double *a, const dimensions &da,
		const double *b, const dimensions &db);
};

} // namespace libtensor

#endif // LIBTENSOR_CONTRACT2_2_3I_H

