#include "contract2_0_4i.h"

namespace libtensor {

void contract2_0_4i::contract(
	double *c,
	const double *a, const dimensions<4> &da, const permutation<4> &pca,
	const double *b, const dimensions<4> &db, const permutation<4> &pcb)
	throw(exception) {

	if( pca.is_identity() && pcb.is_identity()) {
		c_0_0123_0123(c, a, da, b, db);
	} else {
		throw_exc("libtensor::contract2_2_2i", "contract(...)",
			"Contraction not implemented");
	}
}

void contract2_0_4i::c_0_0123_0123(double *c,
	const double *a, const dimensions<4> &da, const double *b,
	const dimensions<4> &db) throw(exception) {

#ifdef LIBTENSOR_DEBUG

#endif

	size_t sza = da.get_size();
	size_t szb = db.get_size();
//	*c = cblas_ddot(da.get_size(), a, 1, db.get_size(), b, 1);
	*c = cblas_ddot(sza, a, 1, b, 1);
}

} // namespace libtensor

