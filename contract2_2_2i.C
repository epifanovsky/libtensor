#include "contract2_2_2i.h"

namespace libtensor {

void contract2_2_2i::contract(
	double *c, const dimensions &dc, const permutation &pcc,
	const double *a, const dimensions &da, const permutation &pca,
	const double *b, const dimensions &db, const permutation &pcb)
	throw(exception) {

	if(pcc.is_identity() && pca.is_identity() && pcb.is_identity()) {
		c_01_01_0123(c, dc, a, da, b, db);
	} else {
		throw_exc("libtensor::contract2_2_2i", "contract(...)",
			"Contraction not implemented");
	}
}

void contract2_2_2i::c_01_01_0123(double *c, const dimensions &dc,
	const double *a, const dimensions &da, const double *b,
	const dimensions &db) {

}

} // namespace libtensor

