#include "contract2_2_2i.h"

namespace libtensor {

void contract2_2_2i::contract(
	double *c, const dimensions<2> &dc, const permutation<2> &pc,
	const double *a, const dimensions<2> &da, const permutation<2> &pa,
	const double *b, const dimensions<4> &db, const permutation<4> &pb)
	throw(exception) {

	if(pc.is_identity() && pa.is_identity() && pb.is_identity()) {
		c_01_01_0123(c, dc, a, da, b, db);
	} else {
		throw_exc("contract2_2_2i", "contract()",
			"Contraction not implemented");
	}
}

void contract2_2_2i::c_01_01_0123(double *c, const dimensions<2> &dc,
	const double *a, const dimensions<2> &da, const double *b,
	const dimensions<4> &db) throw(exception) {
#ifdef LIBTENSOR_DEBUG
	if(dc[0]!=db[0]) {
		throw_exc("contract2_2_2i", "c_01_01_0123()",
			"Inconsistent dimension: i");
	}
	if(dc[1]!=db[1]) {
		throw_exc("contract2_2_2i", "c_01_01_0123()",
			"Inconsistent dimension: j");
	}
	if(da[0]!=db[2]) {
		throw_exc("contract2_2_2i", "c_01_01_0123()",
			"Inconsistent dimension: k");
	}
	if(da[1]!=db[3]) {
		throw_exc("contract2_2_2i", "c_01_01_0123()",
			"Inconsistent dimension: l");
	}
#endif
	size_t sza = da.get_size(), szc = dc.get_size();
	const double *pb = b;
	double *pc = c;
	for(size_t i=0; i<szc; i++) {
		*pc = cblas_ddot(sza, a, 1, pb, 1);
		pc++; pb+=sza;
	}
}

} // namespace libtensor

