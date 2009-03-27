#include "tod_contract2_impl_022.h"

namespace libtensor {

void tod_contract2_impl<0,2,2>::contract(
	double *c, const dimensions<2> &dc, const permutation<2> &pc,
	const double *a, const dimensions<2> &da, const permutation<2> &pa,
	const double *b, const dimensions<4> &db, const permutation<4> &pb)
	throw(exception) {

	if(pc.is_identity() && pa.is_identity() && pb.is_identity()) {
		c_01_01_0123(c, dc, a, da, b, db);
	} else {
		throw_exc("tod_contract2_impl<0,2,2>", "contract()",
			"Contraction not implemented");
	}
}

void tod_contract2_impl<0,2,2>::contract(
	double *c, const dimensions<2> &dc, const permutation<2> &pc,
	const double *a, const dimensions<2> &da, const permutation<2> &pa,
	const double *b, const dimensions<4> &db, const permutation<4> &pb,
	double d) throw(exception) {

	if(pc.is_identity() && pa.is_identity() && pb.is_identity()) {
		c_01_01_0123(c, dc, a, da, b, db, d);
	} else {
		throw_exc("tod_contract2_impl<0,2,2>", "contract()",
			"Contraction not implemented");
	}
}

void tod_contract2_impl<0,2,2>::c_01_01_0123(double *c, const dimensions<2> &dc,
	const double *a, const dimensions<2> &da, const double *b,
	const dimensions<4> &db) throw(exception) {

	size_t sza = da.get_size(), szc = dc.get_size();
	const double *pb = b;
	double *pc = c;
	for(size_t i=0; i<szc; i++) {
		*pc = cblas_ddot(sza, a, 1, pb, 1);
		pc++; pb+=sza;
	}
}

void tod_contract2_impl<0,2,2>::c_01_01_0123(double *c, const dimensions<2> &dc,
	const double *a, const dimensions<2> &da, const double *b,
	const dimensions<4> &db, double d) throw(exception) {

	size_t sza = da.get_size(), szc = dc.get_size();
	const double *pb = b;
	double *pc = c;
	if(d == 1.0) {
		for(size_t i=0; i<szc; i++) {
			*pc += cblas_ddot(sza, a, 1, pb, 1);
			pc++; pb+=sza;
		}
	} else if(d == -1.0) {
		for(size_t i=0; i<szc; i++) {
			*pc -= cblas_ddot(sza, a, 1, pb, 1);
			pc++; pb+=sza;
		}
	} else {
		for(size_t i=0; i<szc; i++) {
			*pc += d*cblas_ddot(sza, a, 1, pb, 1);
			pc++; pb+=sza;
		}
	}
}

} // namespace libtensor

