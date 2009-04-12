#include "tod_contract2_impl_221.h"

namespace libtensor {

void tod_contract2_impl<2,2,1>::contract(double *c, const dimensions<4> &dc,
	const permutation<4> &pc, const double *a, const dimensions<3> &da,
	const permutation<3> &pa, const double *b, const dimensions<3> &db,
	const permutation<3> &pb) throw(exception) {

	if(pc.is_identity() && pa.is_identity() && pb.is_identity()) {
		c_0123_012_012(c, dc, a, da, b, db);
	} else {
		throw_exc("tod_contract2_impl<2,2,1>", "contract()",
			"Contraction not implemented");
	}
}

void tod_contract2_impl<2,2,1>::contract(double *c, const dimensions<4> &dc,
	const permutation<4> &pc, const double *a, const dimensions<3> &da,
	const permutation<3> &pa, const double *b, const dimensions<3> &db,
	const permutation<3> &pb, double d) throw(exception) {

	if(pc.is_identity() && pa.is_identity() && pb.is_identity()) {
		c_0123_012_012(c, dc, a, da, b, db, d);
	} else {
		throw_exc("tod_contract2_impl<2,2,1>", "contract()",
			"Contraction not implemented");
	}
}

void tod_contract2_impl<2,2,1>::c_0123_012_012(double *c,
	const dimensions<4> &dc, const double *a, const dimensions<3> &da,
	const double *b, const dimensions<3> &db) {

	// c_{ijkl} = \sum_{p} a_{ijp} b_{klp}

	size_t szp = da[2];
	size_t szij = da[0]*da[1];
	size_t szkl = db[0]*db[1];

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		szij, szkl, szp, 1.0, a, szp, b, szp, 0.0, c, szkl);
}

void tod_contract2_impl<2,2,1>::c_0123_012_012(double *c,
	const dimensions<4> &dc, const double *a, const dimensions<3> &da,
	const double *b, const dimensions<3> &db, double d) {

	// c_{ijkl} = c_{ijkl} + d \sum_{p} a_{ijp} b_{klp}

	size_t szp = da[2];
	size_t szij = da[0]*da[1];
	size_t szkl = db[0]*db[1];

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		szij, szkl, szp, d, a, szp, b, szp, 1.0, c, szkl);
}

} // namespace libtensor

