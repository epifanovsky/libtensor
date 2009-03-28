#include "tod_contract2_impl_131.h"

namespace libtensor {

void tod_contract2_impl<1,3,1>::contract(double *c, const dimensions<4> &dc,
	const permutation<4> &pc, const double *a, const dimensions<2> &da,
	const permutation<2> &pa, const double *b, const dimensions<4> &db,
	const permutation<4> &pb) throw(exception) {

	// jpkl[0123] -> jklp[0231]
	permutation<4> p_0231;
	p_0231.permute(1,3).permute(1,2);

	permutation<4> p_1023;
	p_1023.permute(0,1);

	permutation<2> p_10;
	p_10.permute(0,1);

	if(pc.equals(p_1023) && pa.equals(p_10) && pb.equals(p_0231)) {
		c_1023_10_0231(c, dc, a, da, b, db);
	} else {
		throw_exc("tod_contract2_impl<1,3,1>", "contract()",
			"Contraction not implemented");
	}
}

void tod_contract2_impl<1,3,1>::contract(double *c, const dimensions<4> &dc,
	const permutation<4> &pc, const double *a, const dimensions<2> &da,
	const permutation<2> &pa, const double *b, const dimensions<4> &db,
	const permutation<4> &pb, double d) throw(exception) {

	// jpkl[0123] -> jklp[0231]
	permutation<4> p_0231;
	p_0231.permute(1,3).permute(1,2);

	permutation<4> p_1023;
	p_1023.permute(0,1);

	permutation<2> p_10;
	p_10.permute(0,1);

	if(pc.equals(p_1023) && pa.equals(p_10) && pb.equals(p_0231)) {
		c_1023_10_0231(c, dc, a, da, b, db, d);
	} else {
		throw_exc("tod_contract2_impl<1,3,1>", "contract()",
			"Contraction not implemented");
	}
}

void tod_contract2_impl<1,3,1>::c_1023_10_0231(double *c,
	const dimensions<4> &dc, const double *a, const dimensions<2> &da,
	const double *b, const dimensions<4> &db) {

	// c_{jikl} = \sum_{p} a_{pi} b_{jpkl}

	size_t szj = db[0], szm = da[0], szi = da[1];
	size_t szmkl = db.get_increment(0);
	size_t szkl = db.get_increment(1);
	size_t szikl = szi*szkl;

	const double *pb = b;
	double *pc = c;
	for(size_t j=0; j<szj; j++) {
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
			szi, szkl, szm, 1.0, a, szi, pb, szkl, 0.0, pc, szkl);
		pb += szmkl; pc += szikl;
	}
}

void tod_contract2_impl<1,3,1>::c_1023_10_0231(double *c,
	const dimensions<4> &dc, const double *a, const dimensions<2> &da,
	const double *b, const dimensions<4> &db, double d) {

	// c_{jikl} = c_{jikl} + d \sum_{p} a_{pi} b_{jpkl}

	size_t szj = db[0], szm = da[0], szi = da[1];
	size_t szmkl = db.get_increment(0);
	size_t szkl = db.get_increment(1);
	size_t szikl = szi*szkl;

	const double *pb = b;
	double *pc = c;
	for(size_t j=0; j<szj; j++) {
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
			szi, szkl, szm, d, a, szi, pb, szkl, 1.0, pc, szkl);
		pb += szmkl; pc += szikl;
	}
}

} // namespace libtensor

