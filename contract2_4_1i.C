#include "contract2_4_1i.h"

namespace libtensor {

void contract2_4_1i::contract(double *c, const dimensions<4> &dc,
	const permutation<4> &pc, const double *a, const dimensions<2> &da,
	const permutation<2> &pa, const double *b, const dimensions<4> &db,
	const permutation<4> &pb) throw(exception) {

#ifdef LIBTENSOR_DEBUG
	dimensions<2> da1(da);
	dimensions<4> db1(db), dc1(dc);
	da1.permute(pa); db1.permute(pb); dc1.permute(pc);
	if(dc1[0]!=da1[0]) {
		throw_exc("contract2_4_1i", "contract()",
                        "Inconsistent dimension: i");
	}
	if(dc1[1]!=db1[0]) {
		throw_exc("contract2_4_1i", "contract()",
                        "Inconsistent dimension: j");
	}
	if(dc1[2]!=db1[1]) {
		throw_exc("contract2_4_1i", "contract()",
                        "Inconsistent dimension: k");
	}
	if(dc1[3]!=db1[2]) {
		throw_exc("contract2_4_1i", "contract()",
                        "Inconsistent dimension: l");
	}
	if(da1[1]!=db1[3]) {
		throw_exc("contract2_4_1i", "contract()",
                        "Inconsistent dimension: m");
	}
#endif // LIBTENSOR_DEBUG

	// jmkl[0123] -> jklm[0231]
	permutation<4> p_0231;
	p_0231.permute(1,3).permute(1,2);

	permutation<4> p_1023;
	p_1023.permute(0,1);

	permutation<2> p_10;
	p_10.permute(0,1);

	if(pc.equals(p_1023) && pa.equals(p_10) && pb.equals(p_0231)) {
		c_1023_10_0231(c, dc, a, da, b, db);
	} else {
		throw_exc("contract2_4_1i", "contract()",
			"Contraction not implemented");
	}
}

void contract2_4_1i::c_1023_10_0231(double *c, const dimensions<4> &dc,
	const double *a, const dimensions<2> &da, const double *b,
	const dimensions<4> &db) {

	// c_{jikl} = \sum_{m} a_{mi} b_{jmkl}

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

} // namespace libtensor

