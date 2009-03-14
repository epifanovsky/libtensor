#include "contract2_2_3i.h"

namespace libtensor {

void contract2_2_3i::contract(
	double *c, const dimensions &dc, const permutation &pc,
	const double *a, const dimensions &da, const permutation &pa,
	const double *b, const dimensions &db, const permutation &pb)
	throw(exception) {

#ifdef LIBTENSOR_DEBUG
	dimensions da1(da), db1(db), dc1(dc);
	da1.permute(pa); db1.permute(pb); dc1.permute(pc);
	if(dc1[0]!=da1[0]) {
		throw_exc("contract2_2_3i", "contract(...)",
                        "Inconsistent dimension: i");
	}
	if(dc1[1]!=db1[0]) {
		throw_exc("contract2_2_3i", "contract(...)",
                        "Inconsistent dimension: j");
	}
	if(da1[1]!=db1[1]) {
		throw_exc("contract2_2_3i", "contract(...)",
                        "Inconsistent dimension: k");
	}
	if(da1[2]!=db1[2]) {
		throw_exc("contract2_2_3i", "contract(...)",
                        "Inconsistent dimension: l");
	}
	if(da1[3]!=db1[3]) {
		throw_exc("contract2_2_3i", "contract(...)",
                        "Inconsistent dimension: m");
	}
#endif // LIBTENSOR_DEBUG

	permutation p_1203(4);
	p_1203.permute(0,2).permute(0,1);
	if(pc.is_identity() && pa.equals(p_1203) && pb.equals(p_1203)) {
		c_01_1203_1203(c, dc, a, da, b, db);
	} else {
		throw_exc("contract2_2_3i", "contract(...)",
			"Contraction not implemented");
	}
}

void contract2_2_3i::c_01_1203_1203(double *c, const dimensions &dc,
	const double *a, const dimensions &da, const double *b,
	const dimensions &db) {

	size_t szkl = da[0]*da[1];
	size_t szim = da.get_increment(1);
	size_t szjm = db.get_increment(1);
	size_t szij = dc.get_size();

	for(size_t ij=0; ij<szij; ij++) c[ij] = 0.0;

	const double *pa = a, *pb = b;
	for(size_t kl=0; kl<szkl; kl++) {
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
			dc[0], dc[1], da[3], 1.0,
			a, da[2], b, db[2], 1.0, c, dc[0]);
	}
}

} // namespace libtensor

