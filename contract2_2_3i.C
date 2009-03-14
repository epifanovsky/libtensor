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

	// ijkl[0123] -> kijl[2013]
	permutation p_2013(4);
	p_2013.permute(0,2).permute(1,2);
	if(pc.is_identity() && pa.equals(p_2013) && pb.equals(p_2013)) {
		c_01_2013_2013(c, dc, a, da, b, db);
	} else {
		throw_exc("contract2_2_3i", "contract(...)",
			"Contraction not implemented");
	}
}

void contract2_2_3i::c_01_2013_2013(double *c, const dimensions &dc,
	const double *a, const dimensions &da, const double *b,
	const dimensions &db) {

	// c_ij = \sum_klm a_klim b_kljm

	size_t szkl = da[0]*da[1];
	size_t szim = da.get_increment(1);
	size_t szjm = db.get_increment(1);
	size_t szij = dc.get_size();

	for(size_t ij=0; ij<szij; ij++) c[ij] = 0.0;

	size_t szi = dc[0], szj = dc[1], szm = da[3];
	const double *pa = a, *pb = b;
	for(size_t kl=0; kl<szkl; kl++) {
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
			szi, szj, szm, 1.0,
			a, szi, b, szj, 1.0, c, szi);
	}
}

} // namespace libtensor

