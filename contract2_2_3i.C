#include "contract2_2_3i.h"

namespace libtensor {

void contract2_2_3i::contract(
	double *c, const dimensions<2> &dc, const permutation<2> &pc,
	const double *a, const dimensions<4> &da, const permutation<4> &pa,
	const double *b, const dimensions<4> &db, const permutation<4> &pb)
	throw(exception) {

#ifdef LIBTENSOR_DEBUG
	dimensions<4> da1(da), db1(db);
	dimensions<2> dc1(dc);
	da1.permute(pa); db1.permute(pb); dc1.permute(pc);
	if(dc1[0]!=da1[0]) {
		throw_exc("contract2_2_3i", "contract()",
                        "Inconsistent dimension: i");
	}
	if(dc1[1]!=db1[0]) {
		throw_exc("contract2_2_3i", "contract()",
                        "Inconsistent dimension: j");
	}
	if(da1[1]!=db1[1]) {
		throw_exc("contract2_2_3i", "contract()",
                        "Inconsistent dimension: k");
	}
	if(da1[2]!=db1[2]) {
		throw_exc("contract2_2_3i", "contract()",
                        "Inconsistent dimension: l");
	}
	if(da1[3]!=db1[3]) {
		throw_exc("contract2_2_3i", "contract()",
                        "Inconsistent dimension: m");
	}
#endif // LIBTENSOR_DEBUG

	// ijkl[0123] -> kijl[2013]
	permutation<4> p_2013;
	p_2013.permute(0,2).permute(1,2);

	permutation<2> p_10;
	p_10.permute(0,1);

	if(pc.is_identity() && pa.is_identity() && pb.is_identity()) {
		c_01_0123_0123(c, dc, a, da, b, db);
	} else {
	if(pc.equals(p_10) && pa.is_identity() && pb.is_identity()) {
		c_01_0123_0123(c, dc, b, db, a, da);
	} else {
	if(pc.is_identity() && pa.equals(p_2013) && pb.equals(p_2013)) {
		c_01_2013_2013(c, dc, a, da, b, db);
	} else {
		throw_exc("contract2_2_3i", "contract()",
			"Contraction not implemented");
	}
	}
	}
}

void contract2_2_3i::c_01_0123_0123(double *c, const dimensions<2> &dc,
	const double *a, const dimensions<4> &da, const double *b,
	const dimensions<4> &db) {

	// c_ij = \sum_klm a_iklm b_jklm

	size_t szi = dc[0], szj = dc[1], szklm = da.get_increment(0);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		szi, szj, szklm, 1.0, a, szklm, b, szklm, 0.0, c, szj);
}

void contract2_2_3i::c_01_2013_2013(double *c, const dimensions<2> &dc,
	const double *a, const dimensions<4> &da, const double *b,
	const dimensions<4> &db) {

	// c_ij = \sum_klm a_klim b_kljm

	size_t szkl = da[0]*da[1];
	size_t szim = da.get_increment(1);
	size_t szjm = db.get_increment(1);
	size_t szij = dc.get_size();

	size_t szi = dc[0], szj = dc[1], szm = da[3];
	const double *pa = a, *pb = b;
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		szi, szj, szm, 1.0, pa, szm, pb, szm, 0.0, c, szj);
	for(size_t kl=1; kl<szkl; kl++) {
		pa += szim; pb += szjm;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
			szi, szj, szm, 1.0, pa, szm, pb, szm, 1.0, c, szj);
	}
}

} // namespace libtensor

