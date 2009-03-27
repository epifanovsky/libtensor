#include "tod_contract2_impl_113.h"

namespace libtensor {

void tod_contract2_impl<1,1,3>::contract(
	double *c, const dimensions<2> &dc, const permutation<2> &pc,
	const double *a, const dimensions<4> &da, const permutation<4> &pa,
	const double *b, const dimensions<4> &db, const permutation<4> &pb)
	throw(exception) {

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
		throw_exc("tod_contract2_impl<1,1,3>", "contract()",
			"Contraction not implemented");
	}
	}
	}
}

void tod_contract2_impl<1,1,3>::contract(
	double *c, const dimensions<2> &dc, const permutation<2> &pc,
	const double *a, const dimensions<4> &da, const permutation<4> &pa,
	const double *b, const dimensions<4> &db, const permutation<4> &pb,
	double d) throw(exception) {

	// ijkl[0123] -> kijl[2013]
	permutation<4> p_2013;
	p_2013.permute(0,2).permute(1,2);

	permutation<2> p_10;
	p_10.permute(0,1);

	if(pc.is_identity() && pa.is_identity() && pb.is_identity()) {
		c_01_0123_0123(c, dc, a, da, b, db, d);
	} else {
	if(pc.equals(p_10) && pa.is_identity() && pb.is_identity()) {
		c_01_0123_0123(c, dc, b, db, a, da, d);
	} else {
	if(pc.is_identity() && pa.equals(p_2013) && pb.equals(p_2013)) {
		c_01_2013_2013(c, dc, a, da, b, db, d);
	} else {
		throw_exc("tod_contract2_impl<1,1,3>", "contract()",
			"Contraction not implemented");
	}
	}
	}
}

void tod_contract2_impl<1,1,3>::c_01_0123_0123(double *c,
	const dimensions<2> &dc, const double *a, const dimensions<4> &da,
	const double *b, const dimensions<4> &db) {

	// c_{ij} = \sum_{pqr} a_{ipqr} b_{jpqr}

	size_t szi = dc[0], szj = dc[1], szklm = da.get_increment(0);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		szi, szj, szklm, 1.0, a, szklm, b, szklm, 0.0, c, szj);
}

void tod_contract2_impl<1,1,3>::c_01_0123_0123(double *c,
	const dimensions<2> &dc, const double *a, const dimensions<4> &da,
	const double *b, const dimensions<4> &db, double d) {

	// c_{ij} = c_{ij} + d \sum_{pqr} a_{ipqr} b_{jpqr}

	size_t szi = dc[0], szj = dc[1], szklm = da.get_increment(0);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		szi, szj, szklm, d, a, szklm, b, szklm, 1.0, c, szj);
}

void tod_contract2_impl<1,1,3>::c_01_2013_2013(double *c,
	const dimensions<2> &dc, const double *a, const dimensions<4> &da,
	const double *b, const dimensions<4> &db) {

	// c_{ij} = \sum_{pqr} a_{pqir} b_{pqjr}

	size_t szpq = da[0]*da[1];
	size_t szir = da.get_increment(1);
	size_t szjr = db.get_increment(1);
	size_t szij = dc.get_size();

	size_t szi = dc[0], szj = dc[1], szr = da[3];
	const double *pa = a, *pb = b;
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		szi, szj, szr, 1.0, pa, szr, pb, szr, 0.0, c, szj);
	for(size_t pq=1; pq<szpq; pq++) {
		pa += szir; pb += szjr;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
			szi, szj, szr, 1.0, pa, szr, pb, szr, 1.0, c, szj);
	}
}

void tod_contract2_impl<1,1,3>::c_01_2013_2013(double *c,
	const dimensions<2> &dc, const double *a, const dimensions<4> &da,
	const double *b, const dimensions<4> &db, double d) {

	// c_{ij} = \sum_{pqr} a_{pqir} b_{pqjr}

	size_t szpq = da[0]*da[1];
	size_t szir = da.get_increment(1);
	size_t szjr = db.get_increment(1);
	size_t szij = dc.get_size();

	size_t szi = dc[0], szj = dc[1], szr = da[3];
	const double *pa = a, *pb = b;
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		szi, szj, szr, d, pa, szr, pb, szr, 1.0, c, szj);
	for(size_t pq=1; pq<szpq; pq++) {
		pa += szir; pb += szjr;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
			szi, szj, szr, d, pa, szr, pb, szr, 1.0, c, szj);
	}
}

} // namespace libtensor

