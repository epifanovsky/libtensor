#include "../../linalg/linalg.h"
#include "kern_mul_ijkl_pkiq_pjql.h"

namespace libtensor {


const char *kern_mul_ijkl_pkiq_pjql::k_clazz = "kern_mul_ijkl_pkiq_pjql";


void kern_mul_ijkl_pkiq_pjql::run(const loop_registers<2, 1> &r) {

	if(m_sia == m_nq && m_sqb == m_nl && m_skc == m_nl &&
		m_ska == m_sia * m_ni && m_sjb == m_sqb * m_nq &&
		m_sjc == m_skc * m_nk && m_spa == m_ska * m_nk &&
		m_spb == m_sjb * m_nj && m_sic == m_sjc * m_nj) {

		linalg::ijkl_pkiq_pjql_x(m_ni, m_nj, m_nk, m_nl, m_np, m_nq,
			r.m_ptra[0], r.m_ptra[1], r.m_ptrb[0], m_d);
		return;
	}

	const double *pa = r.m_ptra[0], *pb = r.m_ptra[1];
	double *pc = r.m_ptrb[0];

	for(size_t p = 0; p < m_np; p++) {
		const double *pb1 = pb;
		double *pc1 = pc;
		for(size_t j = 0; j < m_nj; j++) {
			const double *pa2 = pa, *pb2 = pb1;
			double *pc2 = pc1;
			for(size_t k = 0; k < m_nk; k++) {
				linalg::ij_ip_pj_x(m_ni, m_nl, m_nq, pa2, m_sia,
					pb2, m_sqb, pc2, m_sic, m_d);
				pa2 += m_ska;
				pc2 += m_skc;
			}
			pb1 += m_sjb;
			pc1 += m_sjc;
		}
		pa += m_spa;
		pb += m_spb;
	}
}


kernel_base<2, 1> *kern_mul_ijkl_pkiq_pjql::match(
	const kern_mul_ijk_piq_pjqk &z, list_t &in, list_t &out) {

	if(in.empty()) return 0;

	//	Rename k -> l.

	//	Minimize skc > 0.
	//	-----------------
	//	w   a    b    c
	//	nl  0    1    1
	//	nq  1    sqb  0
	//	ni  sia  0    sic
	//	nj  0    sjb  sjc
	//	np  spa  spb  0
	//	nk  ska  0    skc  --> c_i#j#k#l = a_p#k#i#q b_p#j#q#l
	//	-----------------      [ijkl_pkiq_pjql]
	//

	iterator_t ik = in.end();
	size_t skc_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) > 0 && i->stepa(1) == 0 && i->stepb(0) > 0) {
			if(i->stepa(0) % (z.m_sia * z.m_ni) ||
				z.m_spa % (i->weight() * i->stepa(0)))
				continue;
			if(i->stepb(0) % z.m_nk ||
				z.m_sjc % (i->weight() * i->stepb(0)))
				continue;
			if(skc_min == 0 || skc_min > i->stepb(0)) {
				ik = i; skc_min = i->stepb(0);
			}
		}
	}
	if(ik == in.end()) return 0;

	kern_mul_ijkl_pkiq_pjql zz;
	zz.m_d = z.m_d;
	zz.m_ni = z.m_ni;
	zz.m_nj = z.m_nj;
	zz.m_nk = ik->weight();
	zz.m_nl = z.m_nk;
	zz.m_np = z.m_np;
	zz.m_nq = z.m_nq;
	zz.m_spa = z.m_spa;
	zz.m_ska = ik->stepa(0);
	zz.m_sia = z.m_sia;
	zz.m_spb = z.m_spb;
	zz.m_sjb = z.m_sjb;
	zz.m_sqb = z.m_sqb;
	zz.m_sic = z.m_sic;
	zz.m_sjc = z.m_sjc;
	zz.m_skc = ik->stepb(0);
	in.splice(out.begin(), out, ik);

	kernel_base<2, 1> *kern = 0;

	return new kern_mul_ijkl_pkiq_pjql(zz);
}


} // namespace libtensor
