#include "../../linalg/linalg.h"
#include "kern_mul_ijkl_pkiq_pjlq.h"

namespace libtensor {


const char *kern_mul_ijkl_pkiq_pjlq::k_clazz = "kern_mul_ijkl_pkiq_pjlq";


void kern_mul_ijkl_pkiq_pjlq::run(const loop_registers<2, 1> &r) {

	if(m_sia == m_nq && m_slb == m_nq && m_skc == m_nl &&
		m_ska == m_sia * m_ni && m_sjb == m_slb * m_nl &&
		m_sjc == m_skc * m_nk && m_spa == m_ska * m_nk &&
		m_spb == m_sjb * m_nj && m_sic == m_sjc * m_nj) {

		linalg::ijkl_pkiq_pjlq_x(m_ni, m_nj, m_nk, m_nl, m_np, m_nq,
			r.m_ptra[0], r.m_ptra[1], r.m_ptrb[0], m_d);
		return;
	}

	const double *pa = r.m_ptra[0], *pb = r.m_ptra[1];
	double *pc = r.m_ptrb[0];
	for(size_t i = 0; i < m_ni; i++) {
		const double *pa1 = pa, *pb1 = pb;
		double *pc1 = pc;
		for(size_t j = 0; j < m_nj; j++) {
			const double *pa2 = pa1, *pb2 = pb1;
			for(size_t p = 0; p < m_np; p++) {
				linalg::ij_ip_jp_x(m_nk, m_nl, m_nq, pa2, m_ska,
					pb2, m_slb, pc1, m_skc, m_d);
				pa2 += m_spa;
				pb2 += m_spb;
			}
			pb1 += m_sjb;
			pc1 += m_sjc;
		}
		pa += m_sia;
		pc += m_sic;
	}
}


kernel_base<2, 1> *kern_mul_ijkl_pkiq_pjlq::match(
	const kern_mul_ijk_piq_pjkq &z, list_t &in, list_t &out) {

	if(in.empty()) return 0;

	//	Rename k -> l.

	//	Minimize ska > 0:
	//	-----------------
	//	w   a    b    c
	//	nq  1    1    0
	//	nl  0    slb  1
	//	ni  sia  0    sic
	//	np  spa  spb  0
	//	nj  0    sjb  sjc
	//	nk  ska  0    skc  -->  c_i#j#k#l = a_p#k#i#q b_p#j#l#q
	//	-----------------       [ijkl_pkiq_pjlq]
	//

	iterator_t ik = in.end();
	size_t ska_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) > 0 && i->stepa(1) == 0 && i->stepb(0) > 0) {
			if(i->stepa(0) % (z.m_sia * z.m_ni) ||
				z.m_spa % (i->weight() * i->stepa(0)))
				continue;
			if(i->stepb(0) % z.m_nk ||
				z.m_sjc % (i->weight() * i->stepb(0)))
				continue;
			if(ska_min == 0 || ska_min > i->stepa(0)) {
				ik = i; ska_min = i->stepa(0);
			}
		}
	}
	if(ik == in.end()) return 0;

	kern_mul_ijkl_pkiq_pjlq zz;
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
	zz.m_slb = z.m_skb;
	zz.m_sic = z.m_sic;
	zz.m_sjc = z.m_sjc;
	zz.m_skc = ik->stepb(0);
	in.splice(out.begin(), out, ik);

	kernel_base<2, 1> *kern = 0;

	return new kern_mul_ijkl_pkiq_pjlq(zz);
}


} // namespace libtensor
