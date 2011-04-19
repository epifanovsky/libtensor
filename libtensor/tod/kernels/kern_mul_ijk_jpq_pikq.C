#include "../../linalg/linalg.h"
#include "kern_mul_ijk_jpq_pikq.h"

namespace libtensor {


const char *kern_mul_ijk_jpq_pikq::k_clazz = "kern_mul_ijk_jpq_pikq";


void kern_mul_ijk_jpq_pikq::run(const loop_registers<2, 1> &r) {

	if(m_spa == m_nq && m_sja == m_spa * m_np && m_skb == m_nq &&
		m_sib == m_skb * m_nk && m_spb == m_sib * m_ni &&
		m_sjc == m_nk && m_sic == m_sjc * m_nj) {

		linalg::ijk_pikq_jpq_x(m_ni, m_nj, m_nk, m_np, m_nq,
			r.m_ptra[1], r.m_ptra[0], r.m_ptrb[0], m_d);
		return;
	}

	const double *pb = r.m_ptra[1];
	double *pc = r.m_ptrb[0];

	for(size_t i = 0; i < m_ni; i++) {
		const double *pa1 = r.m_ptra[0], *pb1 = pb;
		for(size_t p = 0; p < m_np; p++) {
			linalg::ij_ip_jp_x(m_nj, m_nk, m_nq, pa1, m_sja,
				pb1, m_skb, pc, m_sjc, m_d);
			pa1 += m_spa;
			pb1 += m_spb;
		}
		pb += m_sib;
		pc += m_sic;
	}
}


kernel_base<2, 1> *kern_mul_ijk_jpq_pikq::match(const kern_mul_ij_ipq_pjq &z,
	list_t &in, list_t &out) {

	if(in.empty()) return 0;

	//	Rename i -> j, j -> k.

	//	Minimize sic > 0:
	//	-----------------
	//	w   a    b    c
	//	nq  1    1    0
	//	nk  0    skb  1
	//	nj  sja  0    sjc
	//	np  spa  spb  0
	//	ni  0    sib  sic  -->  c_i#j#k = a_j#p#q b_p#i#k#q
	//	-----------------       [ijk_jpq_pikq]
	//

	iterator_t ii = in.end();
	size_t sic_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepb(0) > 0) {
			if(i->stepa(1) % (z.m_sjb * z.m_nj) ||
				z.m_spb % (i->weight() * i->stepa(1)))
				continue;
			if(i->stepb(0) % (z.m_sic * z.m_ni)) continue;
			if(sic_min == 0 || sic_min > i->stepb(0)) {
				ii = i; sic_min = i->stepb(0);
			}
		}
	}
	if(ii == in.end()) return 0;

	kern_mul_ijk_jpq_pikq zz;
	zz.m_d = z.m_d;
	zz.m_ni = ii->weight();
	zz.m_nj = z.m_ni;
	zz.m_nk = z.m_nj;
	zz.m_np = z.m_np;
	zz.m_nq = z.m_nq;
	zz.m_sja = z.m_sia;
	zz.m_spa = z.m_spa;
	zz.m_spb = z.m_spb;
	zz.m_sib = ii->stepa(1);
	zz.m_skb = z.m_sjb;
	zz.m_sic = ii->stepb(0);
	zz.m_sjc = z.m_sic;
	in.splice(out.begin(), out, ii);

	kernel_base<2, 1> *kern = 0;

	return new kern_mul_ijk_jpq_pikq(zz);
}


} // namespace libtensor
