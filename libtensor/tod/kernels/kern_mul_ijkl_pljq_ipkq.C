#include "../../linalg/linalg.h"
#include "kern_mul_ijkl_pljq_ipkq.h"

namespace libtensor {


const char *kern_mul_ijkl_pljq_ipkq::k_clazz = "kern_mul_ijkl_pljq_ipkq";


void kern_mul_ijkl_pljq_ipkq::run(const loop_registers<2, 1> &r) {

	if(m_sja == m_nq && m_skb == m_nq && m_skc == m_nl &&
		m_sla == m_sja * m_nj && m_spb == m_skb * m_nk &&
		m_sjc == m_skc * m_nk && m_spa == m_sla * m_nl &&
		m_sib == m_spb * m_np && m_sic == m_sjc * m_nj) {

	}

	const double *pa = r.m_ptra[0], *pb = r.m_ptra[1];
	double *pc = r.m_ptrb[0];
	for(size_t i = 0; i < m_ni; i++) {
		const double *pa1 = pa, *pb1 = pb;
		double *pc1 = pc;
		for(size_t j = 0; j < m_nj; j++) {
			const double *pa2 = pa1, *pb2 = pb1;
			for(size_t p = 0; p < m_np; p++) {
				linalg::ij_ip_jp_x(m_nk, m_nl, m_nq, pb2, m_skb,
					pa2, m_sla, pc1, m_skc, m_d);
				pa2 += m_spa;
				pb2 += m_spb;
			}
			pa1 += m_sja;
			pc1 += m_sjc;
		}
		pb += m_sib;
		pc += m_sic;
	}
}


kernel_base<2, 1> *kern_mul_ijkl_pljq_ipkq::match(
	const kern_mul_ijk_pkiq_pjq &z, list_t &in, list_t &out) {

	if(in.empty()) return 0;

	//	Rename i -> j, j -> k, k -> l.

	//	Minimize sjc > 0:
	//	-----------------
	//	w   a    b    c
	//	nq  1    1    0
	//	nl  sla  0    1
	//	nk  0    skb  skc
	//	np  spa  spb  0
	//	nj  sja  0    sjc
	//	ni  0    sib  sic  -->  c_i#j#k#l = a_p#l#j#q b_i#p#k#q
	//	-----------------       [ijkl_pljq_ipkq]

	iterator_t ii = in.end();
	size_t sic_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepb(0) > 0) {
			if(i->stepa(1) % z.m_spb) continue;
			if(i->stepb(0) % z.m_sic) continue;
			if(sic_min == 0 || sic_min > i->stepb(0)) {
				ii = i; sic_min = i->stepb(0);
			}
		}
	}
	if(ii == in.end()) return 0;

	kern_mul_ijkl_pljq_ipkq zz;
	zz.m_d = z.m_d;
	zz.m_ni = ii->weight();
	zz.m_nj = z.m_ni;
	zz.m_nk = z.m_nj;
	zz.m_nl = z.m_nk;
	zz.m_np = z.m_np;
	zz.m_nq = z.m_nq;
	zz.m_spa = z.m_spa;
	zz.m_sla = z.m_ska;
	zz.m_sja = z.m_sia;
	zz.m_sib = ii->stepa(1);
	zz.m_spb = z.m_spb;
	zz.m_skb = z.m_sjb;
	zz.m_sic = ii->stepb(0);
	zz.m_sjc = z.m_sic;
	zz.m_skc = z.m_sjc;
	in.splice(out.begin(), out, ii);

	kernel_base<2, 1> *kern = 0;

	return new kern_mul_ijkl_pljq_ipkq(zz);
}


} // namespace libtensor
