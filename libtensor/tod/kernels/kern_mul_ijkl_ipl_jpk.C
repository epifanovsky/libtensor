#include "../../linalg/linalg.h"
#include "kern_mul_ijkl_ipl_jpk.h"

namespace libtensor {


const char *kern_mul_ijkl_ipl_jpk::k_clazz = "kern_mul_ijkl_ipl_jpk";


void kern_mul_ijkl_ipl_jpk::run(const loop_registers<2, 1> &r) {

	if(m_skc == m_nl && m_sjc == m_skc * m_nk && m_sic == m_sjc * m_nj) {

		linalg::ijkl_ipl_jpk_x(m_ni, m_nj, m_nk, m_nl, m_np, r.m_ptra[0],
			m_spa, m_sia, r.m_ptra[1], m_spb, m_sjb, r.m_ptrb[0],
			m_d);
		return;
	}

	const double *pa = r.m_ptra[0];
	double *pc = r.m_ptrb[0];

	for(size_t i = 0; i < m_ni; i++) {
		for(size_t j = 0; j < m_nj; j++) {
			linalg::ij_pi_pj_x(m_nk, m_nl, m_np,
				r.m_ptra[1] + j * m_sjb, m_spb,
				pa, m_spa, pc + j * m_sjc, m_skc, m_d);
		}
		pa += m_sia;
		pc += m_sic;
	}
}


kernel_base<2, 1> *kern_mul_ijkl_ipl_jpk::match(const kern_mul_ijk_ipk_pj &z,
	list_t &in, list_t &out) {

	if(in.empty()) return 0;

	//	Rename j -> k, k -> l

	//	Minimize sjb > 0:
	//	------------------
	//	w   a    b    c
	//	nl  1    0    1
	//	np  spa  spb  0
	//	nk  0    1    skc
	//	ni  sia  0    sic
	//	nj  0    sjb  sjc  --> c_i#j#k#l = a_i#p#l b_j#p#k
	//	-----------------      [ijkl_ipl_jpk]
	//

	iterator_t ij = in.end();
	size_t sjb_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepb(0) > 0) {
			if(i->stepa(1) % (z.m_spb * z.m_np)) continue;
			if(i->stepb(0) % (z.m_sjc * z.m_nj)) continue;
			if(z.m_sic % i->weight()) continue;
			if(sjb_min == 0 || sjb_min > i->stepa(1)) {
				ij = i; sjb_min = i->stepa(1);
			}
		}
	}
	if(ij == in.end()) return 0;

	kern_mul_ijkl_ipl_jpk zz;
	zz.m_d = z.m_d;
	zz.m_ni = z.m_ni;
	zz.m_nj = ij->weight();
	zz.m_nk = z.m_nj;
	zz.m_nl = z.m_nk;
	zz.m_np = z.m_np;
	zz.m_sia = z.m_sia;
	zz.m_spa = z.m_spa;
	zz.m_sjb = ij->stepa(1);
	zz.m_spb = z.m_spb;
	zz.m_sic = z.m_sic;
	zz.m_sjc = ij->stepb(0);
	zz.m_skc = z.m_sjc;
	in.splice(out.begin(), out, ij);

	kernel_base<2, 1> *kern = 0;

	return new kern_mul_ijkl_ipl_jpk(zz);
}


} // namespace libtensor
