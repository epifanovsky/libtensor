#include "../../linalg/linalg.h"
#include "kern_mul_ijk_kp_jpi.h"

namespace libtensor {


const char *kern_mul_ijk_kp_jpi::k_clazz = "kern_mul_ijk_kp_jpi";


void kern_mul_ijk_kp_jpi::run(const loop_registers<2, 1> &r) {

	const double *pb = r.m_ptra[1];
	double *pc = r.m_ptrb[0];

	for(size_t j = 0; j < m_nj; j++) {
		linalg::ij_pi_jp_x(m_ni, m_nk, m_np, pb, m_spb, r.m_ptra[0],
			m_ska, pc, m_sic, m_d);
		pb += m_sjb;
		pc += m_sjc;
	}
}


kernel_base<2, 1> *kern_mul_ijk_kp_jpi::match(const kern_mul_ij_jp_pi &z,
	list_t &in, list_t &out) {

	if(in.empty()) return 0;

	//	Rename j->k

	//	1. Minimize sjc > 0:
	//	-----------------
	//	w   a    b    c
	//	ni  0    1    sic
	//	np  1    spb  0
	//	nk  ska  0    1
	//	nj  0    sjb  sjc  -->  c_i#j#k = a_k#p b_j#p#i
	//	-----------------       [ijk_kp_jpi]
	//

	iterator_t ij = in.end();
	size_t sjc_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepb(0) > 0) {
			if(i->stepa(1) % z.m_spb) continue;
			if(i->stepb(0) % z.m_nj) continue;
			if(z.m_sic % i->weight()) continue;
			if(sjc_min == 0 || sjc_min > i->stepb(0)) {
				ij = i; sjc_min = i->stepb(0);
			}
		}
	}
	if(ij == in.end()) return 0;

	kern_mul_ijk_kp_jpi zz;
	zz.m_d = z.m_d;
	zz.m_ni = z.m_ni;
	zz.m_nj = ij->weight();
	zz.m_nk = z.m_nj;
	zz.m_np = z.m_np;
	zz.m_ska = z.m_sja;
	zz.m_spb = z.m_spb;
	zz.m_sjb = ij->stepa(1);
	zz.m_sjc = ij->stepb(0);
	zz.m_sic = z.m_sic;
	in.splice(out.begin(), out, ij);

	kernel_base<2, 1> *kern = 0;

	return new kern_mul_ijk_kp_jpi(zz);
}


} // namespace libtensor
