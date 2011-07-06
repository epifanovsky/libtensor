#include "../../linalg/linalg.h"
#include "kern_mul_ijk_pkj_pi.h"

namespace libtensor {


const char *kern_mul_ijk_pkj_pi::k_clazz = "kern_mul_ijk_pkj_pi";


void kern_mul_ijk_pkj_pi::run(const loop_registers<2, 1> &r) {

	linalg::ijk_pi_pkj_x(m_ni, m_nj, m_nk, m_np, r.m_ptra[1], m_spb,
		r.m_ptra[0], m_ska, m_spa, r.m_ptrb[0], m_sjc, m_sic, m_d);
}


kernel_base<2, 1> *kern_mul_ijk_pkj_pi::match(const kern_mul_ij_pji_p &z,
	list_t &in, list_t &out) {

	if(in.empty()) return 0;

	//	Rename i -> j, j -> k.

	//	Minimize sic > 0:
	//	-------------------
	//	w   a    b     c
	//	nj  1    0     sjc
	//	np  spa  spb   0
	//	nk  ska  0     1
	//	ni  0    1     sic  -->  c_i#j#k = a_p#k#j b_p#i
	//	-------------------      [ijk_pkj_pi]
	//

	iterator_t ii = in.end();
	size_t sic_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) == 0 && i->stepa(1) == 1 && i->stepb(0) > 0) {
			if(z.m_spa % (i->weight() * i->stepa(1))) continue;
			if(i->stepb(0) % (z.m_ni * z.m_sic)) continue;
			if(sic_min == 0 || sic_min > i->stepb(0)) {
				ii = i; sic_min = i->stepb(0);
			}
		}
	}
	if(ii == in.end()) return 0;

	kern_mul_ijk_pkj_pi zz;
	zz.m_d = z.m_d;
	zz.m_ni = ii->weight();
	zz.m_nj = z.m_ni;
	zz.m_nk = z.m_nj;
	zz.m_np = z.m_np;
	zz.m_ska = z.m_sja;
	zz.m_spa = z.m_spa;
	zz.m_spb = z.m_spb;
	zz.m_sic = ii->stepb(0);
	zz.m_sjc = z.m_sic;
	in.splice(out.begin(), out, ii);

	kernel_base<2, 1> *kern = 0;

	return new kern_mul_ijk_pkj_pi(zz);
}


} // namespace libtensor
