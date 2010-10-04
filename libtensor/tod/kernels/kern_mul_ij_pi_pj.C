#include "../../linalg/linalg.h"
#include "kern_mul_ij_pi_pj.h"

namespace libtensor {


const char *kern_mul_ij_pi_pj::k_clazz = "kern_mul_ij_pi_pj";


void kern_mul_ij_pi_pj::run(const loop_registers<2, 1> &r) {

	linalg::ij_pi_pj_x(m_ni, m_nj, m_np, r.m_ptra[0], m_spa, r.m_ptra[1],
		m_spb, r.m_ptrb[0], m_sic, m_d);
}


kernel_base<2, 1> *kern_mul_ij_pi_pj::match(const kern_mul_i_p_pi &z,
	list_t &in, list_t &out) {

	if(in.empty()) return 0;
	if(z.m_sic != 1) return 0;

	//	Rename i -> j

	//	1. Minimize sic > 0:
	//	-----------------
	//	w   a    b    c
	//	nj  0    1    1
	//	np  spa  spb  0
	//	ni  1    0    sic  -->  c_j#i = a_p$j b_p%i
	//	-----------------       sz(i) = w1, sz(j) = w3,
	//	                        sz(p) = w2
	//	                        sz(#) = k4, sz($) = k3',
	//	                        sz(%) = k2
	//	                        [ij_pi_pj]

	iterator_t ii = in.end();
	size_t sic_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) == 1 && i->stepa(1) == 0 && i->stepb(0) > 0) {
			if(i->stepb(0) % z.m_ni) continue;
			if(z.m_spa % i->weight()) continue;
			if(sic_min == 0 || sic_min > i->stepb(0)) {
				ii = i; sic_min = i->stepb(0);
			}
		}
	}
	if(ii == in.end()) return 0;

	kern_mul_ij_pi_pj zz;
	zz.m_d = z.m_d;
	zz.m_ni = ii->weight();
	zz.m_nj = z.m_ni;
	zz.m_np = z.m_np;
	zz.m_spa = z.m_spa;
	zz.m_spb = z.m_spb;
	zz.m_sic = ii->stepb(0);
	in.splice(out.begin(), out, ii);

	kernel_base<2, 1> *kern = 0;

	return new kern_mul_ij_pi_pj(zz);
}


} // namespace libtensor
