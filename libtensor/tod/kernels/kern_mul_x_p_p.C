#include "../../linalg/linalg.h"
#include "kern_mul_x_p_p.h"

namespace libtensor {


const char *kern_mul_x_p_p::k_clazz = "kern_mul_x_p_p";


void kern_mul_x_p_p::run(const loop_registers<2, 1> &r) {

	r.m_ptrb[0][0] += linalg::x_p_p(m_np, r.m_ptra[0], m_spa, r.m_ptra[1],
		m_spb) * m_d;
}


kernel_base<2, 1> *kern_mul_x_p_p::match(const kern_mul_generic &z,
	list_t &in, list_t &out) {

	if(in.empty()) return 0;

	//	1. Minimize k1 > 0:
	//	------------
	//	w   a   b  c
	//	np  k1  1  0  -->  c_# = a_p# b_p
	//	------------       sz(p) = np, sz(#) = k1
	//	                   [x_p_p]

	iterator_t ip = in.end();
	size_t k1_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) > 0 && i->stepa(1) == 1 && i->stepb(0) == 0) {
			if(k1_min == 0 || k1_min > i->stepa(0)) {
				ip = i; k1_min = i->stepa(0);
			}
		}
	}
	if(ip == in.end()) return 0;

	kern_mul_x_p_p zz;
	zz.m_d = z.m_d;
	zz.m_np = ip->weight();
	zz.m_spa = ip->stepa(0);
	zz.m_spb = 1;
	in.splice(out.begin(), out, ip);

	kernel_base<2, 1> *kern = 0;

	return new kern_mul_x_p_p(zz);
}


} // namespace libtensor
