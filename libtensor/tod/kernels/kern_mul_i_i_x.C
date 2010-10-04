#include "../../linalg/linalg.h"
#include "kern_mul_i_i_x.h"

namespace libtensor {


const char *kern_mul_i_i_x::k_clazz = "kern_mul_i_i_x";


void kern_mul_i_i_x::run(const loop_registers<2, 1> &r) {

	linalg::i_i_x(m_ni, r.m_ptra[0], m_sia, r.m_ptra[1][0] * m_d,
		r.m_ptrb[0], m_sic);
}


kernel_base<2, 1> *kern_mul_i_i_x::match(const kern_mul_generic &z,
	list_t &in, list_t &out) {

	if(in.empty()) return 0;

	//	2. Minimize sic > 0:
	//	-------------
	//	w   a  b  c
	//	ni  1  0  sic  -->  c_i# = a_i b
	//	-------------       sz(i) = ni, sz(#) = k1a
	//	                    [i_i_x]

	iterator_t ii = in.end();
	size_t sic_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) == 1 && i->stepa(1) == 0 && i->stepb(0) > 0) {
			if(sic_min == 0 || sic_min > i->stepb(0)) {
				ii = i; sic_min = i->stepb(0);
			}
		}
	}
	if(ii == in.end()) return 0;

	kern_mul_i_i_x zz;
	zz.m_d = z.m_d;
	zz.m_ni = ii->weight();
	zz.m_sia = 1;
	zz.m_sic = ii->stepb(0);
	in.splice(out.begin(), out, ii);

	kernel_base<2, 1> *kern = 0;

	return new kern_mul_i_i_x(zz);
}


} // namespace libtensor
