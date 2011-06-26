#include "../../linalg/linalg.h"
#include "kern_add_i_i_x_x.h"

namespace libtensor {


const char *kern_add_i_i_x_x::k_clazz = "kern_add_i_i_x_x";


void kern_add_i_i_x_x::run(const loop_registers<2, 1> &r) {

	linalg::add_i_i_x_x(m_ni, r.m_ptra[0], m_sia, r.m_ptra[1][0],
		r.m_ptrb[0], m_sic, m_d);
}


kernel_base<2, 1> *kern_add_i_i_x_x::match(const kern_add_generic &z,
	list_t &in, list_t &out) {

	if(in.empty()) return 0;

	//	Minimize sic > 0:
	//	-------------
	//	w   a  b  c
	//	ni  1  0  sic  -->  c_i# = (a_i + b) d
	//	-------------       [i_i_x_x]
	//

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

	kern_add_i_i_x_x zz;
	zz.m_d = z.m_d;
	zz.m_ni = ii->weight();
	zz.m_sia = 1;
	zz.m_sic = ii->stepb(0);
	in.splice(out.begin(), out, ii);

	kernel_base<2, 1> *kern = 0;

	return new kern_add_i_i_x_x(zz);
}


} // namespace libtensor
