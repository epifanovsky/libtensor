#include "../../linalg/linalg.h"
#include "kern_mul_x_pq_qp.h"
#include "kern_mul_i_ipq_qp.h"

namespace libtensor {


const char *kern_mul_x_pq_qp::k_clazz = "kern_mul_x_pq_qp";


void kern_mul_x_pq_qp::run(const loop_registers<2, 1> &r) {

	r.m_ptrb[0][0] += linalg::x_pq_qp(m_np, m_nq, r.m_ptra[0], m_spa,
		r.m_ptra[1], m_sqb) * m_d;
}


kernel_base<2, 1> *kern_mul_x_pq_qp::match(const kern_mul_x_p_p &z,
	list_t &in, list_t &out) {

	if(in.empty()) return 0;

	//	Minimize spb > 0:
	//	---------------
	//	w   a    b    c
	//	np  spa  1    0
	//	nq  1    spb  0  -->  c_# = a_p#q b_q#p
	//	---------------       [x_pq_qp]
	//

	iterator_t iq = in.end();
	size_t spb_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) == 1 && i->stepa(1) > 0 && i->stepb(0) == 0) {
			if(i->stepa(1) % z.m_np) continue;
			if(spb_min == 0 || spb_min > i->stepa(1)) {
				iq = i; spb_min = i->stepa(1);
			}
		}
	}
	if(iq == in.end()) return 0;

	kern_mul_x_pq_qp zz;
	zz.m_d = z.m_d;
	zz.m_np = z.m_np;
	zz.m_nq = iq->weight();
	zz.m_spa = z.m_spa;
	zz.m_sqb = iq->stepa(1);
	in.splice(out.begin(), out, iq);

	kernel_base<2, 1> *kern = 0;

	if(kern = kern_mul_i_ipq_qp::match(zz, in, out)) return kern;

	return new kern_mul_x_pq_qp(zz);
}


} // namespace libtensor
