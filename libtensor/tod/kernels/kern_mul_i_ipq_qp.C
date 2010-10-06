#include "../../linalg/linalg.h"
#include "kern_mul_i_ipq_qp.h"
#include "kern_mul_ij_jpq_iqp.h"

namespace libtensor {


const char *kern_mul_i_ipq_qp::k_clazz = "kern_mul_i_ipq_qp";


void kern_mul_i_ipq_qp::run(const loop_registers<2, 1> &r) {

	linalg::i_ipq_qp_x(m_ni, m_np, m_nq, r.m_ptra[0], m_spa, m_sia,
		r.m_ptra[1], m_sqb, r.m_ptrb[0], m_sic, m_d);
}


kernel_base<2, 1> *kern_mul_i_ipq_qp::match(const kern_mul_x_pq_qp &z,
	list_t &in, list_t &out) {

	if(in.empty()) return 0;

	//	Minimize sia > 0:
	//	-----------------
	//	w   a    b    c
	//	np  spa  1    0
	//	nq  1    spb  0
	//	ni  sia  0    sic  -->  c_i# = a_i#p#q b_q#p
	//	-----------------       [i_ipq_qp]
	//

	iterator_t ii = in.end();
	size_t sia_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) > 0 && i->stepa(1) == 0 && i->stepb(0) > 0) {
			if(i->stepa(0) % z.m_spa) continue;
			if(sia_min == 0 || sia_min > i->stepa(0)) {
				ii = i; sia_min = i->stepa(0);
			}
		}
	}
	if(ii == in.end()) return 0;

	kern_mul_i_ipq_qp zz;
	zz.m_d = z.m_d;
	zz.m_ni = ii->weight();
	zz.m_np = z.m_np;
	zz.m_nq = z.m_nq;
	zz.m_sia = ii->stepa(0);
	zz.m_spa = z.m_spa;
	zz.m_sqb = z.m_sqb;
	zz.m_sic = ii->stepb(0);
	in.splice(out.begin(), out, ii);

	kernel_base<2, 1> *kern = 0;

	if(kern = kern_mul_ij_jpq_iqp::match(zz, in, out)) return kern;

	return new kern_mul_i_ipq_qp(zz);
}


} // namespace libtensor
