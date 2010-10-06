#include "../../linalg/linalg.h"
#include "kern_mul_ij_jpq_iqp.h"

namespace libtensor {


const char *kern_mul_ij_jpq_iqp::k_clazz = "kern_mul_ij_jpq_iqp";


void kern_mul_ij_jpq_iqp::run(const loop_registers<2, 1> &r) {

	linalg::ij_ipq_jqp_x(m_ni, m_nj, m_nq, m_np, r.m_ptra[1], m_sqb, m_sib,
		r.m_ptra[0], m_spa, m_sja, r.m_ptrb[0], m_sic, m_d);
}


kernel_base<2, 1> *kern_mul_ij_jpq_iqp::match(const kern_mul_i_ipq_qp &z,
	list_t &in, list_t &out) {

	if(in.empty()) return 0;
	if(z.m_sic != 1) return 0;

	//	Rename i -> j.

	//	Minimize sib > 0:
	//	-----------------
	//	w   a    b    c
	//	np  spa  1    0
	//	nq  1    spb  0
	//	nj  sja  0    sjc
	//	ni  0    sib  sic  -->  c_i# = a_i#p#q b_q#p
	//	-----------------       [i_ipq_qp]
	//

	iterator_t ii = in.end();
	size_t sib_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepb(0) > 0) {
			if(i->stepa(0) % z.m_spa) continue;
			if(sib_min == 0 || sib_min > i->stepa(0)) {
				ii = i; sib_min = i->stepa(0);
			}
		}
	}
	if(ii == in.end()) return 0;

	kern_mul_ij_jpq_iqp zz;
	zz.m_d = z.m_d;
	zz.m_ni = ii->weight();
	zz.m_nj = z.m_ni;
	zz.m_np = z.m_np;
	zz.m_nq = z.m_nq;
	zz.m_sja = z.m_sia;
	zz.m_spa = z.m_spa;
	zz.m_sib = ii->stepa(1);
	zz.m_sqb = z.m_sqb;
	zz.m_sic = ii->stepb(0);
	in.splice(out.begin(), out, ii);

	kernel_base<2, 1> *kern = 0;

	return new kern_mul_ij_jpq_iqp(zz);
}


} // namespace libtensor
