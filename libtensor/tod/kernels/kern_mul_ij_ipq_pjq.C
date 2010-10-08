#include "../../linalg/linalg.h"
#include "kern_mul_ij_ipq_pjq.h"

namespace libtensor {


const char *kern_mul_ij_ipq_pjq::k_clazz = "kern_mul_ij_ipq_pjq";


void kern_mul_ij_ipq_pjq::run(const loop_registers<2, 1> &r) {

	const double *pa = r.m_ptra[0], *pb = r.m_ptra[1];

	for(size_t p = 0; p < m_np; p++) {
		linalg::ij_ip_jp_x(m_ni, m_nj, m_nq, pa, m_sia, pb, m_sjb,
			r.m_ptrb[0], m_sic, m_d);
		pa += m_spa;
		pb += m_spb;
	}
}


kernel_base<2, 1> *kern_mul_ij_ipq_pjq::match(const kern_mul_ij_ip_jp &z,
	list_t &in, list_t &out) {

	if(in.empty()) return 0;

	//	Rename p -> q

	//	Minimize spa > 0:
	//	-----------------
	//	w   a    b    c
	//	nq  1    1    0
	//	nj  0    sjb  1
	//	ni  sia  0    sic
	//	np  spa  spb  0    -->  c_i#j = a_i#p#q b_p#j#q
	//	-----------------       [ij_ipq_pjq]
	//

	iterator_t ip = in.end();
	size_t spa_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) > 0 && i->stepa(1) > 0 && i->stepb(0) == 0) {
			if(i->stepa(0) % z.m_np ||
				z.m_sia % (i->weight() * i->stepa(0)))
				continue;
			if(i->stepa(1) % (z.m_sjb * z.m_nj)) continue;
			if(spa_min == 0 || spa_min > i->stepa(0)) {
				ip = i; spa_min = i->stepa(0);
			}
		}
	}
	if(ip == in.end()) return 0;

	kern_mul_ij_ipq_pjq zz;
	zz.m_d = z.m_d;
	zz.m_ni = z.m_ni;
	zz.m_nj = z.m_nj;
	zz.m_np = ip->weight();
	zz.m_nq = z.m_np;
	zz.m_sia = z.m_sia;
	zz.m_spa = ip->stepa(0);
	zz.m_spb = ip->stepa(1);
	zz.m_sjb = z.m_sjb;
	zz.m_sic = z.m_sic;
	in.splice(out.begin(), out, ip);

	kernel_base<2, 1> *kern = 0;

	return new kern_mul_ij_ipq_pjq(zz);
}


} // namespace libtensor
