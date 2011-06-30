#include "../../linalg/linalg.h"
#include "kern_mul_ijk_ip_pkj.h"

namespace libtensor {


const char *kern_mul_ijk_ip_pkj::k_clazz = "kern_mul_ijk_ip_pkj";


void kern_mul_ijk_ip_pkj::run(const loop_registers<2, 1> &r) {

	linalg::ijk_ip_pkj_x(m_ni, m_nj, m_nk, m_np, r.m_ptra[0], m_sia,
		r.m_ptra[1], m_skb, m_spb, r.m_ptrb[0], m_sjc, m_sic, m_d);
}


kernel_base<2, 1> *kern_mul_ijk_ip_pkj::match(const kern_mul_ij_p_pji &z,
	list_t &in, list_t &out) {

	if(in.empty()) return 0;
	if(z.m_spa != 1) return 0;

	//	Rename i -> j, j -> k.

	//	Minimize sia > 0:
	//	-----------------
	//	w   a    b    c
	//	nj  0    1    sjc
	//	np  1    spb  0
	//	nk  0    skb  1
	//	ni  sia  0    sic  -->  c_i#j#k = a_i#p b_p#k#j
	//	-----------------       [ijk_ip_pkj]
	//

	iterator_t ii = in.end();
	size_t sia_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) > 0 && i->stepa(1) == 0 && i->stepb(0) > 0) {
			if(i->stepa(0) % z.m_np) continue;
			if(i->stepb(0) % (z.m_ni * z.m_sic)) continue;
			if(sia_min == 0 || sia_min > i->stepa(0)) {
				ii = i; sia_min = i->stepa(0);
			}
		}
	}
	if(ii == in.end()) return 0;

	kern_mul_ijk_ip_pkj zz;
	zz.m_d = z.m_d;
	zz.m_ni = ii->weight();
	zz.m_nj = z.m_ni;
	zz.m_nk = z.m_nj;
	zz.m_np = z.m_np;
	zz.m_sia = ii->stepa(0);
	zz.m_skb = z.m_sjb;
	zz.m_spb = z.m_spb;
	zz.m_sic = ii->stepb(0);
	zz.m_sjc = z.m_sic;
	in.splice(out.begin(), out, ii);

	kernel_base<2, 1> *kern = 0;

	return new kern_mul_ijk_ip_pkj(zz);
}


} // namespace libtensor
