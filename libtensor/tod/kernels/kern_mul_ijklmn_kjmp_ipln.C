#include "../../linalg/linalg.h"
#include "kern_mul_ijklmn_kjmp_ipln.h"

namespace libtensor {


const char *kern_mul_ijklmn_kjmp_ipln::k_clazz = "kern_mul_ijklmn_kjmp_ipln";


void kern_mul_ijklmn_kjmp_ipln::run(const loop_registers<2, 1> &r) {

	for(size_t i = 0; i < m_ni; i++)
	for(size_t j = 0; j < m_nj; j++)
	for(size_t k = 0; k < m_nk; k++)
	for(size_t l = 0; l < m_nl; l++) {
		linalg::ij_ip_pj_x(m_nm, m_nn, m_np,
			r.m_ptra[0] + k * m_ska + j * m_sja, m_sma,
			r.m_ptra[1] + i * m_sib + l * m_slb, m_spb,
			r.m_ptrb[0] + i * m_sic + j * m_sjc + k * m_skc +
				l * m_slc, m_smc,
			m_d);
	}
}


kernel_base<2, 1> *kern_mul_ijklmn_kjmp_ipln::match(
	const kern_mul_ijklm_jlp_ipkm &z, list_t &in, list_t &out) {

	if(in.empty()) return 0;

	//	Rename k -> l, l -> m, m -> n.

	//	Minimize ska > 0.
	//	-----------------
	//	w   a    b    c
	//	nn  0    1    1
	//	np  1    spb  0
	//	nm  sma  0    smc
	//	ni  0    sib  sic
	//	nj  sja  0    sjc
	//	nl  0    slb  slc
	//	nk  ska  0    skc  --> c_i#j#k#l#m#n = a_k#j#m#p b_i#p#l#n
	//	-----------------      [ijklmn_kjmp_ipln]
	//

	iterator_t ik = in.end();
	size_t ska_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) > 0 && i->stepa(1) == 0 && i->stepb(0) > 0) {
			if(i->stepa(0) % (z.m_nj * z.m_sja)) continue;
			if(i->stepb(0) % (z.m_nk * z.m_skc)) continue;
			if(z.m_sjc % (i->weight() * i->stepb(0))) continue;
			if(ska_min == 0 || ska_min > i->stepa(0)) {
				ik = i; ska_min = i->stepa(0);
			}
		}
	}
	if(ik == in.end()) return 0;

	kern_mul_ijklmn_kjmp_ipln zz;
	zz.m_d = z.m_d;
	zz.m_ni = z.m_ni;
	zz.m_nj = z.m_nj;
	zz.m_nk = ik->weight();
	zz.m_nl = z.m_nk;
	zz.m_nm = z.m_nl;
	zz.m_nn = z.m_nm;
	zz.m_np = z.m_np;
	zz.m_ska = ik->stepa(0);
	zz.m_sja = z.m_sja;
	zz.m_sma = z.m_sla;
	zz.m_sib = z.m_sib;
	zz.m_spb = z.m_spb;
	zz.m_slb = z.m_skb;
	zz.m_sic = z.m_sic;
	zz.m_sjc = z.m_sjc;
	zz.m_skc = ik->stepb(0);
	zz.m_slc = z.m_skc;
	zz.m_smc = z.m_slc;
	in.splice(out.begin(), out, ik);

	kernel_base<2, 1> *kern = 0;

	return new kern_mul_ijklmn_kjmp_ipln(zz);
}


} // namespace libtensor
