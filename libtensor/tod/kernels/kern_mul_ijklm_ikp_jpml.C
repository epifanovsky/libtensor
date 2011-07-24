#include "../../linalg/linalg.h"
#include "kern_mul_ijklm_ikp_jpml.h"

namespace libtensor {


const char *kern_mul_ijklm_ikp_jpml::k_clazz = "kern_mul_ijklm_ikp_jpml";


void kern_mul_ijklm_ikp_jpml::run(const loop_registers<2, 1> &r) {

	for(size_t i = 0; i < m_ni; i++)
	for(size_t j = 0; j < m_nj; j++) {
		linalg::ijk_ip_pkj_x(m_nk, m_nl, m_nm, m_np,
			r.m_ptra[0] + i * m_sia, m_ska,
			r.m_ptra[1] + j * m_sjb, m_smb, m_spb,
			r.m_ptrb[0] + i * m_sic + j * m_sjc, m_slc, m_skc, m_d);
	}
}


kernel_base<2, 1> *kern_mul_ijklm_ikp_jpml::match(
	const kern_mul_ijkl_ijp_plk &z, list_t &in, list_t &out) {

	if(in.empty()) return 0;

	//	Rename j -> k, k -> l, l -> m.

	//	Minimize sjb > 0:
	//	-----------------
	//	w   a    b    c
	//	nl  0    1    slc
	//	np  1    spb  0
	//	nm  0    smb  1
	//	nk  ska  0    skc
	//	ni  sia  0    sic
	//	nj  0    sjb  sjc  -->  c_i#j#k#l#m = a_i#k#p b_j#p#m#l
	//	-----------------       [ijklm_ikp_jpml]
	//

	iterator_t ij = in.end();
	size_t sjb_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepb(0) > 0) {
			if(i->stepa(1) % (z.m_np * z.m_spb)) continue;
			if(i->stepb(0) % (z.m_nj * z.m_sjc)) continue;
			if(z.m_sic % (i->weight() * i->stepb(0))) continue;
			if(sjb_min == 0 || sjb_min > i->stepa(1)) {
				ij = i; sjb_min = i->stepa(1);
			}
		}
	}
	if(ij == in.end()) return 0;

	kern_mul_ijklm_ikp_jpml zz;
	zz.m_d = z.m_d;
	zz.m_ni = z.m_ni;
	zz.m_nj = ij->weight();
	zz.m_nk = z.m_nj;
	zz.m_nl = z.m_nk;
	zz.m_nm = z.m_nl;
	zz.m_np = z.m_np;
	zz.m_sia = z.m_sia;
	zz.m_ska = z.m_sja;
	zz.m_sjb = ij->stepa(1);
	zz.m_spb = z.m_spb;
	zz.m_smb = z.m_slb;
	zz.m_sic = z.m_sic;
	zz.m_sjc = ij->stepb(0);
	zz.m_skc = z.m_sjc;
	zz.m_slc = z.m_skc;
	in.splice(out.begin(), out, ij);

	kernel_base<2, 1> *kern = 0;

	return new kern_mul_ijklm_ikp_jpml(zz);
}


} // namespace libtensor
