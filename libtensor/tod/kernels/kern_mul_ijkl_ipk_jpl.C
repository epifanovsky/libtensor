#include "../../linalg/linalg.h"
#include "kern_mul_ijkl_ipk_jpl.h"

namespace libtensor {


const char *kern_mul_ijkl_ipk_jpl::k_clazz = "kern_mul_ijkl_ipk_jpl";


void kern_mul_ijkl_ipk_jpl::run(const loop_registers<2, 1> &r) {

	for(size_t i = 0; i < m_ni; i++)
	for(size_t j = 0; j < m_nj; j++) {
		linalg::ij_pi_pj_x(m_nk, m_nl, m_np,
			r.m_ptra[0] + i * m_sia, m_spa,
			r.m_ptra[1] + j * m_sjb, m_spb,
			r.m_ptrb[0] + i * m_sic + j * m_sjc, m_skc, m_d);
	}
}


kernel_base<2, 1> *kern_mul_ijkl_ipk_jpl::match(const kern_mul_ijk_pj_ipk &z,
	list_t &in, list_t &out) {

	if(in.empty()) return 0;

	//	Rename i -> j, j -> k, k -> l.

	//	Minimize sia > 0:
	//	------------------
	//	w   a    b     c
	//	nk  1    0     skc
	//	np  spa  spb   0
	//	nl  0    1     1
	//	nj  0    sjb   sjc
	//	ni  sia  0     sic  -->  c_i#j#k#l = a_i#p#k b_j#p#l
	//	------------------       [ijkl_ipk_jpl]
	//

	iterator_t ii = in.end();
	size_t sia_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) > 0 && i->stepa(1) == 0 && i->stepb(0) > 0) {
			if(i->stepa(0) % (z.m_spa * z.m_np)) continue;
			if(i->stepb(0) % (z.m_sic * z.m_ni)) continue;
			if(sia_min == 0 || sia_min > i->stepa(0)) {
				ii = i; sia_min = i->stepa(0);
			}
		}
	}
	if(ii == in.end()) return 0;

	kern_mul_ijkl_ipk_jpl zz;
	zz.m_d = z.m_d;
	zz.m_ni = ii->weight();
	zz.m_nj = z.m_ni;
	zz.m_nk = z.m_nj;
	zz.m_nl = z.m_nk;
	zz.m_np = z.m_np;
	zz.m_sia = ii->stepa(0);
	zz.m_spa = z.m_spa;
	zz.m_sjb = z.m_sib;
	zz.m_spb = z.m_spb;
	zz.m_sic = ii->stepb(0);
	zz.m_sjc = z.m_sic;
	zz.m_skc = z.m_sjc;
	in.splice(out.begin(), out, ii);

	kernel_base<2, 1> *kern = 0;

	return new kern_mul_ijkl_ipk_jpl(zz);
}


} // namespace libtensor
