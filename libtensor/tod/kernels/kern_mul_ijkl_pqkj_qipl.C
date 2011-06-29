#include "../../linalg/linalg.h"
#include "kern_mul_ijkl_pqkj_qipl.h"

namespace libtensor {


const char *kern_mul_ijkl_pqkj_qipl::k_clazz = "kern_mul_ijkl_pqkj_qipl";


void kern_mul_ijkl_pqkj_qipl::run(const loop_registers<2, 1> &r) {

//	if(m_sic == m_nj * m_sjc && m_sjc == m_nk * m_skc && m_skc == m_nl &&
//		m_spa == m_nq * m_sqa && m_sqa == m_nk * m_ska &&
//		m_ska == m_nj && m_sib == m_nq * m_sqb &&
//		m_sqb == m_np * m_spb && m_spb == m_nl) {
//
//		linalg::ijkl_ipql_qpkj_x(m_ni, m_nj, m_nk, m_nl, m_nq, m_np,
//			r.m_ptra[1], r.m_ptra[0], r.m_ptrb[0], m_d);
//		return;
//	}

	for(size_t i = 0; i < m_ni; i++) {
	for(size_t k = 0; k < m_nk; k++) {
	for(size_t p = 0; p < m_np; p++) {

		linalg::ij_pi_pj_x(m_nj, m_nl, m_nq,
			r.m_ptra[0] + p * m_spa + k * m_ska, m_sqa,
			r.m_ptra[1] + i * m_sib + p * m_spb, m_sqb,
			r.m_ptrb[0] + i * m_sic + k * m_skc, m_sjc,
			m_d);
	}
	}
	}
}


kernel_base<2, 1> *kern_mul_ijkl_pqkj_qipl::match(
	const kern_mul_ijk_pqji_qpk &z, list_t &in, list_t &out) {

	if(in.empty()) return 0;

	//	Rename i -> j, j -> k, k -> l

	//	Minimize sib > 0:
	//	------------------
	//	w   a    b     c
	//	nj  1    0     sjc
	//	np  spa  spb   0
	//	nl  0    1     1
	//	nk  ska  0     skc
	//	nq  sqa  sqb   0
	//	ni  0    sib   sic  -->  c_i#j#k#l = a_p#q#k#j b_q#i#p#l
	//	------------------       [ijkl_pqkj_qipl]
	//

	iterator_t ii = in.end();
	size_t sib_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepb(0) > 0) {
			if(i->stepa(1) % (z.m_np * z.m_spb)) continue;
			if(z.m_sqb % (i->weight() * i->stepa(1))) continue;
			if(i->stepb(0) % (z.m_ni * z.m_sic)) continue;
			if(sib_min == 0 || sib_min > i->stepa(1)) {
				ii = i; sib_min = i->stepa(1);
			}
		}
	}
	if(ii == in.end()) return 0;

	kern_mul_ijkl_pqkj_qipl zz;
	zz.m_d = z.m_d;
	zz.m_ni = ii->weight();
	zz.m_nj = z.m_ni;
	zz.m_nk = z.m_nj;
	zz.m_nl = z.m_nk;
	zz.m_np = z.m_np;
	zz.m_nq = z.m_nq;
	zz.m_spa = z.m_spa;
	zz.m_sqa = z.m_sqa;
	zz.m_ska = z.m_sja;
	zz.m_sqb = z.m_sqb;
	zz.m_sib = ii->stepa(1);
	zz.m_spb = z.m_spb;
	zz.m_sic = ii->stepb(0);
	zz.m_sjc = z.m_sic;
	zz.m_skc = z.m_sjc;
	in.splice(out.begin(), out, ii);

	kernel_base<2, 1> *kern = 0;

	return new kern_mul_ijkl_pqkj_qipl(zz);
}


} // namespace libtensor
