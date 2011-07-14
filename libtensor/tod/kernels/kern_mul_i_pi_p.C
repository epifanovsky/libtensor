#include "../../linalg/linalg.h"
#include "kern_mul_i_pi_p.h"
#include "kern_mul_ij_pi_jp.h"
#include "kern_mul_ij_pi_pj.h"
#include "kern_mul_ij_pj_ip.h"
#include "kern_mul_ij_pj_pi.h"
#include "kern_mul_ij_pji_p.h"
#include "kern_mul_ijk_pi_pkj.h"

namespace libtensor {


const char *kern_mul_i_pi_p::k_clazz = "kern_mul_i_pi_p";


void kern_mul_i_pi_p::run(const loop_registers<2, 1> &r) {

	linalg::i_pi_p_x(m_ni, m_np, r.m_ptra[0], m_spa, r.m_ptra[1], m_spb,
		r.m_ptrb[0], m_sic, m_d);
}


kernel_base<2, 1> *kern_mul_i_pi_p::match(const kern_mul_i_i_x &z,
	list_t &in, list_t &out) {

	if(in.empty()) return 0;

	//	Minimize spb > 0:
	//	-------------------
	//	w   a    b     c
	//	ni  1    0     sic
	//	np  spa  spb   0     -->  c_i# = a_p#i b_p#
	//	-------------------       [i_pi_p]
	//

	iterator_t ip = in.end();
	size_t spb_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) > 0 && i->stepa(1) > 0 && i->stepb(0) == 0) {
			if(i->stepa(0) % z.m_ni) continue;
			if(spb_min == 0 || spb_min > i->stepa(1)) {
				ip = i; spb_min = i->stepa(1);
			}
		}
	}
	if(ip == in.end()) return 0;

	kern_mul_i_pi_p zz;
	zz.m_d = z.m_d;
	zz.m_ni = z.m_ni;
	zz.m_np = ip->weight();
	zz.m_spa = ip->stepa(0);
	zz.m_spb = ip->stepa(1);
	zz.m_sic = z.m_sic;
	in.splice(out.begin(), out, ip);

	kernel_base<2, 1> *kern = 0;

	if(kern = kern_mul_ij_pi_jp::match(zz, in, out)) return kern;
	if(kern = kern_mul_ij_pi_pj::match(zz, in, out)) return kern;
	if(kern = kern_mul_ij_pj_ip::match(zz, in, out)) return kern;
	if(kern = kern_mul_ij_pj_pi::match(zz, in, out)) return kern;
	if(kern = kern_mul_ij_pji_p::match(zz, in, out)) return kern;
	if(kern = kern_mul_ijk_pi_pkj::match(zz, in, out)) return kern;

	return new kern_mul_i_pi_p(zz);
}


} // namespace libtensor
