#include "../defs.h"
#include "../exception.h"
#include "../blas.h"
#include "../linalg.h"
#include "../linalg/linalg.h"
#include "loop_list_mul.h"
#include "overflow.h"

namespace libtensor {


const char *loop_list_mul::k_clazz = "loop_list_mul";


void loop_list_mul::run_loop(list_t &loop, registers &r, double c) {

	//~ std::cout << "[";
	match_l1(loop, c);
	//~ std::cout << "]" << std::endl;

	loop_list_mul::start_timer(m_kernelname);

	iterator_t begin = loop.begin(), end = loop.end();
	if(begin != end) {
		loop_list_base<2, 1, loop_list_mul>::exec(
			*this, begin, end, r);
	}

	loop_list_mul::stop_timer(m_kernelname);
}


void loop_list_mul::match_l1(list_t &loop, double d) {

	if(loop.size() < 1) return;

	//	1. Minimize k1 > 0:
	//	------------
	//	w   a   b  c
	//	np  k1  1  0  -->  c_# = a_p# b_p
	//	------------       sz(p) = np, sz(#) = k1
	//	                   [x_p_p]
	//
	//	2. Minimize k1a:
	//	-------------
	//	w   a  b  c
	//	ni  1  0  k1a  -->  c_i# = a_i b
	//	-------------       sz(i) = ni, sz(#) = k1a
	//	                    [i_i_x]
	//
	//	3. Minimize k1b:
	//	-------------
	//	w   a  b  c
	//	ni  0  1  k1b  -->  c_i# = a b_i
	//	-------------       sz(i) = ni, sz(#) = k1b
	//	                    [i_x_i]
	//
	iterator_t i1 = loop.end(), i2 = loop.end(), i3 = loop.end();
	size_t k1_min = 0, k1a_min = 0, k1b_min = 0;
	for(iterator_t i = loop.begin(); i != loop.end(); i++) {
		if(i->stepa(0) > 0 && i->stepa(1) == 1 && i->stepb(0) == 0) {
			if(k1_min == 0 || k1_min > i->stepa(0)) {
				i1 = i; k1_min = i->stepa(0);
			}
		}
		if(i->stepa(0) == 1 && i->stepa(1) == 0) {
			if(k1a_min == 0 || k1a_min > i->stepb(0)) {
				i2 = i; k1a_min = i->stepb(0);
			}
		}
		if(i->stepa(0) == 0 && i->stepa(1) == 1) {
			if(k1b_min == 0 || k1b_min > i->stepb(0)) {
				i3 = i; k1b_min = i->stepb(0);
			}
		}
	}
	if(i1 != loop.end() && k1_min == 1) {
		m_kernelname = "x_p_p[1]";
		//~ std::cout << m_kernelname;
		i1->fn() = &loop_list_mul::fn_x_p_p;
		args_x_p_p &args = m_x_p_p;
		args.d = d;
		args.np = i1->weight();
		args.spa = i1->stepa(0);
		args.spb = 1;
		match_x_p_p(loop, d, i1->weight(), i1->stepa(0));
		loop.splice(loop.end(), loop, i1);
		return;
	}
	if(i2 != loop.end() && k1b_min != 1) {
		m_kernelname = "i_i_x";
		//~ std::cout << m_kernelname;
		i2->fn() = &loop_list_mul::fn_i_i_x;
		args_i_i_x &args = m_i_i_x;
		args.d = d;
		args.ni = i2->weight();
		args.sic = i2->stepb(0);
		match_i_i_x(loop, d, i2->weight(), i2->stepb(0));
		loop.splice(loop.end(), loop, i2);
		return;
	}
	if(i3 != loop.end()) {
		m_kernelname = "i_x_i";
		//~ std::cout << m_kernelname;
		i3->fn() = &loop_list_mul::fn_i_x_i;
		args_i_x_i &args = m_i_x_i;
		args.d = d;
		args.ni = i3->weight();
		args.sic = i3->stepb(0);
		match_i_x_i(loop, d, i3->weight(), i3->stepb(0));
		loop.splice(loop.end(), loop, i3);
		return;
	}
	if(i1 != loop.end()) {
		m_kernelname = "x_p_p[2]";
		//~ std::cout << m_kernelname;
		i1->fn() = &loop_list_mul::fn_x_p_p;
		args_x_p_p &args = m_x_p_p;
		args.d = d;
		args.np = i1->weight();
		args.spa = i1->stepa(0);
		args.spb = 1;
		match_x_p_p(loop, d, i1->weight(), i1->stepa(0));
		loop.splice(loop.end(), loop, i1);
		return;
	}

	iterator_t i0 = loop.begin();

	//~ std::cout << "generic";
	m_kernelname = "generic";
	i0->fn() = &loop_list_mul::fn_generic;
	m_generic.m_d = d;
	m_generic.m_n = i0->weight();
	m_generic.m_stepa = i0->stepa(0);
	m_generic.m_stepb = i0->stepa(1);
	m_generic.m_stepc = i0->stepb(0);
	loop.splice(loop.end(), loop, i0);
}


void loop_list_mul::match_x_p_p(list_t &loop, double d, size_t np, size_t spa) {

	if(loop.size() < 2) return;

	//	Found pattern (k1 > 0):
	//	-------------
	//	w   a    b  c
	//	np  spa  1  0  -->  c_# = a_p# b_p
	//	-------------       sz(p) = np, sz(#) = spa
	//	                    [x_p_p]
	//

	//	1. If spa == 1, minimize k2a:
	//	----------------
	//	w   a       b  c
	//	np  1       1  0
	//	w2  k2a*np  0  1  -->  c_i = a_i$p b_p
	//	----------------       sz(i) = w2, sz(p) = np, sz($) = k2a
	//	                       [i_ip_p]
	//
	//	2. If spa == 1, minimize k2b:
	//	----------------
	//	w   a  b       c
	//	np  1  1       0
	//	w2  0  k2b*np  1  -->  c_i = a_p b_i%p
	//	----------------       sz(i) = w2, sz(p) = np, sz(%) = k2b
	//	                       [i_p_ip]
	//
	//	3. Minimize k2c:
	//	---------------------
	//	w   a       b       c
	//	np  k1'*nq  1       0
	//	nq  1       k2c*np  0  -->  c = a_p$q b_q%p
	//	---------------------       sz(p) = np, sz(q) = nq, sz($) = k1',
	//	                            sz(%) = k2c
	//	                            [x_pq_qp]
	//
	size_t k2a_min = 0, k2b_min = 0, k2c_min = 0;
	iterator_t i1 = loop.end(), i2 = loop.end(), i3 = loop.end();
	for(iterator_t i = loop.begin(); i != loop.end(); i++) {
		if(i->stepb(0) > 1) continue;
		if(spa == 1 && i->stepb(0) == 1) {
			if(i->stepa(1) == 0 && i->stepa(0) % np == 0) {
				register size_t k2a = i->stepa(0) / np;
				if(k2a_min == 0 || k2a_min > k2a) {
					k2a_min = k2a; i1 = i;
				}
			}
			if(i->stepa(0) == 0 && i->stepa(1) % np == 0) {
				register size_t k2b = i->stepa(1) / np;
				if(k2b_min == 0 || k2b_min > k2b) {
					k2b_min = k2b; i2 = i;
				}
			}
		}
		if(i->stepa(0) == 1 && i->stepb(0) == 0
			&& spa % i->weight() == 0) {
			register size_t k2c = i->stepa(1) / np;
			if(k2c_min == 0 || k2c_min > k2c) {
				k2c_min = k2c; i3 = i;
			}
		}
	}

	if(i1 != loop.end()) {
		m_kernelname = "i_ip_p";
		//~ std::cout << m_kernelname;
		i1->fn() = &loop_list_mul::fn_i_ip_p;
		args_i_ip_p &args = m_i_ip_p;
		args.d = d;
		args.ni = i1->weight();
		args.np = np;
		args.sia = i1->stepa(0);
		args.sic = 1;
		args.spb = 1;
		match_dgemv_n_a_l3(loop, d, np, i1->weight(), i1->stepa(0));
		loop.splice(loop.end(), loop, i1);
		return;
	}

	if(i2 != loop.end()) {
		m_kernelname = "i_p_ip";
		//~ std::cout << m_kernelname;
		i2->fn() = &loop_list_mul::fn_i_p_ip;
		args_i_p_ip &args = m_i_p_ip;
		args.d = d;
		args.ni = i2->weight();
		args.np = np;
		args.sib = i2->stepa(1);
		args.sic = 1;
		args.spa = 1;
		match_dgemv_n_b_l3(loop, d, np, i2->weight(), i2->stepa(1));
		loop.splice(loop.end(), loop, i2);
		return;
	}

	if(i3 != loop.end()) {
		m_kernelname = "x_pq_qp";
		//~ std::cout << m_kernelname;
		i3->fn() = &loop_list_mul::fn_x_pq_qp;
		args_x_pq_qp &args = m_x_pq_qp;
		args.d = d;
		args.np = np;
		args.nq = i3->weight();
		args.spa = spa;
		args.sqb = i3->stepa(1);
		match_x_pq_qp(loop, d, np, i3->weight(), spa, i3->stepa(1));
		loop.splice(loop.end(), loop, i3);
		return;
	}
}


void loop_list_mul::match_i_i_x(list_t &loop, double d, size_t ni, size_t k1) {

	if(loop.size() < 2) return;

	//	Found pattern:
	//	------------
	//	w   a  b  c
	//	ni  1  0  k1  -->  c_i# = a_i b
	//	------------       sz(i) = ni, sz(#) = k1
	//	                   [i_i_x]
	//

	//	1. Minimize k2a:
	//	------------------
	//	w   a       b   c
	//	ni  1       0   k1
	//	w2  k2a*w1  1   0   -->  c_i# = a_p$i b_p
	//	------------------       sz(i) = ni, sz(p) = w2
	//	                         sz(#) = k1, sz($) = k2a
	//	                         [i_pi_p]
	//
	//	2. Minimize k2b:
	//	------------------
	//	w   a       b   c
	//	ni  1       0   k1
	//	w2  k2b*w1  k3  0   -->  c_i# = a_p$i b_p%
	//	------------------       sz(i) = ni, sz(p) = w2
	//	                         sz(#) = k1, sz($) = k2b, sz(%) = k3
	//	                         [i_pi_p]
	//	-----------------
	//	w   a       b   c
	//	ni  1       0   1
	//	w2  k2b*w1  k3  0  -->  c_i = a_p$i b_p%
	//	-----------------       sz(i) = ni, sz(p) = w2
	//	                        sz($) = k2b, sz(%) = k3
	//	                        [i_pi_p]
	//
	size_t k2a_min = 0, k2b_min = 0;
	iterator_t i1 = loop.end(), i2 = loop.end();
	for(iterator_t i = loop.begin(); i != loop.end(); i++) {
		if(i->stepb(0) != 0) continue;
		if(i->stepa(0) % ni != 0) continue;

		register size_t k2 = i->stepa(0) / ni;
		if(i->stepa(1) == 1 && (k2a_min == 0 || k2a_min > k2)) {
			k2a_min = k2; i1 = i;
		}
		if(k2b_min == 0 || k2b_min > k2) {
			k2b_min = k2; i2 = i;
		}
	}
	if(i1 != loop.end() && !(k1 == 1 && i2 != loop.end())) {
		m_kernelname = "i_pi_p";
		//~ std::cout << m_kernelname;
		i1->fn() = &loop_list_mul::fn_i_pi_p;
		args_i_pi_p &args = m_i_pi_p;
		args.d = d;
		args.ni = ni;
		args.np = i1->weight();
		args.sic = k1;
		args.spa = i1->stepa(0);
		args.spb = 1;
		match_dgemv_t_a1_l3(loop, d, ni, i1->weight(), k1,
			i1->stepa(0));
		loop.splice(loop.end(), loop, i1);
		return;
	}
	if(i2 != loop.end()) {
		m_kernelname = "i_pi_p";
		//~ std::cout << m_kernelname;
		i2->fn() = &loop_list_mul::fn_i_pi_p;
		args_i_pi_p &args = m_i_pi_p;
		args.d = d;
		args.ni = ni;
		args.np = i2->weight();
		args.sic = k1;
		args.spa = i2->stepa(0);
		args.spb = i2->stepa(1);
		if(k1 == 1) {
			match_dgemv_t_a2_l3(loop, d, ni, i2->weight(),
				i2->stepa(0), i2->stepa(1));
		}
		loop.splice(loop.end(), loop, i2);
		return;
	}
}


void loop_list_mul::match_i_x_i(list_t &loop, double d, size_t ni, size_t k1) {

	if(loop.size() < 2) return;

	//	Found pattern:
	//	------------
	//	w   a  b  c
	//	ni  0  1  k1  -->  c_i# = a b_i
	//	------------       sz(i) = ni, sz(#) = k1
	//	                   [i_x_i]

	//	1. Minimize k2a:
	//	-----------------
	//	w   a  b       c
	//	ni  0  1       k1
	//	w2  1  k2a*w1  0   -->  c_i# = a_p b_p%i
	//	-----------------       sz(i) = ni, sz(p) = w2
	//	                        sz(#) = k1, sz(%) = k2a
	//	                        [i_p_pi]
	//
	//	2. Minimize k2b:
	//	------------------
	//	w   a   b       c
	//	ni  0   1       k1
	//	w2  k3  k2b*w1  0  -->  c_i# = a_p$ b_p%i
	//      ------------------      sz(i) = ni, sz(p) = w2
	//	                        sz(#) = k1, sz($) = k3, sz(%) = k2b
	//	                        [i_p_pi]
	//	-----------------
	//	w   a   b       c
	//	ni  0   1       1
	//	w2  k3  k2b*w1  0  -->  c_i = a_p$ b_p%i
	//	-----------------       sz(i) = ni, sz(p) = w2
	//	                        sz($) = k2b, sz(%) = k3
	//	                        [i_p_pi]
	//
	size_t k2a_min = 0, k2b_min = 0;
	iterator_t i1 = loop.end(), i2 = loop.end();
	for(iterator_t i = loop.begin(); i != loop.end(); i++) {
		if(i->stepb(0) != 0) continue;
		if(i->stepa(1) % ni != 0) continue;

		register size_t k2 = i->stepa(1) / ni;
		if(i->stepa(0) == 1 && (k2a_min == 0 || k2a_min > k2)) {
			k2a_min = k2; i1 = i;
		}
		if(k2b_min == 0 || k2b_min > k2) {
			k2b_min = k2; i2 = i;
		}
	}
	if(i1 != loop.end() && !(k1 == 1 && i2 != loop.end())) {
		m_kernelname = "i_p_pi";
		//~ std::cout << m_kernelname;
		i1->fn() = &loop_list_mul::fn_i_p_pi;
		args_i_p_pi &args = m_i_p_pi;
		args.d = d;
		args.ni = ni;
		args.np = i1->weight();
		args.sic = k1;
		args.spa = 1;
		args.spb = i1->stepa(1);
		match_dgemv_t_b1_l3(loop, d, ni, i1->weight(), k1,
			i1->stepa(1));
		loop.splice(loop.end(), loop, i1);
		return;
	}
	if(i2 != loop.end()) {
		m_kernelname = "i_p_pi";
		//~ std::cout << m_kernelname;
		i2->fn() = &loop_list_mul::fn_i_p_pi;
		args_i_p_pi &args = m_i_p_pi;
		args.d = d;
		args.ni = ni;
		args.np = i2->weight();
		args.sic = k1;
		args.spa = i2->stepa(0);
		args.spb = i2->stepa(1);
		if(k1 == 1) {
			match_dgemv_t_b2_l3(loop, d, ni, i2->weight(),
				i2->stepa(1), i2->stepa(0));
		}
		loop.splice(loop.end(), loop, i2);
		return;
	}
}


void loop_list_mul::match_dgemv_n_a_l3(list_t &loop, double d, size_t w1,
	size_t w2, size_t k1w1) {

	if(loop.size() < 3) return;

	//	Found pattern:
	//	---------------
	//	w   a      b  c
	//	w1  1      1  0
	//	w2  k1*w1  0  1  -->  c_i = a_i$p b_p
	//	---------------       sz(i) = w2, sz(p) = w1, sz($) = k1
	//	                      [dgemv_n_a]

	//	1. Minimize k2:
	//	-----------------------
	//	w   a      b      c
	//	w1  1      1      0
	//	w2  k1*w1  0      1
	//	w3  0      k2*w1  k3*w2  -->  c_j#i = a_i$p b_j%p
	//	-----------------------       sz(i) = w2, sz(j) = w3,
	//	                              sz(p) = w1
	//	                              sz(#) = k3, sz($) = k1,
	//	                              sz(%) = k2
	//	                              [ij_jp_ip]
	//
	size_t k2_min = 0;
	iterator_t i1 = loop.end();
	for(iterator_t i = loop.begin(); i != loop.end(); i++) {
		if(i->stepa(0) != 0) continue;
		if(i->stepb(0) % w2 != 0) continue;
		if(i->stepa(1) % w1 != 0) continue;

		register size_t k2 = i->stepa(1) / w1;
		if(k2_min == 0 || k2_min > k2) {
			k2_min = k2; i1 = i;
		}
	}
	if(i1 != loop.end()) {
		m_kernelname = "ij_jp_ip";
		//~ std::cout << m_kernelname;
		i1->fn() = &loop_list_mul::fn_ij_jp_ip;
		args_ij_jp_ip &args = m_ij_jp_ip;
		args.d = d;
		args.ni = i1->weight();
		args.nj = w2;
		args.np = w1;
		args.sib = i1->stepa(1);
		args.sic = i1->stepb(0);
		args.sja = k1w1;
		loop.splice(loop.end(), loop, i1);
		return;
	}
}


void loop_list_mul::match_dgemv_n_b_l3(list_t &loop, double d, size_t w1,
	size_t w2, size_t k1w1) {

	if(loop.size() < 3) return;

	//	Found pattern:
	//	---------------
	//	w   a  b      c
	//	w1  1  1      0
	//	w2  0  k1*w1  1  -->  c_i = a_p b_i%p
	//	---------------       sz(i) = w2, sz(p) = w1, sz(%) = k1
	//	                      [dgemv_n_b]
	//

	//	1. Minimize k2:
	//	-----------------------
	//	w   a      b      c
	//	w1  1      1      0
	//	w2  0      k1*w1  1
	//	w3  k2*w1  0      k3*w2  -->  c_j#i = a_j$p b_i%p
	//	-----------------------       sz(i) = w2, sz(j) = w3,
	//	                              sz(p) = w1
	//	                              sz(#) = k3, sz($) = k2,
	//	                              sz(%) = k1
	//	                              [ij_ip_jp]
	//
	size_t k2_min = 0;
	iterator_t i1 = loop.end();
	for(iterator_t i = loop.begin(); i != loop.end(); i++) {
		if(i->stepa(1) != 0) continue;
		if(i->stepb(0) % w2 != 0) continue;
		if(i->stepa(0) % w1 != 0) continue;

		register size_t k2 = i->stepa(0) / w1;
		if(k2_min == 0 || k2_min > k2) {
			k2_min = k2; i1 = i;
		}
	}
	if(i1 != loop.end()) {
		m_kernelname = "ij_ip_jp";
		//~ std::cout << m_kernelname;
		i1->fn() = &loop_list_mul::fn_ij_ip_jp;
		args_ij_ip_jp &args = m_ij_ip_jp;
		args.d = d;
		args.ni = i1->weight();
		args.nj = w2;
		args.np = w1;
		args.sia = i1->stepa(0);
		args.sic = i1->stepb(0);
		args.sjb = k1w1;
		loop.splice(loop.end(), loop, i1);
		return;
	}
}


void loop_list_mul::match_dgemv_t_a1_l3(list_t &loop, double d, size_t w1,
	size_t w2, size_t k1, size_t k2w1) {

	if(loop.size() < 3) return;

	//	Found pattern:
	//	-----------------
	//	w   a      b   c
	//	w1  1      0   k1
	//	w2  k2*w1  1   0   -->  c_i# = a_p$i b_p
	//	-----------------       sz(i) = w1, sz(p) = w2
	//	                        sz(#) = k1, sz($) = k2
	//	                        [dgemv_t_a]
	//

	//	1. Minimize k4:
	//	------------------------
	//	w   a      b      c
	//	w1  1      0      k1'*w3
	//	w2  k2*w1  1      0
	//	w3  0      k4*w2  1       -->  c_i#j = a_p$i b_j%p
	//	------------------------       sz(i) = w1, sz(j) = w3,
	//	                               sz(p) = w2
	//	                               sz(#) = k1', sz($) = k2,
	//	                               sz(%) = k4
	//	                               [ij_pi_jp]
	//
	size_t k4_min = 0;
	iterator_t i1 = loop.end();
	for(iterator_t i = loop.begin(); i != loop.end(); i++) {
		if(i->stepa(0) != 0 || i->stepb(0) != 1) continue;
		if(k1 % i->weight() != 0) continue;
		if(i->stepa(1) % w2 != 0) continue;

		register size_t k4 = i->stepa(1) / w2;
		if(k4_min == 0 || k4_min > k4) {
			k4_min = k4; i1 = i;
		}
	}
	if(i1 != loop.end()) {
		m_kernelname = "ij_pi_jp";
		//~ std::cout << m_kernelname;
		i1->fn() = &loop_list_mul::fn_ij_pi_jp;
		args_ij_pi_jp &args = m_ij_pi_jp;
		args.d = d;
		args.ni = w1;
		args.nj = i1->weight();
		args.np = w2;
		args.sic = k1;
		args.sjb = i1->stepa(1);
		args.spa = k2w1;
		loop.splice(loop.end(), loop, i1);
		return;
	}
}


void loop_list_mul::match_dgemv_t_a2_l3(list_t &loop, double d, size_t w1,
	size_t w2, size_t k2w1, size_t k3) {

	if(loop.size() < 3) return;

	//	Found pattern:
	//	---------------
	//	w   a     b   c
	//	w1  1     0   1
	//	w2  k2w1  k3  0  -->  c_i = a_p$i b_p%
	//	---------------       sz(i) = w1, sz(p) = w2,
	//	                      sz($) = k2, sz(%) = k3
	//	                      [dgemv_t_a]
	//

	//	1. If k3 == 1, minimize k5:
	//	-----------------------
	//	w   a      b      c
	//	w1  1      0      1
	//	w2  k2*w1  1      0
	//	w3  0      k5*w2  k6*w1  --> c_j#i = a_p$i b_j%p
	//	-----------------------      sz(i) = w1, sz(j) = w3,
	//	                             sz(p) = w2
	//	                             sz(#) = k6, sz($) = k2,
	//	                             sz(%) = k5
	//	                             [ij_pj_ip]
	//
	if(k3 == 1) {
		size_t k5_min = 0;
		iterator_t i1 = loop.end();
		for(iterator_t i = loop.begin(); i != loop.end(); i++) {
			if(i->stepa(0) != 0) continue;
			if(i->stepb(0) % w1 != 0) continue;
			if(i->stepa(1) % w2 != 0) continue;

			register size_t k5 = i->stepa(1) / w2;
			if(k5_min == 0 || k5_min > k5) {
				k5_min = k5; i1 = i;
			}
		}
		if(i1 != loop.end()) {
			m_kernelname = "ij_pj_ip";
			//~ std::cout << m_kernelname;
			i1->fn() = &loop_list_mul::fn_ij_pj_ip;
			args_ij_pj_ip &args = m_ij_pj_ip;
			args.d = d;
			args.ni = i1->weight();
			args.nj = w1;
			args.np = w2;
			args.sib = i1->stepa(1);
			args.sic = i1->stepb(0);
			args.spa = k2w1;
			loop.splice(loop.end(), loop, i1);
			return;
		}
	}

	//	2. Minimize k4:
	//	------------------------
	//	w   a      b       c
	//	w1  1      0       1
	//	w2  k2*w1  k3'*w3  0
	//	w3  0      1       k4*w1  --> c_j#i = a_p$i b_p%j
	//	------------------------      sz(i) = w1, sz(j) = w3,
	//	                              sz(p) = w2,
	//	                              sz(#) = k4, sz($) = k2,
	//	                              sz(%) = k3'
	//	                              [ij_pj_pi]
	//
	size_t k4_min = 0;
	iterator_t i2 = loop.end();
	for(iterator_t i = loop.begin(); i != loop.end(); i++) {
		if(i->stepa(0) != 0 || i->stepa(1) != 1) continue;
		if(k3 % i->weight() != 0) continue;
		if(i->stepb(0) % w1 != 0) continue;

		register size_t k4 = i->stepb(0) / w1;
		if(k4_min == 0 || k4_min > k4) {
			k4_min = k4; i2 = i;
		}
	}
	if(i2 != loop.end()) {
		m_kernelname = "ij_pj_pi";
		//~ std::cout << m_kernelname;
		i2->fn() = &loop_list_mul::fn_ij_pj_pi;
		args_ij_pj_pi &args = m_ij_pj_pi;
		args.d = d;
		args.ni = i2->weight();
		args.nj = w1;
		args.np = w2;
		args.sic = i2->stepb(0);
		args.spa = k2w1;
		args.spb = k3;
		loop.splice(loop.end(), loop, i2);
		return;
	}
}


void loop_list_mul::match_dgemv_t_b1_l3(list_t &loop, double d, size_t w1,
	size_t w2, size_t k1, size_t k2w1) {

	if(loop.size() < 3) return;

	//	Found pattern:
	//	----------------
	//	w   a  b      c
	//	w1  0  1      k1
	//	w2  1  k2*w1  0   -->  c_i# = a_p b_p%i
	//	----------------       sz(i) = w1, sz(p) = w2
	//	                       sz(#) = k1, sz(%) = k2
	//	                       [dgemv_t_b]
	//

	//	1. Minimize k4:
	//	------------------------
	//	w   a      b      c
	//	w1  0      1      k1'*w3
	//	w2  1      k2*w1  0
	//	w3  k4*w2  0      1       -->  c_i#j = a_j%p b_p$i
	//	------------------------       sz(i) = w1, sz(j) = w3,
	//	                               sz(p) = w2
	//	                               sz(#) = k1', sz($) = k2,
	//	                               sz(%) = k4
	//	                               [ij_jp_pi]
	//
	size_t k4_min = 0;
	iterator_t i1 = loop.end();
	for(iterator_t i = loop.begin(); i != loop.end(); i++) {
		if(i->stepa(1) != 0 || i->stepb(0) != 1) continue;
		if(k1 % i->weight() != 0) continue;
		if(i->stepa(0) % w2 != 0) continue;

		register size_t k4 = i->stepa(0) / w2;
		if(k4_min == 0 || k4_min > k4) {
			k4_min = k4; i1 = i;
		}
	}
	if(i1 != loop.end()) {
		m_kernelname = "ij_jp_pi";
		//~ std::cout << m_kernelname;
		i1->fn() = &loop_list_mul::fn_ij_jp_pi;
		args_ij_jp_pi &args = m_ij_jp_pi;
		args.d = d;
		args.ni = w1;
		args.nj = i1->weight();
		args.np = w2;
		args.sic = k1;
		args.sja = i1->stepa(0);
		args.spb = k2w1;
		loop.splice(loop.end(), loop, i1);
		return;
	}
}


void loop_list_mul::match_dgemv_t_b2_l3(list_t &loop, double d, size_t w1,
	size_t w2, size_t k2w1, size_t k3) {

	if(loop.size() < 3) return;

	//	Found pattern:
	//	----------------
	//	w   a   b      c
	//	w1  0   1      1
	//	w2  k3  k2w1   0  -->  c_i = a_p$ b_p%i
	//	----------------       sz(i) = w1, sz(p) = w2,
	//	                       sz($) = k3, sz(%) = k2
	//	                       [dgemv_t_b]
	//

	//	1. If k3 == 1, minimize k5:
	//	-----------------------
	//	w   a      b      c
	//	w1  0      1      1
	//	w2  1      k2*w1  0
	//	w3  k5*w2  0      k6*w1  --> c_j#i = a_j$p b_p%i
	//	-----------------------      sz(i) = w1, sz(j) = w3, sz(p) = w2
	//	                             sz(#) = k6, sz($) = k5, sz(%) = k2
	//	                             [ij_ip_pj]
	//
	if(k3 == 1) {
		size_t k5_min = 0;
		iterator_t i1 = loop.end();
		for(iterator_t i = loop.begin(); i != loop.end(); i++) {
			if(i->stepa(1) != 0) continue;
			if(i->stepb(0) % w1 != 0) continue;
			if(i->stepa(0) % w2 != 0) continue;

			register size_t k5 = i->stepa(0) / w2;
			if(k5_min == 0 || k5_min > k5) {
				k5_min = k5; i1 = i;
			}
		}
		if(i1 != loop.end()) {
			m_kernelname = "ij_ip_pj";
			//~ std::cout << m_kernelname;
			i1->fn() = &loop_list_mul::fn_ij_ip_pj;
			args_ij_ip_pj &args = m_ij_ip_pj;
			args.d = d;
			args.ni = i1->weight();
			args.nj = w1;
			args.np = w2;
			args.sia = i1->stepa(0);
			args.sic = i1->stepb(0);
			args.spb = k2w1;
			loop.splice(loop.end(), loop, i1);
			return;
		}
	}

	//	2. Minimize k4:
	//	-----------------------
	//	w   a       b     c
	//	w1  0       1     1
	//	w2  k3'*w3  k2w1  0
	//	w3  1       0     k4*w1  -->  c_j#i = a_p$j b_p%i
	//	-----------------------       sz(i) = w1, sz(j) = w3,
	//	                              sz(p) = w2
	//	                              sz(#) = k4, sz($) = k3',
	//	                              sz(%) = k2
	//	                              [ij_pi_pj]
	//
	size_t k4_min = 0;
	iterator_t i2 = loop.end();
	for(iterator_t i = loop.begin(); i != loop.end(); i++) {
		if(i->stepa(0) != 1 || i->stepa(1) != 0) continue;
		if(k3 % i->weight() != 0) continue;
		if(i->stepb(0) % w1 != 0) continue;

		register size_t k4 = i->stepb(0) / w1;
		if(k4_min == 0 || k4_min > k4) {
			k4_min = k4; i2 = i;
		}
	}
	if(i2 != loop.end()) {
		m_kernelname = "ij_pi_pj";
		//~ std::cout << m_kernelname;
		i2->fn() = &loop_list_mul::fn_ij_pi_pj;
		args_ij_pi_pj &args = m_ij_pi_pj;
		args.d = d;
		args.ni = i2->weight();
		args.nj = w1;
		args.np = w2;
		args.sic = i2->stepb(0);
		args.spa = k3;
		args.spb = k2w1;
		loop.splice(loop.end(), loop, i2);
		return;
	}
}


void loop_list_mul::match_x_pq_qp(list_t &loop, double d, size_t np, size_t nq,
	size_t spa, size_t sqb) {

	if(loop.size() < 3) return;

	//	Found pattern:
	//	---------------
	//	w   a    b    c
	//	np  spa  1    0
	//	nq  1    sqb  0  -->  c = a_p$q b_q%p
	//	---------------       sz(p) = np, sz(q) = nq, sz($q) = spa,
	//	                      sz(%p) = sqb
	//	                      [x_pq_qp]
	//

	//	1. Minimize k1a:
	//	-------------------
	//	w   a        b    c
	//	np  spa      1    0
	//	nq  1        sqb  0
	//	ni  k1a*spa  0    1  -->  c_i = a_i@p$q b_q%p
	//	-------------------       [i_ipq_qp]
	//
	//	2. Minimize k1b:
	//	-------------------
	//	w   a    b        c
	//	np  spa  1        0
	//	nq  1    sqb      0
	//	ni  0    k1b*sqb  1  -->  c_i = a_p$q b_i#q%p
	//	-------------------       [i_pq_iqp]

	iterator_t i1 = loop.end(), i2 = loop.end();
	size_t k1a_min = 0, k1b_min = 0;
	for(iterator_t i = loop.begin(); i != loop.end(); i++) {
		if(i->stepa(0) > 0 && i->stepa(0) % spa == 0
			&& i->stepa(1) == 0 && i->stepb(0) == 1) {

			register size_t k1a = i->stepa(0) / spa;
			if(k1a_min == 0 || k1a_min > k1a) {
				i1 = i; k1a_min = k1a;
			}
		}
		if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepa(1) % sqb == 0
			&& i->stepb(0) == 1) {

			register size_t k1b = i->stepa(1) / sqb;
			if(k1b_min == 0 || k1b_min > k1b) {
				i2 = i; k1b_min = k1b;
			}
		}
	}

	if(i1 != loop.end()) {
		m_kernelname = "i_ipq_qp";
		//~ std::cout << m_kernelname << ";";
		i1->fn() = &loop_list_mul::fn_i_ipq_qp;
		args_i_ipq_qp &args = m_i_ipq_qp;
		args.d = d;
		args.ni = i1->weight();
		args.np = np;
		args.nq = nq;
		args.sia = i1->stepa(0);
		args.sic = 1;
		args.spa = spa;
		args.sqb = sqb;
		match_i_ipq_qp(loop, d, i1->weight(), np, nq, i1->stepa(0),
			spa, sqb);
		loop.splice(loop.end(), loop, i1);
		return;
	}

	if(i2 != loop.end()) {
		m_kernelname = "i_pq_iqp";
		//~ std::cout << m_kernelname << ";";
		i2->fn() = &loop_list_mul::fn_i_pq_iqp;
		args_i_pq_iqp &args = m_i_pq_iqp;
		args.d = d;
		args.ni = i2->weight();
		args.np = np;
		args.nq = nq;
		args.sib = i2->stepa(1);
		args.sic = 1;
		args.spa = spa;
		args.sqb = sqb;
		match_i_pq_iqp(loop, d, i2->weight(), np, nq, i2->stepa(1),
			spa, sqb);
		loop.splice(loop.end(), loop, i2);
		return;
	}
}


void loop_list_mul::match_i_ipq_qp(list_t &loop, double d, size_t nj, size_t np,
	size_t nq, size_t sja, size_t spa, size_t sqb) {

	if(loop.size() < 4) return;

	//	Found pattern:
	//	-------------------
	//	w   a        b    c
	//	np  spa      1    0
	//	nq  1        sqb  0
	//	nj  k1a*spa  0    1  -->  c_j = a_j@p$q b_q%p
	//	-------------------       [i_ipq_qp]

	//	1. Minimize x1:
	//	----------------------
	//	w   a    b       c
	//	np  spa  1       0
	//	nq  1    sqb     0
	//	nj  sja  0       1
	//	ni  0    x1*sqb  x2*nj  -->  c_i&j = a_j@p$q b_i#q%p
	//	----------------------       [ij_jpq_iqp]

	iterator_t i1 = loop.end();
	size_t x1_min = 0;
	for(iterator_t i = loop.begin(); i != loop.end(); i++) {

		if(i->stepa(0) != 0) continue;
		if(i->stepa(1) == 0 || i->stepa(1) % sqb != 0) continue;
		if(i->stepb(0) == 0 || i->stepb(0) % nj != 0) continue;

		register size_t x1 = i->stepa(1) / sqb;
		if(x1_min == 0 || x1_min > x1) {
			i1 = i; x1_min = x1;
		}
	}

	if(i1 != loop.end()) {
		m_kernelname = "ij_jpq_iqp";
		//~ std::cout << m_kernelname;
		i1->fn() = &loop_list_mul::fn_ij_jpq_iqp;
		args_ij_jpq_iqp &args = m_ij_jpq_iqp;
		args.d = d;
		args.ni = i1->weight();
		args.nj = nj;
		args.np = np;
		args.nq = nq;
		args.sib = i1->stepa(1);
		args.sic = i1->stepb(0);
		args.sja = sja;
		args.spa = spa;
		args.sqb = sqb;
		loop.splice(loop.end(), loop, i1);
		return;
	}
}


void loop_list_mul::match_i_pq_iqp(list_t &loop, double d, size_t ni, size_t np,
	size_t nq, size_t sib, size_t spa, size_t sqb) {

	if(loop.size() < 4) return;

	//	Found pattern:
	//	---------------
	//	w   a    b    c
	//	np  spa  1    0
	//	nq  1    sqb  0
	//	ni  0    sib  1  -->  c_i = a_p$q b_i#q%p
	//	---------------       [i_pq_iqp]

	//	1. Minimize x1:
	//	----------------------
	//	w   a       b    c
	//	np  spa     1    0
	//	nq  1       sqb  0
	//	ni  0       sib  1
	//	nj  x1*spa  0    x2*ni  -->  c_j&i = a_j@p$q b_i#q%p
	//	----------------------       [ij_ipq_jqp]

	iterator_t i1 = loop.end();
	size_t x1_min = 0;
	for(iterator_t i = loop.begin(); i != loop.end(); i++) {

		if(i->stepa(0) == 0 || i->stepa(0) % spa != 0) continue;
		if(i->stepa(1) != 0) continue;
		if(i->stepb(0) == 0 || i->stepb(0) % ni != 0) continue;

		register size_t x1 = i->stepa(0) / spa;
		if(x1_min == 0 || x1_min > x1) {
			i1 = i; x1_min = x1;
		}
	}

	if(i1 != loop.end()) {
		m_kernelname = "ij_ipq_jqp";
		//~ std::cout << m_kernelname;
		i1->fn() = &loop_list_mul::fn_ij_ipq_jqp;
		args_ij_ipq_jqp &args = m_ij_ipq_jqp;
		args.d = d;
		args.ni = i1->weight();
		args.nj = ni;
		args.np = np;
		args.nq = nq;
		args.sia = i1->stepa(0);
		args.sic = i1->stepb(0);
		args.sjb = sib;
		args.spa = spa;
		args.sqb = sqb;
		loop.splice(loop.end(), loop, i1);
		return;
	}
}


void loop_list_mul::fn_generic(registers &r) const {
	for(size_t i = 0; i < m_generic.m_n; i++) {
		r.m_ptrb[0][i * m_generic.m_stepc] +=
			r.m_ptra[0][i * m_generic.m_stepa] *
			r.m_ptra[1][i * m_generic.m_stepb] *
			m_generic.m_d;
	}
}


void loop_list_mul::fn_x_p_p(registers &r) const {

	static const char *method = "fn_x_p_p(registers&)";

	const args_x_p_p &args = m_x_p_p;

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = (args.np - 1) * args.spa;
	if(r.m_ptra[0] + sz >= r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = (args.np - 1) * args.spb;
	if(r.m_ptra[1] + sz >= r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	if(r.m_ptrb[0] >= r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	*r.m_ptrb[0] += args.d * linalg::x_p_p(r.m_ptra[0], r.m_ptra[1],
		args.np, args.spa, args.spb);
}


void loop_list_mul::fn_x_pq_qp(registers &r) const {

	static const char *method = "fn_x_pq_qp(registers&)";

	const args_x_pq_qp &args = m_x_pq_qp;

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = (args.np - 1) * args.spa + args.nq;
	if(r.m_ptra[0] + sz > r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = (args.nq - 1) * args.sqb + args.np;
	if(r.m_ptra[1] + sz > r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	if(r.m_ptrb[0] >= r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	*r.m_ptrb[0] += args.d * linalg::x_pq_qp(r.m_ptra[0], r.m_ptra[1],
		args.np, args.nq, args.spa, args.sqb);
}


void loop_list_mul::fn_i_i_x(registers &r) const {

	static const char *method = "fn_i_i_x(registers&)";

	const args_i_i_x &args = m_i_i_x;

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = args.ni;
	if(r.m_ptra[0] + sz > r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	if(r.m_ptra[1] >= r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	sz = (args.ni - 1) * args.sic + 1;
	if(r.m_ptrb[0] + sz > r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	linalg::i_i_x(r.m_ptra[0], *r.m_ptra[1] * args.d, r.m_ptrb[0],
		args.ni, 1, args.sic);
}


void loop_list_mul::fn_i_x_i(registers &r) const {

	static const char *method = "fn_i_x_i(registers&)";

	const args_i_x_i &args = m_i_x_i;

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	if(r.m_ptra[0] >= r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = args.ni;
	if(r.m_ptra[1] + sz > r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	sz = (args.ni - 1) * args.sic + 1;
	if(r.m_ptrb[0] + sz > r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	linalg::i_i_x(r.m_ptra[1], *r.m_ptra[0] * args.d, r.m_ptrb[0],
		args.ni, 1, args.sic);
}


void loop_list_mul::fn_i_ip_p(registers &r) const {

	static const char *method = "fn_i_ip_p(registers&)";

	const args_i_ip_p &args = m_i_ip_p;

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = (args.ni - 1) * args.sia + args.np;
	if(r.m_ptra[0] + sz > r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = (args.np - 1) * args.spb + 1;
	if(r.m_ptra[1] + sz > r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	sz = (args.ni - 1) * args.sic + 1;
	if(r.m_ptrb[0] + sz > r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	linalg::i_ip_p(r.m_ptra[0], r.m_ptra[1], r.m_ptrb[0], args.d,
		args.ni, args.np, args.sia, args.sic, args.spb);
}


void loop_list_mul::fn_i_pi_p(registers &r) const {

	static const char *method = "fn_i_pi_p(registers&)";

	const args_i_pi_p &args = m_i_pi_p;

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = (args.np - 1) * args.spa + args.ni;
	if(r.m_ptra[0] + sz > r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = (args.np - 1) * args.spb + 1;
	if(r.m_ptra[1] + sz > r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	sz = (args.ni - 1) * args.sic + 1;
	if(r.m_ptrb[0] + sz > r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	linalg::i_pi_p(r.m_ptra[0], r.m_ptra[1], r.m_ptrb[0], args.d,
		args.ni, args.np, args.sic, args.spa, args.spb);
}


void loop_list_mul::fn_i_p_ip(registers &r) const {

	static const char *method = "fn_i_p_ip(registers&)";

	const args_i_p_ip &args = m_i_p_ip;

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = (args.np - 1) * args.spa + 1;
	if(r.m_ptra[0] + sz > r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = (args.ni - 1) * args.sib + args.np;
	if(r.m_ptra[1] + sz > r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	sz = (args.ni - 1) * args.sic + 1;
	if(r.m_ptrb[0] + sz > r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	linalg::i_ip_p(r.m_ptra[1], r.m_ptra[0], r.m_ptrb[0], args.d,
		args.ni, args.np, args.sib, args.sic, args.spa);
}


void loop_list_mul::fn_i_p_pi(registers &r) const {

	static const char *method = "fn_i_p_pi(registers&)";

	const args_i_p_pi &args = m_i_p_pi;

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = (args.np - 1) * args.spa + 1;
	if(r.m_ptra[0] + sz > r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = (args.np - 1) * args.spb + args.ni;
	if(r.m_ptra[1] + sz > r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	sz = (args.ni - 1) * args.sic + 1;
	if(r.m_ptrb[0] + sz > r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	linalg::i_pi_p(r.m_ptra[1], r.m_ptra[0], r.m_ptrb[0], args.d,
		args.ni, args.np, args.sic, args.spb, args.spa);
}


void loop_list_mul::fn_ij_ip_pj(registers &r) const {

	static const char *method = "fn_ij_ip_pj(registers&)";

	const args_ij_ip_pj &args = m_ij_ip_pj;

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = (args.ni - 1) * args.sia + args.np;
	if(r.m_ptra[0] + sz > r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = (args.np - 1) * args.spb + args.nj;
	if(r.m_ptra[1] + sz > r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	sz = (args.ni - 1) * args.sic + args.nj;
	if(r.m_ptrb[0] + sz > r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	linalg::ij_ip_pj(r.m_ptra[0], r.m_ptra[1], r.m_ptrb[0], args.d,
		args.ni, args.nj, args.np, args.sia, args.sic, args.spb);
}


void loop_list_mul::fn_ij_ip_jp(registers &r) const {

	static const char *method = "fn_ij_ip_jp(registers&)";

	const args_ij_ip_jp &args = m_ij_ip_jp;

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = (args.ni - 1) * args.sia + args.np;
	if(r.m_ptra[0] + sz > r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = (args.nj - 1) * args.sjb + args.np;
	if(r.m_ptra[1] + sz > r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	sz = (args.ni - 1) * args.sic + args.nj;
	if(r.m_ptrb[0] + sz > r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	linalg::ij_ip_jp(r.m_ptra[0], r.m_ptra[1], r.m_ptrb[0], args.d,
		args.ni, args.nj, args.np, args.sia, args.sic, args.sjb);
}


void loop_list_mul::fn_ij_pi_pj(registers &r) const {

	static const char *method = "fn_ij_pi_pj(registers&)";

	const args_ij_pi_pj &args = m_ij_pi_pj;

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = (args.np - 1) * args.spa + args.ni;
	if(r.m_ptra[0] + sz > r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = (args.np - 1) * args.spb + args.nj;
	if(r.m_ptra[1] + sz > r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	sz = (args.ni - 1) * args.sic + args.nj;
	if(r.m_ptrb[0] + sz > r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	linalg::ij_pi_pj(r.m_ptra[0], r.m_ptra[1], r.m_ptrb[0], args.d,
		args.ni, args.nj, args.np, args.sic, args.spa, args.spb);
}


void loop_list_mul::fn_ij_pi_jp(registers &r) const {

	static const char *method = "fn_ij_pi_jp(registers&)";

	const args_ij_pi_jp &args = m_ij_pi_jp;

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = (args.np - 1) * args.spa + args.ni;
	if(r.m_ptra[0] + sz > r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = (args.nj - 1) * args.sjb + args.np;
	if(r.m_ptra[1] + sz > r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	sz = (args.ni - 1) * args.sic + args.nj;
	if(r.m_ptrb[0] + sz > r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	linalg::ij_pi_jp(r.m_ptra[0], r.m_ptra[1], r.m_ptrb[0], args.d,
		args.ni, args.nj, args.np, args.sic, args.sjb, args.spa);
}


void loop_list_mul::fn_ij_pj_ip(registers &r) const {

	static const char *method = "fn_ij_pj_ip(registers&)";

	const args_ij_pj_ip &args = m_ij_pj_ip;

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = (args.np - 1) * args.spa + args.nj;
	if(r.m_ptra[0] + sz > r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = (args.ni - 1) * args.sib + args.np;
	if(r.m_ptra[1] + sz > r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	sz = (args.ni - 1) * args.sic + args.nj;
	if(r.m_ptrb[0] + sz > r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	linalg::ij_ip_pj(r.m_ptra[1], r.m_ptra[0], r.m_ptrb[0], args.d,
		args.ni, args.nj, args.np, args.sib, args.sic, args.spa);
}


void loop_list_mul::fn_ij_jp_ip(registers &r) const {

	static const char *method = "fn_ij_jp_ip(registers&)";
	
	const args_ij_jp_ip &args = m_ij_jp_ip;

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = (args.nj - 1) * args.sja + args.np;
	if(r.m_ptra[0] + sz > r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = (args.ni - 1) * args.sib + args.np;
	if(r.m_ptra[1] + sz > r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	sz = (args.ni - 1) * args.sic + args.nj;
	if(r.m_ptrb[0] + sz > r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	linalg::ij_ip_jp(r.m_ptra[1], r.m_ptra[0], r.m_ptrb[0], args.d,
		args.ni, args.nj, args.np, args.sib, args.sic, args.sja);
}


void loop_list_mul::fn_ij_pj_pi(registers &r) const {

	static const char *method = "fn_ij_pj_pi(registers&)";

	const args_ij_pj_pi &args = m_ij_pj_pi;

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = (args.np - 1) * args.spa + args.nj;
	if(r.m_ptra[0] + sz > r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = (args.np - 1) * args.spb + args.ni;
	if(r.m_ptra[1] + sz > r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	sz = (args.ni - 1) * args.sic + args.nj;
	if(r.m_ptrb[0] + sz > r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	linalg::ij_pi_pj(r.m_ptra[1], r.m_ptra[0], r.m_ptrb[0], args.d,
		args.ni, args.nj, args.np, args.sic, args.spb, args.spa);
}


void loop_list_mul::fn_ij_jp_pi(registers &r) const {

	static const char *method = "fn_ij_jp_pi(registers&)";

	const args_ij_jp_pi &args = m_ij_jp_pi;

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = (args.nj - 1) * args.sja + args.np;
	if(r.m_ptra[0] + sz > r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = (args.np - 1) * args.spb + args.ni;
	if(r.m_ptra[1] + sz > r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	sz = (args.ni - 1) * args.sic + args.nj;
	if(r.m_ptrb[0] + sz > r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	linalg::ij_pi_jp(r.m_ptra[1], r.m_ptra[0], r.m_ptrb[0], args.d,
		args.ni, args.nj, args.np, args.sic, args.sja, args.spb);
}


void loop_list_mul::fn_i_ipq_qp(registers &r) const {

	static const char *method = "fn_i_ipq_qp(registers&)";

	const args_i_ipq_qp &args = m_i_ipq_qp;

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = (args.ni - 1) * args.sia + (args.np - 1) * args.spa + args.nq;
	if(r.m_ptra[0] + sz > r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = (args.nq - 1) * args.sqb + args.np;
	if(r.m_ptra[1] + sz > r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	sz = (args.ni - 1) * args.sic + 1;
	if(r.m_ptrb[0] + sz > r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	linalg::i_ipq_qp(r.m_ptra[0], r.m_ptra[1], r.m_ptrb[0], args.d,
		args.ni, args.np, args.nq, args.sia, args.sic, args.spa,
		args.sqb);
}


void loop_list_mul::fn_i_pq_iqp(registers &r) const {

	static const char *method = "fn_i_pq_iqp(registers&)";

	const args_i_pq_iqp &args = m_i_pq_iqp;

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = (args.np - 1) * args.spa + args.nq;
	if(r.m_ptra[0] + sz > r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = (args.ni - 1) * args.sib + (args.nq - 1) * args.sqb + args.np;
	if(r.m_ptra[1] + sz > r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	sz = (args.ni - 1) * args.sic + 1;
	if(r.m_ptrb[0] + sz > r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	linalg::i_ipq_qp(r.m_ptra[1], r.m_ptra[0], r.m_ptrb[0], args.d,
		args.ni, args.nq, args.np, args.sib, args.sic, args.sqb,
		args.spa);
}


void loop_list_mul::fn_ij_ipq_jqp(registers &r) const {

	static const char *method = "fn_ij_ipq_jqp(registers&)";

	const args_ij_ipq_jqp &args = m_ij_ipq_jqp;

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = (args.ni - 1) * args.sia + (args.np - 1) * args.spa + args.nq;
	if(r.m_ptra[0] + sz > r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = (args.nj - 1) * args.sjb + (args.nq - 1) * args.sqb + args.np;
	if(r.m_ptra[1] + sz > r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	sz = (args.ni - 1) * args.sic + args.nj;
	if(r.m_ptrb[0] + sz > r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	linalg::ij_ipq_jqp(r.m_ptra[0], r.m_ptra[1], r.m_ptrb[0], args.d,
		args.ni, args.nj, args.np, args.nq, args.sia, args.sic,
		args.sjb, args.spa, args.sqb);
}


void loop_list_mul::fn_ij_jpq_iqp(registers &r) const {

	static const char *method = "fn_ij_jpq_iqp(registers&)";

	const args_ij_jpq_iqp &args = m_ij_jpq_iqp;

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = (args.nj - 1) * args.sja + (args.np - 1) * args.spa + args.nq;
	if(r.m_ptra[0] + sz > r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = (args.ni - 1) * args.sib + (args.nq - 1) * args.sqb + args.np;
	if(r.m_ptra[1] + sz > r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	sz = (args.ni - 1) * args.sic + args.nj;
	if(r.m_ptrb[0] + sz > r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	// switch a<->b: c_ij = b_iqp a_jpq
	// rename p<->q: c_ij = b_ipq a_jqp
	// therefore: ni := ni, nj := nj
	//            sia := sib, sic := sic, sjb = sja,
	//            spa := sqb, sqb := spa
	linalg::ij_ipq_jqp(r.m_ptra[1], r.m_ptra[0], r.m_ptrb[0], args.d,
		args.ni, args.nj, args.nq, args.np, args.sib, args.sic,
		args.sja, args.sqb, args.spa);
}


} // namespace libtensor
