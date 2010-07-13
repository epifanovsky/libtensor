#include "../defs.h"
#include "../exception.h"
#include "../blas.h"
#include "../linalg.h"
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

	//	1. Minimize k1 > 0:
	//	------------
	//	w   a   b  c
	//	w1  k1  1  0  -->  c_# = a_p# b_p
	//	------------       sz(p) = w1, sz(#) = k1
	//	                   [ddot]
	//
	//	2. Minimize k1a:
	//	-------------
	//	w   a  b  c
	//	w1  1  0  k1a  -->  c_i# = a_i b
	//	-------------       sz(i) = w1, sz(#) = k1a
	//	                    [daxpy_a]
	//
	//	3. Minimize k1b:
	//	-------------
	//	w   a  b  c
	//	w1  0  1  k1b  -->  c_i# = a b_i
	//	-------------       sz(i) = w1, sz(#) = k1b
	//	                    [daxpy_b]
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
		//~ std::cout << "ddot1";
		m_kernelname = "ddot1";
		i1->fn() = &loop_list_mul::fn_ddot;
		m_ddot.m_d = d;
		m_ddot.m_n = i1->weight();
		m_ddot.m_stepa = i1->stepa(0);
		m_ddot.m_stepb = 1;
		match_ddot_l2(loop, d, i1->weight(), i1->stepa(0));
		loop.splice(loop.end(), loop, i1);
		return;
	}
	if(i2 != loop.end() && k1b_min != 1) {
		//~ std::cout << "daxpy_a";
		m_kernelname = "daxpy_a";
		i2->fn() = &loop_list_mul::fn_daxpy_a;
		m_daxpy_a.m_d = d;
		m_daxpy_a.m_n = i2->weight();
		m_daxpy_a.m_stepc = i2->stepb(0);
		match_daxpy_a_l2(loop, d, i2->weight(), i2->stepb(0));
		loop.splice(loop.end(), loop, i2);
		return;
	}
	if(i3 != loop.end()) {
		//~ std::cout << "daxpy_b";
		m_kernelname = "daxpy_b";
		i3->fn() = &loop_list_mul::fn_daxpy_b;
		m_daxpy_b.m_d = d;
		m_daxpy_b.m_n = i3->weight();
		m_daxpy_b.m_stepc = i3->stepb(0);
		match_daxpy_b_l2(loop, d, i3->weight(), i3->stepb(0));
		loop.splice(loop.end(), loop, i3);
		return;
	}
	if(i1 != loop.end()) {
		//~ std::cout << "ddot2";
		m_kernelname = "ddot2";
		i1->fn() = &loop_list_mul::fn_ddot;
		m_ddot.m_d = d;
		m_ddot.m_n = i1->weight();
		m_ddot.m_stepa = i1->stepa(0);
		m_ddot.m_stepb = 1;
		match_ddot_l2(loop, d, i1->weight(), i1->stepa(0));
		loop.splice(loop.end(), loop, i1);
		return;
	}

	iterator_t i0 = loop.begin();
	if(i0 == loop.end()) return;

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


void loop_list_mul::match_ddot_l2(list_t &loop, double d, size_t w1,
	size_t k1) {

	//	Found pattern (k1 > 0):
	//	------------
	//	w   a   b  c
	//	w1  k1  1  0  -->  c_# = a_p# b_p
	//	------------       sz(p) = w1, sz(#) = k1
	//	                   [ddot]
	//

	//	1. If k1 == 1, minimize k2a:
	//	----------------
	//	w   a       b  c
	//	w1  1       1  0
	//	w2  k2a*w1  0  1  -->  c_i = a_i$p b_p
	//	----------------       sz(i) = w2, sz(p) = w1, sz($) = k2a
	//	                       [dgemv_n_a]
	//
	//	2. If k1 == 1, minimize k2b:
	//	----------------
	//	w   a  b       c
	//	w1  1  1       0
	//	w2  0  k2b*w1  1  -->  c_i = a_p b_i%p
	//	----------------       sz(i) = w2, sz(p) = w1, sz(%) = k2b
	//	                       [dgemv_n_b]
	//
	//	3. Minimize k2c:
	//	---------------------
	//	w   a       b       c
	//	w1  k1'*w2  1       0
	//	w2  1       k2c*w1  0  -->  c = a_p$q b_q%p
	//	---------------------       sz(p) = w1, sz(q) = w2, sz($) = k1',
	//	                            sz(%) = k2c
	//	                            [ddot_trp]
	//
	size_t k2a_min = 0, k2b_min = 0, k2c_min = 0;
	iterator_t i1 = loop.end(), i2 = loop.end(), i3 = loop.end();
	for(iterator_t i = loop.begin(); i != loop.end(); i++) {
		if(i->stepb(0) > 1) continue;
		if(k1 == 1 && i->stepb(0) == 1) {
			if(i->stepa(1) == 0 && i->stepa(0) % w1 == 0) {
				register size_t k2a = i->stepa(0) / w1;
				if(k2a_min == 0 || k2a_min > k2a) {
					k2a_min = k2a; i1 = i;
				}
			}
			if(i->stepa(0) == 0 && i->stepa(1) % w1 == 0) {
				register size_t k2b = i->stepa(1) / w1;
				if(k2b_min == 0 || k2b_min > k2b) {
					k2b_min = k2b; i2 = i;
				}
			}
		}
		if(i->stepa(0) == 1 && i->stepb(0) == 0
			&& k1 % i->weight() == 0) {
			register size_t k2c = i->stepa(1) / w1;
			if(k2c_min == 0 || k2c_min > k2c) {
				k2c_min = k2c; i3 = i;
			}
		}
	}

	if(i1 != loop.end()) {
		//~ std::cout << " dgemv_n_a";
		m_kernelname = "dgemv_n_a";
		i1->fn() = &loop_list_mul::fn_dgemv_n_a;
		m_dgemv_n_a.m_d = d;
		m_dgemv_n_a.m_rows = i1->weight();
		m_dgemv_n_a.m_cols = w1;
		m_dgemv_n_a.m_stepb = 1;
		m_dgemv_n_a.m_lda = i1->stepa(0);
		match_dgemv_n_a_l3(loop, d, w1, i1->weight(), i1->stepa(0));
		loop.splice(loop.end(), loop, i1);
		return;
	}

	if(i2 != loop.end()) {
		//~ std::cout << " dgemv_n_b";
		m_kernelname = "dgemv_n_b";
		i2->fn() = &loop_list_mul::fn_dgemv_n_b;
		m_dgemv_n_b.m_d = d;
		m_dgemv_n_b.m_rows = i2->weight();
		m_dgemv_n_b.m_cols = w1;
		m_dgemv_n_b.m_stepa = 1;
		m_dgemv_n_b.m_ldb = i2->stepa(1);
		match_dgemv_n_b_l3(loop, d, w1, i2->weight(), i2->stepa(1));
		loop.splice(loop.end(), loop, i2);
		return;
	}

	if(i3 != loop.end()) {
		//~ std::cout << " ddot_trp";
		m_kernelname = "ddot_trp";
		i3->fn() = &loop_list_mul::fn_ddot_trp;
		m_ddot_trp.m_d = d;
		m_ddot_trp.m_ni = w1;
		m_ddot_trp.m_nj = i3->weight();
		m_ddot_trp.m_lda = k1;
		m_ddot_trp.m_ldb = i3->stepa(1);
		loop.splice(loop.end(), loop, i3);
		return;
	}
}


void loop_list_mul::match_daxpy_a_l2(list_t &loop, double d, size_t w1,
	size_t k1) {

	//	Found pattern:
	//	------------
	//	w   a  b  c
	//	w1  1  0  k1  -->  c_i# = a_i b
	//	------------       sz(i) = w1, sz(#) = k1
	//	                   [daxpy_a]
	//

	//	1. Minimize k2a:
	//	------------------
	//	w   a       b   c
	//	w1  1       0   k1
	//	w2  k2a*w1  1   0   -->  c_i# = a_p$i b_p
	//	------------------       sz(i) = w1, sz(p) = w2
	//	                         sz(#) = k1, sz($) = k2a
	//	                         [dgemv_t_a]
	//
	//	2. Minimize k2b:
	//	------------------
	//	w   a       b   c
	//	w1  1       0   k1
	//	w2  k2b*w1  k3  0   -->  c_i# = a_p$i b_p%
	//	------------------       sz(i) = w1, sz(p) = w2
	//	                         sz(#) = k1, sz($) = k2b, sz(%) = k3
	//	                         [dgemv_t_a]
	//	-----------------
	//	w   a       b   c
	//	w1  1       0   1
	//	w2  k2b*w1  k3  0  -->  c_i = a_p$i b_p%
	//	-----------------       sz(i) = w1, sz(p) = w2
	//	                        sz($) = k2b, sz(%) = k3
	//	                        [dgemv_t_a]
	//
	size_t k2a_min = 0, k2b_min = 0;
	iterator_t i1 = loop.end(), i2 = loop.end();
	for(iterator_t i = loop.begin(); i != loop.end(); i++) {
		if(i->stepb(0) != 0) continue;
		if(i->stepa(0) % w1 != 0) continue;

		register size_t k2 = i->stepa(0) / w1;
		if(i->stepa(1) == 1 && (k2a_min == 0 || k2a_min > k2)) {
			k2a_min = k2; i1 = i;
		}
		if(k2b_min == 0 || k2b_min > k2) {
			k2b_min = k2; i2 = i;
		}
	}
	if(i1 != loop.end() && !(k1 == 1 && i2 != loop.end())) {
		//~ std::cout << " dgemv_t_a1";
		m_kernelname = "dgemv_t_a";
		i1->fn() = &loop_list_mul::fn_dgemv_t_a;
		m_dgemv_t_a.m_d = d;
		m_dgemv_t_a.m_rows = i1->weight();
		m_dgemv_t_a.m_cols = w1;
		m_dgemv_t_a.m_stepb = 1;
		m_dgemv_t_a.m_lda = i1->stepa(0);
		m_dgemv_t_a.m_stepc = k1;
		match_dgemv_t_a1_l3(loop, d, w1, i1->weight(), k1,
			i1->stepa(0));
		loop.splice(loop.end(), loop, i1);
		return;
	}
	if(i2 != loop.end()) {
		//~ std::cout << " dgemv_t_a2";
		m_kernelname = "dgemv_t_a";
		i2->fn() = &loop_list_mul::fn_dgemv_t_a;
		m_dgemv_t_a.m_d = d;
		m_dgemv_t_a.m_rows = i2->weight();
		m_dgemv_t_a.m_cols = w1;
		m_dgemv_t_a.m_stepb = i2->stepa(1);
		m_dgemv_t_a.m_lda = i2->stepa(0);
		m_dgemv_t_a.m_stepc = k1;
		if(k1 == 1) {
			match_dgemv_t_a2_l3(loop, d, w1, i2->weight(),
				i2->stepa(0), i2->stepa(1));
		}
		loop.splice(loop.end(), loop, i2);
		return;
	}
}


void loop_list_mul::match_daxpy_b_l2(list_t &loop, double d, size_t w1,
	size_t k1) {

	//	Found pattern:
	//	------------
	//	w   a  b  c
	//	w1  0  1  k1  -->  c_i# = a b_i
	//	------------       sz(i) = w1, sz(#) = k1
	//	                   [daxpy_b]

	//	1. Minimize k2a:
	//	-----------------
	//	w   a  b       c
	//	w1  0  1       k1
	//	w2  1  k2a*w1  0   -->  c_i# = a_p b_p%i
	//	-----------------       sz(i) = w1, sz(p) = w2
	//	                        sz(#) = k1, sz(%) = k2a
	//	                        [dgemv_t_b]
	//
	//	2. Minimize k2b:
	//	------------------
	//	w   a   b       c
	//	w1  0   1       k1
	//	w2  k3  k2b*w1  0  -->  c_i# = a_p$ b_p%i
	//      ------------------      sz(i) = w1, sz(p) = w2
	//	                        sz(#) = k1, sz($) = k3, sz(%) = k2b
	//	                        [dgemv_t_b]
	//	-----------------
	//	w   a   b       c
	//	w1  0   1       1
	//	w2  k3  k2b*w1  0  -->  c_i = a_p$ b_p%i
	//	-----------------       sz(i) = w1, sz(p) = w2
	//	                        sz($) = k2b, sz(%) = k3
	//	                        [dgemv_t_b]
	//
	size_t k2a_min = 0, k2b_min = 0;
	iterator_t i1 = loop.end(), i2 = loop.end();
	for(iterator_t i = loop.begin(); i != loop.end(); i++) {
		if(i->stepb(0) != 0) continue;
		if(i->stepa(1) % w1 != 0) continue;

		register size_t k2 = i->stepa(1) / w1;
		if(i->stepa(0) == 1 && (k2a_min == 0 || k2a_min > k2)) {
			k2a_min = k2; i1 = i;
		}
		if(k2b_min == 0 || k2b_min > k2) {
			k2b_min = k2; i2 = i;
		}
	}
	if(i1 != loop.end() && !(k1 == 1 && i2 != loop.end())) {
		//~ std::cout << " dgemv_t_b1";
		m_kernelname = "dgemv_t_b";
		i1->fn() = &loop_list_mul::fn_dgemv_t_b;
		m_dgemv_t_b.m_d = d;
		m_dgemv_t_b.m_rows = i1->weight();
		m_dgemv_t_b.m_cols = w1;
		m_dgemv_t_b.m_stepa = 1;
		m_dgemv_t_b.m_ldb = i1->stepa(1);
		m_dgemv_t_b.m_stepc = k1;
		match_dgemv_t_b1_l3(loop, d, w1, i1->weight(), k1,
			i1->stepa(1));
		loop.splice(loop.end(), loop, i1);
		return;
	}
	if(i2 != loop.end()) {
		//~ std::cout << " dgemv_t_b2";
		m_kernelname = "dgemv_t_b";
		i2->fn() = &loop_list_mul::fn_dgemv_t_b;
		m_dgemv_t_b.m_d = d;
		m_dgemv_t_b.m_rows = i2->weight();
		m_dgemv_t_b.m_cols = w1;
		m_dgemv_t_b.m_stepa = i2->stepa(0);
		m_dgemv_t_b.m_ldb = i2->stepa(1);
		m_dgemv_t_b.m_stepc = k1;
		if(k1 == 1) {
			match_dgemv_t_b2_l3(loop, d, w1, i2->weight(),
				i2->stepa(1), i2->stepa(0));
		}
		loop.splice(loop.end(), loop, i2);
		return;
	}
}


void loop_list_mul::match_dgemv_n_a_l3(list_t &loop, double d, size_t w1,
	size_t w2, size_t k1w1) {

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
	//	                              [dgemm_nt_ba]
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
		//~ std::cout << " dgemm_nt_ba";
		m_kernelname = "dgemm_nt_ba";
		i1->fn() = &loop_list_mul::fn_dgemm_nt_ba;
		m_dgemm_nt_ba.m_d = d;
		m_dgemm_nt_ba.m_rowsb = i1->weight();
		m_dgemm_nt_ba.m_colsa = w2;
		m_dgemm_nt_ba.m_colsb = w1;
		m_dgemm_nt_ba.m_ldb = i1->stepa(1);
		m_dgemm_nt_ba.m_lda = k1w1;
		m_dgemm_nt_ba.m_ldc = i1->stepb(0);
		loop.splice(loop.end(), loop, i1);
		return;
	}
}


void loop_list_mul::match_dgemv_n_b_l3(list_t &loop, double d, size_t w1,
	size_t w2, size_t k1w1) {

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
	//	                              [dgemm_nt_ab]
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
		//~ std::cout << " dgemm_nt_ab";
		m_kernelname = "dgemm_nt_ab";
		i1->fn() = &loop_list_mul::fn_dgemm_nt_ab;
		m_dgemm_nt_ab.m_d = d;
		m_dgemm_nt_ab.m_rowsa = i1->weight();
		m_dgemm_nt_ab.m_colsb = w2;
		m_dgemm_nt_ab.m_colsa = w1;
		m_dgemm_nt_ab.m_lda = i1->stepa(0);
		m_dgemm_nt_ab.m_ldb = k1w1;
		m_dgemm_nt_ab.m_ldc = i1->stepb(0);
		loop.splice(loop.end(), loop, i1);
		return;
	}
}


void loop_list_mul::match_dgemv_t_a1_l3(list_t &loop, double d, size_t w1,
	size_t w2, size_t k1, size_t k2w1) {

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
	//	                               [dgemm_tt_ab]
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
		//~ std::cout << " dgemm_tt_ab";
		m_kernelname = "dgemm_tt_ab";
		i1->fn() = &loop_list_mul::fn_dgemm_tt_ab;
		m_dgemm_tt_ab.m_d = d;
		m_dgemm_tt_ab.m_rowsa = w1;
		m_dgemm_tt_ab.m_colsb = i1->weight();
		m_dgemm_tt_ab.m_colsa = w2;
		m_dgemm_tt_ab.m_lda = k2w1;
		m_dgemm_tt_ab.m_ldb = i1->stepa(1);
		m_dgemm_tt_ab.m_ldc = k1;
		loop.splice(loop.end(), loop, i1);
		return;
	}
}


void loop_list_mul::match_dgemv_t_a2_l3(list_t &loop, double d, size_t w1,
	size_t w2, size_t k2w1, size_t k3) {

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
	//	                             [dgemm_nn_ba]
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
			//~ std::cout << " dgemm_nn_ba";
			m_kernelname = "dgemm_nn_ba";
			i1->fn() = &loop_list_mul::fn_dgemm_nn_ba;
			m_dgemm_nn_ba.m_d = d;
			m_dgemm_nn_ba.m_rowsb = i1->weight();
			m_dgemm_nn_ba.m_colsa = w1;
			m_dgemm_nn_ba.m_colsb = w2;
			m_dgemm_nn_ba.m_ldb = i1->stepa(1);
			m_dgemm_nn_ba.m_lda = k2w1;
			m_dgemm_nn_ba.m_ldc = i1->stepb(0);
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
	//	                              [dgemm_tn_ba]
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
		//~ std::cout << " dgemm_tn_ba";
		m_kernelname = "dgemm_tn_ba";
		i2->fn() = &loop_list_mul::fn_dgemm_tn_ba;
		m_dgemm_tn_ba.m_d = d;
		m_dgemm_tn_ba.m_rowsb = i2->weight();
		m_dgemm_tn_ba.m_colsa = w1;
		m_dgemm_tn_ba.m_colsb = w2;
		m_dgemm_tn_ba.m_ldb = k3;
		m_dgemm_tn_ba.m_lda = k2w1;
		m_dgemm_tn_ba.m_ldc = i2->stepb(0);
		loop.splice(loop.end(), loop, i2);
		return;
	}
}


void loop_list_mul::match_dgemv_t_b1_l3(list_t &loop, double d, size_t w1,
	size_t w2, size_t k1, size_t k2w1) {

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
	//	                               [dgemm_tt_ba]
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
		//~ std::cout << " dgemm_tt_ba";
		m_kernelname = "dgemm_tt_ba";
		i1->fn() = &loop_list_mul::fn_dgemm_tt_ba;
		m_dgemm_tt_ba.m_d = d;
		m_dgemm_tt_ba.m_rowsb = w1;
		m_dgemm_tt_ba.m_colsa = i1->weight();
		m_dgemm_tt_ba.m_colsb = w2;
		m_dgemm_tt_ba.m_ldb = k2w1;
		m_dgemm_tt_ba.m_lda = i1->stepa(0);
		m_dgemm_tt_ba.m_ldc = k1;
		loop.splice(loop.end(), loop, i1);
		return;
	}
}


void loop_list_mul::match_dgemv_t_b2_l3(list_t &loop, double d, size_t w1,
	size_t w2, size_t k2w1, size_t k3) {

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
	//	                             [dgemm_nn_ab]
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
			//~ std::cout << " dgemm_nn_ab";
			m_kernelname = "dgemm_nn_ab";
			i1->fn() = &loop_list_mul::fn_dgemm_nn_ab;
			m_dgemm_nn_ab.m_d = d;
			m_dgemm_nn_ab.m_rowsa = i1->weight();
			m_dgemm_nn_ab.m_colsb = w1;
			m_dgemm_nn_ab.m_colsa = w2;
			m_dgemm_nn_ab.m_lda = i1->stepa(0);
			m_dgemm_nn_ab.m_ldb = k2w1;
			m_dgemm_nn_ab.m_ldc = i1->stepb(0);
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
	//	                              [dgemm_tn_ab]
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
		//~ std::cout << " dgemm_tn_ab";
		m_kernelname = "dgemm_tn_ab";
		i2->fn() = &loop_list_mul::fn_dgemm_tn_ab;
		m_dgemm_tn_ab.m_d = d;
		m_dgemm_tn_ab.m_rowsa = i2->weight();
		m_dgemm_tn_ab.m_colsb = w1;
		m_dgemm_tn_ab.m_colsa = w2;
		m_dgemm_tn_ab.m_lda = k3;
		m_dgemm_tn_ab.m_ldb = k2w1;
		m_dgemm_tn_ab.m_ldc = i2->stepb(0);
		loop.splice(loop.end(), loop, i2);
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


void loop_list_mul::fn_ddot(registers &r) const {

	*r.m_ptrb[0] += m_ddot.m_d * blas_ddot(m_ddot.m_n, r.m_ptra[0],
		m_ddot.m_stepa, r.m_ptra[1], m_ddot.m_stepb);
}


void loop_list_mul::fn_ddot_trp(registers &r) const {

	*r.m_ptrb[0] += m_ddot_trp.m_d * blas::ddot_trp(
		r.m_ptra[0], r.m_ptra[1], m_ddot_trp.m_ni, m_ddot_trp.m_nj,
		m_ddot_trp.m_lda, m_ddot_trp.m_ldb);
}


void loop_list_mul::fn_daxpy_a(registers &r) const {

	blas_daxpy(m_daxpy_a.m_n, *r.m_ptra[1] * m_daxpy_a.m_d, r.m_ptra[0], 1,
		r.m_ptrb[0], m_daxpy_a.m_stepc);
}


void loop_list_mul::fn_daxpy_b(registers &r) const {

	blas_daxpy(m_daxpy_b.m_n, *r.m_ptra[0] * m_daxpy_b.m_d, r.m_ptra[1], 1,
		r.m_ptrb[0], m_daxpy_b.m_stepc);
}


void loop_list_mul::fn_dgemv_n_a(registers &r) const {

	blas_dgemv(false, m_dgemv_n_a.m_rows, m_dgemv_n_a.m_cols,
		m_dgemv_n_a.m_d, r.m_ptra[0], m_dgemv_n_a.m_lda, r.m_ptra[1],
		m_dgemv_n_a.m_stepb, 1.0, r.m_ptrb[0], 1);
}


void loop_list_mul::fn_dgemv_t_a(registers &r) const {

	blas_dgemv(true, m_dgemv_t_a.m_rows, m_dgemv_t_a.m_cols,
		m_dgemv_t_a.m_d, r.m_ptra[0], m_dgemv_t_a.m_lda, r.m_ptra[1],
		m_dgemv_t_a.m_stepb, 1.0, r.m_ptrb[0], m_dgemv_t_a.m_stepc);
}


void loop_list_mul::fn_dgemv_n_b(registers &r) const {

	blas_dgemv(false, m_dgemv_n_b.m_rows, m_dgemv_n_b.m_cols,
		m_dgemv_n_b.m_d, r.m_ptra[1], m_dgemv_n_b.m_ldb, r.m_ptra[0],
		m_dgemv_n_b.m_stepa, 1.0, r.m_ptrb[0], 1);
}


void loop_list_mul::fn_dgemv_t_b(registers &r) const {

	blas_dgemv(true, m_dgemv_t_b.m_rows, m_dgemv_t_b.m_cols,
		m_dgemv_t_b.m_d, r.m_ptra[1], m_dgemv_t_b.m_ldb, r.m_ptra[0],
		m_dgemv_t_b.m_stepa, 1.0, r.m_ptrb[0], m_dgemv_t_b.m_stepc);
}


void loop_list_mul::fn_dgemm_nn_ab(registers &r) const {

	static const char *method = "fn_dgemm_nn_ab(registers&)";

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = (m_dgemm_nn_ab.m_rowsa - 1) * m_dgemm_nn_ab.m_lda +
		m_dgemm_nn_ab.m_colsa;
	if(r.m_ptra[0] + sz > r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = (m_dgemm_nn_ab.m_colsa - 1) * m_dgemm_nn_ab.m_ldb +
		m_dgemm_nn_ab.m_colsb;
	if(r.m_ptra[1] + sz > r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	sz = (m_dgemm_nn_ab.m_rowsa - 1) * m_dgemm_nn_ab.m_ldc +
		m_dgemm_nn_ab.m_colsb;
	if(r.m_ptrb[0] + sz > r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	blas_dgemm(false, false, m_dgemm_nn_ab.m_rowsa, m_dgemm_nn_ab.m_colsb,
		m_dgemm_nn_ab.m_colsa, m_dgemm_nn_ab.m_d, r.m_ptra[0],
		m_dgemm_nn_ab.m_lda, r.m_ptra[1], m_dgemm_nn_ab.m_ldb, 1.0,
		r.m_ptrb[0], m_dgemm_nn_ab.m_ldc);
}


void loop_list_mul::fn_dgemm_nt_ab(registers &r) const {

	static const char *method = "fn_dgemm_nt_ab(registers&)";

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = (m_dgemm_nt_ab.m_rowsa - 1) * m_dgemm_nt_ab.m_lda +
		m_dgemm_nt_ab.m_colsa;
	if(r.m_ptra[0] + sz > r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = (m_dgemm_nt_ab.m_colsb - 1) * m_dgemm_nt_ab.m_ldb +
		m_dgemm_nt_ab.m_colsa;
	if(r.m_ptra[1] + sz > r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	sz = (m_dgemm_nt_ab.m_rowsa - 1) * m_dgemm_nt_ab.m_ldc +
		m_dgemm_nt_ab.m_colsb;
	if(r.m_ptrb[0] + sz > r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	blas_dgemm(false, true, m_dgemm_nt_ab.m_rowsa, m_dgemm_nt_ab.m_colsb,
		m_dgemm_nt_ab.m_colsa, m_dgemm_nt_ab.m_d, r.m_ptra[0],
		m_dgemm_nt_ab.m_lda, r.m_ptra[1], m_dgemm_nt_ab.m_ldb, 1.0,
		r.m_ptrb[0], m_dgemm_nt_ab.m_ldc);
}


void loop_list_mul::fn_dgemm_tn_ab(registers &r) const {

	static const char *method = "fn_dgemm_tn_ab(registers&)";

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = (m_dgemm_tn_ab.m_colsa - 1) * m_dgemm_tn_ab.m_lda +
		m_dgemm_tn_ab.m_rowsa;
	if(r.m_ptra[0] + sz > r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = (m_dgemm_tn_ab.m_colsa - 1) * m_dgemm_tn_ab.m_ldb +
		m_dgemm_tn_ab.m_colsb;
	if(r.m_ptra[1] + sz > r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	sz = (m_dgemm_tn_ab.m_rowsa - 1) * m_dgemm_tn_ab.m_ldc +
		m_dgemm_tn_ab.m_colsb;
	if(r.m_ptrb[0] + sz > r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	blas_dgemm(true, false, m_dgemm_tn_ab.m_rowsa, m_dgemm_tn_ab.m_colsb,
		m_dgemm_tn_ab.m_colsa, m_dgemm_tn_ab.m_d, r.m_ptra[0],
		m_dgemm_tn_ab.m_lda, r.m_ptra[1], m_dgemm_tn_ab.m_ldb, 1.0,
		r.m_ptrb[0], m_dgemm_tn_ab.m_ldc);
}


void loop_list_mul::fn_dgemm_tt_ab(registers &r) const {

	static const char *method = "fn_dgemm_tt_ab(registers&)";

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = (m_dgemm_tt_ab.m_colsa - 1) * m_dgemm_tt_ab.m_lda +
		m_dgemm_tt_ab.m_rowsa;
	if(r.m_ptra[0] + sz > r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = (m_dgemm_tt_ab.m_colsb - 1) * m_dgemm_tt_ab.m_ldb +
		m_dgemm_tt_ab.m_colsa;
	if(r.m_ptra[1] + sz > r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	sz = (m_dgemm_tt_ab.m_rowsa - 1) * m_dgemm_tt_ab.m_ldc +
		m_dgemm_tt_ab.m_colsb;
	if(r.m_ptrb[0] + sz > r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	blas_dgemm(true, true, m_dgemm_tt_ab.m_rowsa, m_dgemm_tt_ab.m_colsb,
		m_dgemm_tt_ab.m_colsa, m_dgemm_tt_ab.m_d, r.m_ptra[0],
		m_dgemm_tt_ab.m_lda, r.m_ptra[1], m_dgemm_tt_ab.m_ldb, 1.0,
		r.m_ptrb[0], m_dgemm_tt_ab.m_ldc);
}


void loop_list_mul::fn_dgemm_nn_ba(registers &r) const {

	static const char *method = "fn_dgemm_nn_ba(registers&)";

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = (m_dgemm_nn_ba.m_rowsb - 1) * m_dgemm_nn_ba.m_ldb +
		m_dgemm_nn_ba.m_colsb;
	if(r.m_ptra[1] + sz > r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = (m_dgemm_nn_ba.m_colsb - 1) * m_dgemm_nn_ba.m_lda +
		m_dgemm_nn_ba.m_colsa;
	if(r.m_ptra[0] + sz > r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	sz = (m_dgemm_nn_ba.m_rowsb - 1) * m_dgemm_nn_ba.m_ldc +
		m_dgemm_nn_ba.m_colsa;
	if(r.m_ptrb[0] + sz > r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	blas_dgemm(false, false, m_dgemm_nn_ba.m_rowsb, m_dgemm_nn_ba.m_colsa,
		m_dgemm_nn_ba.m_colsb, m_dgemm_nn_ba.m_d, r.m_ptra[1],
		m_dgemm_nn_ba.m_ldb, r.m_ptra[0], m_dgemm_nn_ba.m_lda, 1.0,
		r.m_ptrb[0], m_dgemm_nn_ba.m_ldc);
}


void loop_list_mul::fn_dgemm_nt_ba(registers &r) const {

	static const char *method = "fn_dgemm_nt_ba(registers&)";

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = (m_dgemm_nt_ba.m_rowsb - 1) * m_dgemm_nt_ba.m_ldb +
		m_dgemm_nt_ba.m_colsb;
	if(r.m_ptra[1] + sz > r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = (m_dgemm_nt_ba.m_colsa - 1) * m_dgemm_nt_ba.m_lda +
		m_dgemm_nt_ba.m_colsb;
	if(r.m_ptra[0] + sz > r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	sz = (m_dgemm_nt_ba.m_rowsb - 1) * m_dgemm_nt_ba.m_ldc +
		m_dgemm_nt_ba.m_colsa;
	if(r.m_ptrb[0] + sz > r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	blas_dgemm(false, true, m_dgemm_nt_ba.m_rowsb, m_dgemm_nt_ba.m_colsa,
		m_dgemm_nt_ba.m_colsb, m_dgemm_nt_ba.m_d, r.m_ptra[1],
		m_dgemm_nt_ba.m_ldb, r.m_ptra[0], m_dgemm_nt_ba.m_lda, 1.0,
		r.m_ptrb[0], m_dgemm_nt_ba.m_ldc);
}


void loop_list_mul::fn_dgemm_tn_ba(registers &r) const {

	static const char *method = "fn_dgemm_tn_ba(registers&)";

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = (m_dgemm_tn_ba.m_colsb - 1) * m_dgemm_tn_ba.m_ldb +
		m_dgemm_tn_ba.m_rowsb;
	if(r.m_ptra[1] + sz > r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = (m_dgemm_tn_ba.m_colsb - 1) * m_dgemm_tn_ba.m_lda +
		m_dgemm_tn_ba.m_colsa;
	if(r.m_ptra[0] + sz > r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	sz = (m_dgemm_tn_ba.m_rowsb - 1) * m_dgemm_tn_ba.m_ldc +
		m_dgemm_tn_ba.m_colsa;
	if(r.m_ptrb[0] + sz > r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	blas_dgemm(true, false, m_dgemm_tn_ba.m_rowsb, m_dgemm_tn_ba.m_colsa,
		m_dgemm_tn_ba.m_colsb, m_dgemm_tn_ba.m_d, r.m_ptra[1],
		m_dgemm_tn_ba.m_ldb, r.m_ptra[0], m_dgemm_tn_ba.m_lda, 1.0,
		r.m_ptrb[0], m_dgemm_tn_ba.m_ldc);
}


void loop_list_mul::fn_dgemm_tt_ba(registers &r) const {

	static const char *method = "fn_dgemm_tt_ba(registers&)";

#ifdef LIBTENSOR_DEBUG
	register size_t sz;
	sz = (m_dgemm_tt_ba.m_colsb - 1) * m_dgemm_tt_ba.m_ldb +
		m_dgemm_tt_ba.m_rowsb;
	if(r.m_ptra[1] + sz > r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-1");
	}
	sz = (m_dgemm_tt_ba.m_colsa - 1) * m_dgemm_tt_ba.m_lda +
		m_dgemm_tt_ba.m_colsb;
	if(r.m_ptra[0] + sz > r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source-2");
	}
	sz = (m_dgemm_tt_ba.m_rowsb - 1) * m_dgemm_tt_ba.m_ldc +
		m_dgemm_tt_ba.m_colsa;
	if(r.m_ptrb[0] + sz > r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	blas_dgemm(true, true, m_dgemm_tt_ba.m_rowsb, m_dgemm_tt_ba.m_colsa,
		m_dgemm_tt_ba.m_colsb, m_dgemm_tt_ba.m_d, r.m_ptra[1],
		m_dgemm_tt_ba.m_ldb, r.m_ptra[0], m_dgemm_tt_ba.m_lda, 1.0,
		r.m_ptrb[0], m_dgemm_tt_ba.m_ldc);
}


} // namespace libtensor
