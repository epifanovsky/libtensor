#include "../defs.h"
#include "../exception.h"
#include "../blas.h"
#include "../linalg.h"
#include "loop_list_mul.h"
#include "overflow.h"

namespace libtensor {


const char *loop_list_mul::k_clazz = "loop_list_mul";


void loop_list_mul::run_loop(list_t &loop, registers &r, double c) {

	match_l1(loop, c);

	iterator_t begin = loop.begin(), end = loop.end();
	if(begin != end) {
		loop_list_base<2, 1, loop_list_mul>::exec(
			*this, begin, end, r);
	}
}


void loop_list_mul::match_l1(list_t &loop, double d) {

	//	1. Find:
	//	-----------
	//	w   a  b  c
	//	w1  1  1  0  -->  c = a_p b_p
	//	-----------       sz(p) = w1
	//	                  [ddot]
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
	size_t k1a_min = 0, k1b_min = 0;
	for(iterator_t i = loop.begin(); i != loop.end(); i++) {
		if(i->stepa(0) == 1 && i->stepa(1) == 1 && i->stepb(0) == 0) {
			i1 = i;
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
	if(i1 != loop.end()) {
		//~ std::cout << "ddot";
		m_kernelname = "ddot";
		i1->fn() = &loop_list_mul::fn_ddot;
		m_ddot.m_d = d;
		m_ddot.m_n = i1->weight();
		match_ddot_l2(loop, d, i1->weight());
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
}


void loop_list_mul::match_ddot_l2(list_t &loop, double d, size_t w1) {

	//	Found pattern:
	//	-----------
	//	w   a  b  c
	//	w1  1  1  0  -->  c = a_p b_p
	//	-----------       sz(p) = w1
	//	                  [ddot]
	//

	//	1. Minimize k1a:
	//	----------------
	//	w   a       b  c
	//	w1  1       1  0
	//	w2  k1a*w1  0  1  -->  c_i = a_i$p b_p
	//	----------------       sz(i) = w2, sz(p) = w1, sz($) = k1a
	//	                       [dgemv_n_a]
	//
	//	2. Minimize k1b:
	//	----------------
	//	w   a  b       c
	//	w1  1  1       0
	//	w2  0  k1b*w1  1  -->  c_i = a_p b_i%p
	//	----------------       sz(i) = w2, sz(p) = w1, sz(%) = k1b
	//	                       [dgemv_n_b]
	//
	size_t k1a_min = 0, k1b_min = 0;
	iterator_t i1 = loop.end(), i2 = loop.end();
	for(iterator_t i = loop.begin(); i != loop.end(); i++) {
		if(i->stepb(0) != 1) continue;
		if(i->stepa(1) == 0 && i->stepa(0) % w1 == 0) {
			register size_t k1a = i->stepa(0) / w1;
			if(k1a_min == 0 || k1a_min > k1a) {
				k1a_min = k1a; i1 = i;
			}
		}
		if(i->stepa(0) == 0 && i->stepa(1) % w1 == 0) {
			register size_t k1b = i->stepa(1) / w1;
			if(k1b_min == 0 || k1b_min > k1b) {
				k1b_min = k1b; i2 = i;
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

	//	1. Minimize k2:
	//	-----------------
	//	w   a   b      c
	//	w1  0   1      k1
	//	w2  k3  k2*w1  0  -->  c_i# = a_p$ b_p%i
	//      -----------------      sz(i) = w1, sz(p) = w2
	//	                       sz(#) = k1, sz($) = k3, sz(%) = k2
	//	                       [dgemv_t_b]
	size_t k2_min = 0;
	iterator_t i1 = loop.end();
	for(iterator_t i = loop.begin(); i != loop.end(); i++) {
		if(i->stepb(0) != 0) continue;
		if(i->stepa(1) % w1 != 0) continue;

		register size_t k2 = i->stepa(1) / w1;
		if(k2_min == 0 || k2_min > k2) {
			k2_min = k2; i1 = i;
		}
	}
	if(i1 != loop.end()) {
		//~ std::cout << " dgemv_t_b";
		m_kernelname = "dgemv_t_b";
		i1->fn() = &loop_list_mul::fn_dgemv_t_b;
		m_dgemv_t_b.m_d = d;
		m_dgemv_t_b.m_rows = i1->weight();
		m_dgemv_t_b.m_cols = w1;
		m_dgemv_t_b.m_stepa = i1->stepa(0);
		m_dgemv_t_b.m_ldb = i1->stepa(1);
		m_dgemv_t_b.m_stepc = k1;
		if(k1 == 1) {
			match_dgemv_t_b_l3(loop, d, w1, i1->weight(),
				i1->stepa(1), i1->stepa(0));
		}
		loop.splice(loop.end(), loop, i1);
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


void loop_list_mul::match_dgemv_t_b_l3(list_t &loop, double d, size_t w1,
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

	//	1. Minimize k4:
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
	iterator_t i1 = loop.end();
	for(iterator_t i = loop.begin(); i != loop.end(); i++) {
		if(i->stepa(0) != 1 || i->stepa(1) != 0) continue;
		if(k3 % i->weight() != 0) continue;
		if(i->stepb(0) % w1 != 0) continue;

		register size_t k4 = i->stepb(0) / w1;
		if(k4_min == 0 || k4_min > k4) {
			k4_min = k4; i1 = i;
		}
	}
	if(i1 != loop.end()) {
		//~ std::cout << " dgemm_tn_ab";
		m_kernelname = "dgemm_tn_ab";
		i1->fn() = &loop_list_mul::fn_dgemm_tn_ab;
		m_dgemm_tn_ab.m_d = d;
		m_dgemm_tn_ab.m_rowsa = i1->weight();
		m_dgemm_tn_ab.m_colsb = w1;
		m_dgemm_tn_ab.m_colsa = w2;
		m_dgemm_tn_ab.m_lda = k3;
		m_dgemm_tn_ab.m_ldb = k2w1;
		m_dgemm_tn_ab.m_ldc = i1->stepb(0);
		loop.splice(loop.end(), loop, i1);
		return;
	}
}


void loop_list_mul::fn_ddot(registers &r) const {

	*r.m_ptrb[0] += m_ddot.m_d * cblas_ddot(m_ddot.m_n, r.m_ptra[0], 1,
		r.m_ptra[1], 1);
}


void loop_list_mul::fn_daxpy_a(registers &r) const {

	double d = m_daxpy_a.m_d;
	size_t n = m_daxpy_a.m_n;
	size_t stepc = m_daxpy_a.m_stepc;

	cblas_daxpy(n, *r.m_ptra[1] * d, r.m_ptra[0], 1, r.m_ptrb[0], stepc);
}


void loop_list_mul::fn_daxpy_b(registers &r) const {

	double d = m_daxpy_b.m_d;
	size_t n = m_daxpy_b.m_n;
	size_t stepc = m_daxpy_b.m_stepc;

	cblas_daxpy(n, *r.m_ptra[0] * d, r.m_ptra[1], 1, r.m_ptrb[0], stepc);
}


void loop_list_mul::fn_dgemv_n_a(registers &r) const {

	double d = m_dgemv_n_a.m_d;
	size_t rows = m_dgemv_n_a.m_rows;
	size_t cols = m_dgemv_n_a.m_cols;
	size_t lda = m_dgemv_n_a.m_lda;
	size_t stepb = m_dgemv_n_a.m_stepb;

//	tod_contract2<N, M, K>::start_timer("dgemv_n_a");
	cblas_dgemv(CblasRowMajor, CblasNoTrans, rows, cols, d, r.m_ptra[0], lda,
		r.m_ptra[1], stepb, 1.0, r.m_ptrb[0], 1);
//	tod_contract2<N, M, K>::stop_timer("dgemv_n_a");
}


void loop_list_mul::fn_dgemv_t_a(registers &r) const {

	double d = m_dgemv_t_a.m_d;
	size_t rows = m_dgemv_t_a.m_rows;
	size_t cols = m_dgemv_t_a.m_cols;
	size_t lda = m_dgemv_t_a.m_lda;
	size_t stepb = m_dgemv_t_a.m_stepb;
	size_t stepc = m_dgemv_t_a.m_stepc;

//	tod_contract2<N, M, K>::start_timer("dgemv_t_a");
	cblas_dgemv(CblasRowMajor, CblasTrans, rows, cols, d, r.m_ptra[0], lda,
		r.m_ptra[1], stepb, 1.0, r.m_ptrb[0], stepc);
//	tod_contract2<N, M, K>::stop_timer("dgemv_t_a");
}


void loop_list_mul::fn_dgemv_n_b(registers &r) const {

	double d = m_dgemv_n_b.m_d;
	size_t rows = m_dgemv_n_b.m_rows;
	size_t cols = m_dgemv_n_b.m_cols;
	size_t ldb = m_dgemv_n_b.m_ldb;
	size_t stepa = m_dgemv_n_b.m_stepa;

//	tod_contract2<N, M, K>::start_timer("dgemv_n_b");
	cblas_dgemv(CblasRowMajor, CblasNoTrans, rows, cols, d, r.m_ptra[1], ldb,
		r.m_ptra[0], stepa, 1.0, r.m_ptrb[0], 1);
//	tod_contract2<N, M, K>::stop_timer("dgemv_n_b");
}


void loop_list_mul::fn_dgemv_t_b(registers &r) const {

	double d = m_dgemv_t_b.m_d;
	size_t rows = m_dgemv_t_b.m_rows;
	size_t cols = m_dgemv_t_b.m_cols;
	size_t ldb = m_dgemv_t_b.m_ldb;
	size_t stepa = m_dgemv_t_b.m_stepa;
	size_t stepc = m_dgemv_t_b.m_stepc;

//	tod_contract2<N, M, K>::start_timer("dgemv_t_b");
	cblas_dgemv(CblasRowMajor, CblasTrans, rows, cols, d, r.m_ptra[1], ldb,
		r.m_ptra[0], stepa, 1.0, r.m_ptrb[0], stepc);
//	tod_contract2<N, M, K>::stop_timer("dgemv_t_b");
}


void loop_list_mul::fn_dgemm_nt_ab(registers &r) const {

	double d = m_dgemm_nt_ab.m_d;
	size_t rowsa = m_dgemm_nt_ab.m_rowsa;
	size_t colsb = m_dgemm_nt_ab.m_colsb;
	size_t colsa = m_dgemm_nt_ab.m_colsa;
	size_t lda = m_dgemm_nt_ab.m_lda;
	size_t ldb = m_dgemm_nt_ab.m_ldb;
	size_t ldc = m_dgemm_nt_ab.m_ldc;

//	tod_contract2<N, M, K>::start_timer("dgemm_nt_ab");
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		rowsa, colsb, colsa, d, r.m_ptra[0], lda, r.m_ptra[1], ldb,
		1.0, r.m_ptrb[0], ldc);
//	tod_contract2<N, M, K>::stop_timer("dgemm_nt_ab");
}


void loop_list_mul::fn_dgemm_tn_ab(registers &r) const {

	double d = m_dgemm_tn_ab.m_d;
	size_t rowsa = m_dgemm_tn_ab.m_rowsa;
	size_t colsb = m_dgemm_tn_ab.m_colsb;
	size_t colsa = m_dgemm_tn_ab.m_colsa;
	size_t lda = m_dgemm_tn_ab.m_lda;
	size_t ldb = m_dgemm_tn_ab.m_ldb;
	size_t ldc = m_dgemm_tn_ab.m_ldc;

//	tod_contract2<N, M, K>::start_timer("dgemm_tn_ab");
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		rowsa, colsb, colsa, d, r.m_ptra[0], lda, r.m_ptra[1], ldb,
		1.0, r.m_ptrb[0], ldc);
//	tod_contract2<N, M, K>::stop_timer("dgemm_tn_ab");
}


void loop_list_mul::fn_dgemm_tt_ab(registers &r) const {

	double d = m_dgemm_tt_ab.m_d;
	size_t rowsa = m_dgemm_tt_ab.m_rowsa;
	size_t colsb = m_dgemm_tt_ab.m_colsb;
	size_t colsa = m_dgemm_tt_ab.m_colsa;
	size_t lda = m_dgemm_tt_ab.m_lda;
	size_t ldb = m_dgemm_tt_ab.m_ldb;
	size_t ldc = m_dgemm_tt_ab.m_ldc;

//	tod_contract2<N, M, K>::start_timer("dgemm_tt_ab");
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans,
		rowsa, colsb, colsa, d, r.m_ptra[0], lda, r.m_ptra[1], ldb,
		1.0, r.m_ptrb[0], ldc);
//	tod_contract2<N, M, K>::stop_timer("dgemm_tt_ab");
}


void loop_list_mul::fn_dgemm_nn_ba(registers &r) const {

	double d = m_dgemm_nn_ba.m_d;
	size_t rowsb = m_dgemm_nn_ba.m_rowsb;
	size_t colsa = m_dgemm_nn_ba.m_colsa;
	size_t colsb = m_dgemm_nn_ba.m_colsb;
	size_t ldb = m_dgemm_nn_ba.m_ldb;
	size_t lda = m_dgemm_nn_ba.m_lda;
	size_t ldc = m_dgemm_nn_ba.m_ldc;

//	tod_contract2<N, M, K>::start_timer("dgemm_nn_ba");
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		rowsb, colsa, colsb, d, r.m_ptra[1], ldb, r.m_ptra[0], lda,
		1.0, r.m_ptrb[1], ldc);
//	tod_contract2<N, M, K>::stop_timer("dgemm_nn_ba");
}


void loop_list_mul::fn_dgemm_nt_ba(registers &r) const {

	double d = m_dgemm_nt_ba.m_d;
	size_t rowsb = m_dgemm_nt_ba.m_rowsb;
	size_t colsa = m_dgemm_nt_ba.m_colsa;
	size_t colsb = m_dgemm_nt_ba.m_colsb;
	size_t ldb = m_dgemm_nt_ba.m_ldb;
	size_t lda = m_dgemm_nt_ba.m_lda;
	size_t ldc = m_dgemm_nt_ba.m_ldc;

//	tod_contract2<N, M, K>::start_timer("dgemm_nt_ba");
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		rowsb, colsa, colsb, d, r.m_ptra[1], ldb, r.m_ptra[0], lda,
		1.0, r.m_ptrb[0], ldc);
//	tod_contract2<N, M, K>::stop_timer("dgemm_nt_ba");
}


void loop_list_mul::fn_dgemm_tn_ba(registers &r) const {

	double d = m_dgemm_tn_ba.m_d;
	size_t rowsb = m_dgemm_tn_ba.m_rowsb;
	size_t colsa = m_dgemm_tn_ba.m_colsa;
	size_t colsb = m_dgemm_tn_ba.m_colsb;
	size_t ldb = m_dgemm_tn_ba.m_ldb;
	size_t lda = m_dgemm_tn_ba.m_lda;
	size_t ldc = m_dgemm_tn_ba.m_ldc;

//	tod_contract2<N, M, K>::start_timer("dgemm_tn_ba");
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		rowsb, colsa, colsb, d, r.m_ptra[1], ldb, r.m_ptra[0], lda,
		1.0, r.m_ptrb[0], ldc);
//	tod_contract2<N, M, K>::stop_timer("dgemm_tn_ba");
}


/*
void loop_list_add::match_l1(list_t &loop, double c) {

	//	1. Find:
	//	--------
	//	w   a  b
	//	w1  1  1  -->  b_i += a_i
	//	--------       sz(i) = w1
	//	               [daxpy]
	//
	//	2. Minimize k1a:
	//
	//	----------
	//	w   a  b
	//	w1  1  k1a  -->  b_i# += a_i
	//	----------       sz(i) = w1, sz(#) = k1a
	//	                 [daxpy]
	//
	//	3. Minimize k1b:
	//
	//	----------
	//	w   a    b
	//	w1  k1b  1  -->  b_i += a_i#
	//	----------       sz(i) = w1, sz(#) = k1b
	//	                 [daxpy]
	//
	//	4. Fallback [daxpy]
	//
	iterator_t i1 = loop.end(), i2 = loop.end(), i3 = loop.end();
	size_t k1a_min = 0, k1b_min = 0;
	for(iterator_t i = loop.begin(); i != loop.end(); i++) {
		i->fn() = 0;
		if(i->stepa(0) == 1) {
			if(k1a_min == 0 || k1a_min > i->stepb(0)) {
				i2 = i; k1a_min = i->stepb(0);
			}
			if(i->stepb(0) == 1) {
				i1 = i;
			}
		} else {
			if(i->stepb(0) == 1) {
				if(k1b_min == 0 || k1b_min > i->stepa(0)) {
					i3 = i; k1b_min = i->stepa(0);
				}
			}
		}
	}

	if(i1 != loop.end()) {
		i1->fn() = &loop_list_add::fn_daxpy;
		m_daxpy.m_k = c;
		m_daxpy.m_n = i1->weight();
		m_daxpy.m_stepa = 1;
		m_daxpy.m_stepb = 1;
		loop.splice(loop.end(), loop, i1);
		return;
	}

	if(i2 != loop.end()) {
		i2->fn() = &loop_list_add::fn_daxpy;
		m_daxpy.m_k = c;
		m_daxpy.m_n = i2->weight();
		m_daxpy.m_stepa = 1;
		m_daxpy.m_stepb = i2->stepb(0);
		//~ match_l2_a(loop, c, i2->weight());
		loop.splice(loop.end(), loop, i2);
		return;
	}

	if(i3 != loop.end()) {
		i3->fn() = &loop_list_add::fn_daxpy;
		m_daxpy.m_k = c;
		m_daxpy.m_n = i3->weight();
		m_daxpy.m_stepa = i3->stepa(0);
		m_daxpy.m_stepb = 1;
		//~ match_l2_b(loop, c, i3->weight());
		loop.splice(loop.end(), loop, i3);
		return;
	}

	iterator_t i4 = loop.begin();
	i4->fn() = &loop_list_add::fn_daxpy;
	m_daxpy.m_k = c;
	m_daxpy.m_n = i4->weight();
	m_daxpy.m_stepa = i4->stepa(0);
	m_daxpy.m_stepb = i4->stepb(0);
	loop.splice(loop.end(), loop, i4);
}


void loop_list_add::match_l2_a(list_t &loop, double c, size_t w1) {

	//	Found pattern:
	//	---------
	//	w   a  b
	//	w1  1  k1  -->  b_i# += a_i
	//	---------       sz(i) = w1, sz(#) = k1
	//	                [daxpy]

	//	1. Find k2a = 1:
	//	--------------
	//	w   a       b
	//	w1  1       k1
	//	w2  k2a*w1  1   -->  b_ij += a_j$i
	//	--------------       sz(i) = w1, sz(j) = w2, sz($) = k2a
	//	                     [daxpby_trp]
	iterator_t i1 = loop.end();
	for(iterator_t i = loop.begin(); i != loop.end(); i++) {
		if(i->stepb(0) != 1) continue;
		if(i->stepa(0) != w1) continue;
		i1 = i;
		break;
	}

	if(i1 != loop.end()) {
		i1->fn() = &loop_list_add::fn_daxpby_trp;
		m_daxpby_trp.m_k = c;
		m_daxpby_trp.m_ni = w1;
		m_daxpby_trp.m_nj = i1->weight();
		m_daxpby_trp.m_stepa = 1;
		m_daxpby_trp.m_stepb = 1;
		loop.splice(loop.end(), loop, i1);
	}
}


void loop_list_add::match_l2_b(list_t &loop, double c, size_t w1) {

}


void loop_list_add::fn_daxpy(registers &r) const {

	static const char *method = "fn_daxpy(registers&)";

	if(m_daxpy.m_n == 0) return;

#ifdef LIBTENSOR_DEBUG
	if(r.m_ptra[0] + (m_daxpy.m_n - 1) * m_daxpy.m_stepa >=
		r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source");
	}
	if(r.m_ptrb[0] + (m_daxpy.m_n - 1) * m_daxpy.m_stepb >=
		r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	blas_daxpy(m_daxpy.m_n, m_daxpy.m_k, r.m_ptra[0], m_daxpy.m_stepa,
		r.m_ptrb[0], m_daxpy.m_stepb);
}


void loop_list_add::fn_daxpby_trp(registers &r) const {

	static const char *method = "fn_daxpby_trp(registers&)";

	if(m_daxpby_trp.m_ni == 0 || m_daxpby_trp.m_nj) return;

#ifdef LIBTENSOR_DEBUG
	if(r.m_ptra[0] + (m_daxpby_trp.m_ni - 1) * m_daxpby_trp.m_stepa >=
		r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source");
	}
	if(r.m_ptrb[0] + (m_daxpby_trp.m_nj - 1) * m_daxpby_trp.m_stepb >=
		r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	blas::daxpby_trp(r.m_ptra[0], r.m_ptrb[0], m_daxpby_trp.m_ni,
		m_daxpby_trp.m_nj, m_daxpby_trp.m_stepa, m_daxpby_trp.m_stepb,
		m_daxpby_trp.m_k, 1.0);
}
*/

} // namespace libtensor
