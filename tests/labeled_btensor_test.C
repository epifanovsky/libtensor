#include <sstream>
#include <libtensor.h>
#include "labeled_btensor_test.h"

namespace libtensor {

void labeled_btensor_test::perform() throw(libtest::test_exception) {
	test_label();
	test_expr();
	test_expr_copy_1();
	test_expr_copy_2();
	test_expr_copy_3();
	test_expr_copy_4();
}

void labeled_btensor_test::test_label() throw(libtest::test_exception) {
	bispace<1> sp_i(10), sp_j(10), sp_a(20), sp_b(20);
	bispace<4> sp_ijab((sp_i&sp_j)*(sp_a&sp_b));
	btensor<4> t(sp_ijab);

	letter i, j, k, a, b, c;

	if(!t(i|j|a|b).contains(i)) {
		fail_test("labeled_btensor_test::test_label()", __FILE__,
			__LINE__, "Failed label test: t(i|j|a|b).contains(i)");
	}
	if(!t(i|j|a|b).contains(j)) {
		fail_test("labeled_btensor_test::test_label()", __FILE__,
			__LINE__, "Failed label test: t(i|j|a|b).contains(j)");
	}
	if(!t(i|j|a|b).contains(a)) {
		fail_test("labeled_btensor_test::test_label()", __FILE__,
			__LINE__, "Failed label test: t(i|j|a|b).contains(a)");
	}
	if(!t(i|j|a|b).contains(b)) {
		fail_test("labeled_btensor_test::test_label()", __FILE__,
			__LINE__, "Failed label test: t(i|j|a|b).contains(b)");
	}
	if(t(i|j|a|b).contains(k)) {
		fail_test("labeled_btensor_test::test_label()", __FILE__,
			__LINE__, "Failed label test: t(i|j|a|b).contains(k)");
	}

	if(t(i|j|a|b).index_of(i) != 0) {
		fail_test("labeled_btensor_test::test_label()", __FILE__,
			__LINE__, "Failed label test: t(i|j|a|b).index_of(i)");
	}
	if(t(i|j|a|b).index_of(j) != 1) {
		fail_test("labeled_btensor_test::test_label()", __FILE__,
			__LINE__, "Failed label test: t(i|j|a|b).index_of(j)");
	}
	if(t(i|j|a|b).index_of(a) != 2) {
		fail_test("labeled_btensor_test::test_label()", __FILE__,
			__LINE__, "Failed label test: t(i|j|a|b).index_of(a)");
	}
	if(t(i|j|a|b).index_of(b) != 3) {
		fail_test("labeled_btensor_test::test_label()", __FILE__,
			__LINE__, "Failed label test: t(i|j|a|b).index_of(b)");
	}

	if(t(i|j|a|b).letter_at(0) != i) {
		fail_test("labeled_btensor_test::test_label()", __FILE__,
			__LINE__, "Failed label test: t(i|j|a|b).letter_at(0)");
	}
	if(t(i|j|a|b).letter_at(1) != j) {
		fail_test("labeled_btensor_test::test_label()", __FILE__,
			__LINE__, "Failed label test: t(i|j|a|b).letter_at(1)");
	}
	if(t(i|j|a|b).letter_at(2) != a) {
		fail_test("labeled_btensor_test::test_label()", __FILE__,
			__LINE__, "Failed label test: t(i|j|a|b).letter_at(2)");
	}
	if(t(i|j|a|b).letter_at(3) != b) {
		fail_test("labeled_btensor_test::test_label()", __FILE__,
			__LINE__, "Failed label test: t(i|j|a|b).letter_at(3)");
	}
}

void labeled_btensor_test::test_expr() throw(libtest::test_exception) {
	bispace<1> sp_i(2), sp_j(2), sp_a(3), sp_b(3);
	bispace<2> sp_ij(sp_i&sp_j), sp_ab(sp_a&sp_b);
	bispace<2> sp_ji(sp_j&sp_i), sp_ba(sp_b&sp_a);
	bispace<4> sp_ijab(sp_ij*sp_ab), sp_jiab(sp_ji*sp_ab);
	btensor<4> t1_ijab(sp_ijab), t2_ijab(sp_ijab);
	btensor<4> t3_jiab(sp_jiab), t4_jiab(sp_jiab);
	btensor_i<4, double> &t1i_ijab(t1_ijab);

	letter i, j, a, b;

	t3_jiab(j|i|a|b) = t1_ijab(i|j|a|b);
	t3_jiab(j|i|a|b) = -t1_ijab(i|j|a|b);
	t3_jiab(j|i|a|b) = -(t1_ijab(i|j|a|b)*2.0);
	t3_jiab(j|i|a|b) = t1i_ijab(i|j|a|b);
	t3_jiab(j|i|a|b) = t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b);
	t2_ijab(i|j|a|b) = t3_jiab(j|i|a|b) + t4_jiab(j|i|a|b);
	t1_ijab(i|j|a|b) + t3_jiab(j|i|a|b);
	t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b);
	t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b) +
		t4_jiab(j|i|a|b);
	t1_ijab(i|j|a|b) + (t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b));
	(t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b)) +
		(t3_jiab(j|i|a|b) + t4_jiab(j|i|a|b));

	t3_jiab(j|i|a|b) = 0.5*t1_ijab(i|j|a|b);
	t3_jiab(j|i|a|b) = t1_ijab(i|j|a|b)*0.5;
	t3_jiab(j|i|a|b) = 0.5*t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b);
	t1_ijab(i|j|a|b)*2.0 + t2_ijab(i|j|a|b);
	t1_ijab(i|j|a|b) + 0.5*t2_ijab(i|j|a|b);
	t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b)*2.0;
	0.5*t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b)*2.0;
	t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) + 0.5*t3_jiab(j|i|a|b);
	0.5*t1_ijab(i|j|a|b) + (t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b));
	0.5*t1_ijab(i|j|a|b) + (t2_ijab(i|j|a|b) + 2.0*t3_jiab(j|i|a|b));
	0.5*(t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b));
	(t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b))*0.5;
	0.5*(t1_ijab(i|j|a|b) + 2.0*t2_ijab(i|j|a|b));
	t4_jiab(j|i|a|b) = (t1_ijab(i|j|a|b) + 2.0*t2_ijab(i|j|a|b))*0.5;
	2.0*(t1_ijab(i|j|a|b) + 2.0*t2_ijab(i|j|a|b))*0.5;
}

void labeled_btensor_test::test_expr_copy_1() throw(libtest::test_exception) {
	// b(i|j) = a(i|j)

	bispace<1> sp_i(4), sp_j(4);
	bispace<2> sp_ij(sp_i & sp_j);
	btensor<2> bta(sp_ij), btb(sp_ij), btb_ref(sp_ij);

	block_tensor_ctrl<2, double> btctrla(bta), btctrlb(btb),
		btctrlb_ref(btb_ref);
	index<2> i0;
	tensor_i<2, double> &ta = btctrla.req_block(i0);
	tensor_i<2, double> &tb = btctrlb.req_block(i0);
	tensor_i<2, double> &tb_ref = btctrlb_ref.req_block(i0);
	tensor_ctrl<2, double> tca(ta), tcb(tb), tcb_ref(tb_ref);

	dimensions<2> dims(ta.get_dims());

	double *dta = tca.req_dataptr();
	double *dtb1 = tcb.req_dataptr();
	double *dtb2 = tcb_ref.req_dataptr();

	// Fill in random data

	index<2> ida;
	do {
		size_t i;
		i = dims.abs_index(ida);
		dta[i] = dtb2[i] = drand48();
		dtb1[i] = drand48();
	} while(dims.inc_index(ida));

	tca.ret_dataptr(dta); dta = NULL;
	tcb.ret_dataptr(dtb1); dtb1 = NULL;
	tcb_ref.ret_dataptr(dtb2); dtb2 = NULL;

	bta.set_immutable(); btb_ref.set_immutable();

	// Evaluate the expression

	letter i, j;
	btb(i|j) = bta(i|j);

	// Compare against the reference

	compare_ref("labeled_btensor_test::test_expr_copy_1()",
		btb, btb_ref, 1e-15);
}

void labeled_btensor_test::test_expr_copy_2() throw(libtest::test_exception) {
	// b(i|j) = a(j|i)

	bispace<1> sp_i(4), sp_j(4);
	bispace<2> sp_ij(sp_i & sp_j);
	btensor<2> bta(sp_ij), btb(sp_ij), btb_ref(sp_ij);

	block_tensor_ctrl<2, double> btctrla(bta), btctrlb(btb),
		btctrlb_ref(btb_ref);
	index<2> i0;
	tensor_i<2, double> &ta = btctrla.req_block(i0);
	tensor_i<2, double> &tb = btctrlb.req_block(i0);
	tensor_i<2, double> &tb_ref = btctrlb_ref.req_block(i0);
	tensor_ctrl<2, double> tca(ta), tcb(tb), tcb_ref(tb_ref);

	dimensions<2> dims(ta.get_dims());

	double *dta = tca.req_dataptr();
	double *dtb1 = tcb.req_dataptr();
	double *dtb2 = tcb_ref.req_dataptr();

	// Fill in random data

	index<2> ida;
	permutation<2> p; p.permute(0, 1);
	do {
		index<2> idb = ida; idb.permute(p);
		size_t i = dims.abs_index(ida);
		size_t j = dims.abs_index(idb);
		dta[i] = dtb2[j] = drand48();
		dtb1[j] = drand48();
	} while(dims.inc_index(ida));

	tca.ret_dataptr(dta); dta = NULL;
	tcb.ret_dataptr(dtb1); dtb1 = NULL;
	tcb_ref.ret_dataptr(dtb2); dtb2 = NULL;

	bta.set_immutable(); btb_ref.set_immutable();

	// Evaluate the expression

	letter i, j;
	btb(i|j) = bta(j|i);

	// Compare against the reference

	compare_ref("labeled_btensor_test::test_expr_copy_2()",
		btb, btb_ref, 1e-15);
}

void labeled_btensor_test::test_expr_copy_3() throw(libtest::test_exception) {
	// b(i|j) = 1.5*a(j|i)

	bispace<1> sp_i(4), sp_j(4);
	bispace<2> sp_ij(sp_i & sp_j);
	btensor<2> bta(sp_ij), btb(sp_ij), btb_ref(sp_ij);

	block_tensor_ctrl<2, double> btctrla(bta), btctrlb(btb),
		btctrlb_ref(btb_ref);
	index<2> i0;
	tensor_i<2, double> &ta = btctrla.req_block(i0);
	tensor_i<2, double> &tb = btctrlb.req_block(i0);
	tensor_i<2, double> &tb_ref = btctrlb_ref.req_block(i0);
	tensor_ctrl<2, double> tca(ta), tcb(tb), tcb_ref(tb_ref);

	dimensions<2> dims(ta.get_dims());

	double *dta = tca.req_dataptr();
	double *dtb1 = tcb.req_dataptr();
	double *dtb2 = tcb_ref.req_dataptr();

	// Fill in random data

	index<2> ida;
	permutation<2> p; p.permute(0, 1);
	do {
		index<2> idb = ida; idb.permute(p);
		size_t i = dims.abs_index(ida);
		size_t j = dims.abs_index(idb);
		dta[i] = drand48();
		dtb2[j] = 1.5 * dta[i];
		dtb1[j] = drand48();
	} while(dims.inc_index(ida));

	tca.ret_dataptr(dta); dta = NULL;
	tcb.ret_dataptr(dtb1); dtb1 = NULL;
	tcb_ref.ret_dataptr(dtb2); dtb2 = NULL;

	bta.set_immutable(); btb_ref.set_immutable();

	// Evaluate the expression

	letter i, j;
	btb(i|j) = 1.5*bta(j|i);

	// Compare against the reference

	compare_ref("labeled_btensor_test::test_expr_copy_3()",
		btb, btb_ref, 1e-15);
}

void labeled_btensor_test::test_expr_copy_4() throw(libtest::test_exception) {
	// b(i|j) = -a(i|j)

	bispace<1> sp_i(4), sp_j(4);
	bispace<2> sp_ij(sp_i & sp_j);
	btensor<2> bta(sp_ij), btb(sp_ij), btb_ref(sp_ij);

	block_tensor_ctrl<2, double> btctrla(bta), btctrlb(btb),
		btctrlb_ref(btb_ref);
	index<2> i0;
	tensor_i<2, double> &ta = btctrla.req_block(i0);
	tensor_i<2, double> &tb = btctrlb.req_block(i0);
	tensor_i<2, double> &tb_ref = btctrlb_ref.req_block(i0);
	tensor_ctrl<2, double> tca(ta), tcb(tb), tcb_ref(tb_ref);

	dimensions<2> dims(ta.get_dims());

	double *dta = tca.req_dataptr();
	double *dtb1 = tcb.req_dataptr();
	double *dtb2 = tcb_ref.req_dataptr();

	// Fill in random data

	index<2> ida;
	do {
		size_t i;
		i = dims.abs_index(ida);
		dta[i] = drand48();
		dtb2[i] = -dta[i];
		dtb1[i] = drand48();
	} while(dims.inc_index(ida));

	tca.ret_dataptr(dta); dta = NULL;
	tcb.ret_dataptr(dtb1); dtb1 = NULL;
	tcb_ref.ret_dataptr(dtb2); dtb2 = NULL;

	bta.set_immutable(); btb_ref.set_immutable();

	// Evaluate the expression

	letter i, j;
	btb(i|j) = -bta(i|j);

	// Compare against the reference

	compare_ref("labeled_btensor_test::test_expr_copy_4()",
		btb, btb_ref, 1e-15);
}

template<size_t N>
void labeled_btensor_test::compare_ref(const char *test,
	btensor_i<N, double> &bt, btensor_i<N, double> &bt_ref, double thresh)
	throw(libtest::test_exception) {

	btod_compare<N> cmp(bt, bt_ref, thresh);
	if(!cmp.compare()) {
		std::ostringstream ss1, ss2;
		ss2 << "Result does not match reference at element "
			<< cmp.get_diff_index() << ": "
			<< cmp.get_diff_elem_1() << " (act) vs. "
			<< cmp.get_diff_elem_2() << " (ref), "
			<< cmp.get_diff_elem_1() - cmp.get_diff_elem_2()
			<< " (diff) in " << test;
		fail_test("labeled_btensor_test::compare_ref()",
			__FILE__, __LINE__, ss2.str().c_str());
	}

}

} // namespace libtensor
