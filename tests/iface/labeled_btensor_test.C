#include <cstdlib>
#include <sstream>
#include <libtensor/btod/btod_random.h>
#include <libtensor/iface/iface.h>
#include "../compare_ref.h"
#include "labeled_btensor_test.h"

namespace libtensor {

void labeled_btensor_test::perform() throw(libtest::test_exception) {

	allocator<double>::vmm().init(16, 16, 16777216, 16777216);

	try {

		test_label();
		test_expr();
		test_expr_copy_1();
		test_expr_copy_2();
		test_expr_copy_3();
		test_expr_copy_4();
		test_expr_add_1();
		test_expr_add_2();
		test_expr_add_3();
		test_expr_add_4();
		test_expr_add_5();

	} catch(...) {
		allocator<double>::vmm().shutdown();
		throw;
	}

	allocator<double>::vmm().shutdown();
}


void labeled_btensor_test::test_label() throw(libtest::test_exception) {

	static const char *testname =
		"labeled_btensor_test::test_label()";

	try {

	bispace<1> sp_i(10), sp_j(10), sp_a(20), sp_b(20);
	bispace<4> sp_ijab((sp_i&sp_j)|(sp_a&sp_b));
	btensor<4> t(sp_ijab);

	letter i, j, k, a, b, c;

	if(!t(i|j|a|b).contains(i)) {
		fail_test(testname, __FILE__, __LINE__,
			"Failed label test: t(i|j|a|b).contains(i)");
	}
	if(!t(i|j|a|b).contains(j)) {
		fail_test(testname, __FILE__, __LINE__,
			"Failed label test: t(i|j|a|b).contains(j)");
	}
	if(!t(i|j|a|b).contains(a)) {
		fail_test(testname, __FILE__, __LINE__,
			"Failed label test: t(i|j|a|b).contains(a)");
	}
	if(!t(i|j|a|b).contains(b)) {
		fail_test(testname, __FILE__, __LINE__,
			"Failed label test: t(i|j|a|b).contains(b)");
	}
	if(t(i|j|a|b).contains(k)) {
		fail_test(testname, __FILE__, __LINE__,
			"Failed label test: t(i|j|a|b).contains(k)");
	}

	if(t(i|j|a|b).index_of(i) != 0) {
		fail_test(testname, __FILE__, __LINE__,
			"Failed label test: t(i|j|a|b).index_of(i)");
	}
	if(t(i|j|a|b).index_of(j) != 1) {
		fail_test(testname, __FILE__, __LINE__,
			"Failed label test: t(i|j|a|b).index_of(j)");
	}
	if(t(i|j|a|b).index_of(a) != 2) {
		fail_test(testname, __FILE__, __LINE__,
			"Failed label test: t(i|j|a|b).index_of(a)");
	}
	if(t(i|j|a|b).index_of(b) != 3) {
		fail_test(testname, __FILE__, __LINE__,
			"Failed label test: t(i|j|a|b).index_of(b)");
	}

	if(t(i|j|a|b).letter_at(0) != i) {
		fail_test(testname, __FILE__, __LINE__,
			"Failed label test: t(i|j|a|b).letter_at(0)");
	}
	if(t(i|j|a|b).letter_at(1) != j) {
		fail_test(testname, __FILE__, __LINE__,
			"Failed label test: t(i|j|a|b).letter_at(1)");
	}
	if(t(i|j|a|b).letter_at(2) != a) {
		fail_test(testname, __FILE__, __LINE__,
			"Failed label test: t(i|j|a|b).letter_at(2)");
	}
	if(t(i|j|a|b).letter_at(3) != b) {
		fail_test(testname, __FILE__, __LINE__,
			"Failed label test: t(i|j|a|b).letter_at(3)");
	}

	btensor<1> s(sp_i);

	if (s(i).index_of(i) != 0) {
		fail_test(testname, __FILE__, __LINE__,
			"Failed label test: s(i).contains(i)");
	}

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}

}


void labeled_btensor_test::test_expr() throw(libtest::test_exception) {

	static const char *testname =
		"labeled_btensor_test::test_expr()";

	try {

	bispace<1> sp_i(2), sp_j(2), sp_a(3), sp_b(3);
	bispace<2> sp_ij(sp_i&sp_j), sp_ab(sp_a&sp_b);
	bispace<2> sp_ji(sp_j&sp_i), sp_ba(sp_b&sp_a);
	bispace<4> sp_ijab(sp_ij|sp_ab), sp_jiab(sp_ji|sp_ab);
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
	t1_ijab(i|j|a|b) - 2.0*t3_jiab(j|i|a|b);
	2.0*t1_ijab(i|j|a|b) - t3_jiab(j|i|a|b);
	2.0*t1_ijab(i|j|a|b) - 3.0*t3_jiab(j|i|a|b);
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

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


void labeled_btensor_test::test_expr_copy_1() throw(libtest::test_exception) {

	// b(i|j) = a(i|j)

	static const char *testname =
		"labeled_btensor_test::test_expr_copy_1()";

	try {

	bispace<1> sp_i(4), sp_j(4);
	bispace<2> sp_ij(sp_i&sp_j);
	btensor<2> bta(sp_ij), btb(sp_ij), btb_ref(sp_ij);

	block_tensor_ctrl<2, double> btctrla(bta), btctrlb(btb),
		btctrlb_ref(btb_ref);
	index<2> i0;
	tensor_i<2, double> &ta = btctrla.req_block(i0);
	tensor_i<2, double> &tb = btctrlb.req_block(i0);
	tensor_i<2, double> &tb_ref = btctrlb_ref.req_block(i0);

	dimensions<2> dims(ta.get_dims());

	{
		tensor_ctrl<2, double> tca(ta), tcb(tb), tcb_ref(tb_ref);
		double *dta = tca.req_dataptr();
		double *dtb1 = tcb.req_dataptr();
		double *dtb2 = tcb_ref.req_dataptr();

		// Fill in random data

		abs_index<2> aida(dims);
		do {
			size_t i = aida.get_abs_index();
			dta[i] = dtb2[i] = drand48();
			dtb1[i] = drand48();
		} while(aida.inc());

		tca.ret_dataptr(dta); dta = NULL;
		tcb.ret_dataptr(dtb1); dtb1 = NULL;
		tcb_ref.ret_dataptr(dtb2); dtb2 = NULL;
	}

	bta.set_immutable(); btb_ref.set_immutable();

	// Evaluate the expression

	letter i, j;
	btb(i|j) = bta(i|j);

	// Compare against the reference

	compare_ref<2>::compare(testname, btb, btb_ref, 1e-15);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}

}


void labeled_btensor_test::test_expr_copy_2() throw(libtest::test_exception) {

	// b(i|j) = a(j|i)

	static const char *testname =
		"labeled_btensor_test::test_expr_copy_2()";

	try {

	bispace<1> sp_i(4), sp_j(4);
	bispace<2> sp_ij(sp_i&sp_j);
	btensor<2> bta(sp_ij), btb(sp_ij), btb_ref(sp_ij);

	block_tensor_ctrl<2, double> btctrla(bta), btctrlb(btb),
		btctrlb_ref(btb_ref);
	index<2> i0;
	tensor_i<2, double> &ta = btctrla.req_block(i0);
	tensor_i<2, double> &tb = btctrlb.req_block(i0);
	tensor_i<2, double> &tb_ref = btctrlb_ref.req_block(i0);

	dimensions<2> dims(ta.get_dims());

	{
		tensor_ctrl<2, double> tca(ta), tcb(tb), tcb_ref(tb_ref);
		double *dta = tca.req_dataptr();
		double *dtb1 = tcb.req_dataptr();
		double *dtb2 = tcb_ref.req_dataptr();

		// Fill in random data

		abs_index<2> aida(dims);
		permutation<2> p; p.permute(0, 1);
		do {
		    index<2> idb(aida.get_index());
			idb.permute(p);
		    abs_index<2> aidb(idb, dims);
			size_t i = aida.get_abs_index();
			size_t j = aidb.get_abs_index();
			dta[i] = dtb2[j] = drand48();
			dtb1[j] = drand48();
		} while(aida.inc());

		tca.ret_dataptr(dta); dta = NULL;
		tcb.ret_dataptr(dtb1); dtb1 = NULL;
		tcb_ref.ret_dataptr(dtb2); dtb2 = NULL;
	}

	bta.set_immutable(); btb_ref.set_immutable();

	// Evaluate the expression

	letter i, j;
	btb(i|j) = bta(j|i);

	// Compare against the reference

	compare_ref<2>::compare(testname, btb, btb_ref, 1e-15);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


void labeled_btensor_test::test_expr_copy_3() throw(libtest::test_exception) {

	// b(i|j) = 1.5*a(j|i)

	static const char *testname =
		"labeled_btensor_test::test_expr_copy_3()";

	try {

	bispace<1> sp_i(4), sp_j(4);
	bispace<2> sp_ij(sp_i&sp_j);
	btensor<2> bta(sp_ij), btb(sp_ij), btb_ref(sp_ij);

	block_tensor_ctrl<2, double> btctrla(bta), btctrlb(btb),
		btctrlb_ref(btb_ref);
	index<2> i0;
	tensor_i<2, double> &ta = btctrla.req_block(i0);
	tensor_i<2, double> &tb = btctrlb.req_block(i0);
	tensor_i<2, double> &tb_ref = btctrlb_ref.req_block(i0);

	dimensions<2> dims(ta.get_dims());

	{
		tensor_ctrl<2, double> tca(ta), tcb(tb), tcb_ref(tb_ref);
		double *dta = tca.req_dataptr();
		double *dtb1 = tcb.req_dataptr();
		double *dtb2 = tcb_ref.req_dataptr();

		// Fill in random data

		abs_index<2> aida(dims);
		permutation<2> p; p.permute(0, 1);
		do {
			index<2> idb(aida.get_index());
			idb.permute(p);
			abs_index<2> aidb(idb, dims);
			size_t i = aida.get_abs_index();
			size_t j = aidb.get_abs_index();
			dta[i] = drand48();
			dtb2[j] = 1.5 * dta[i];
			dtb1[j] = drand48();
		} while(aida.inc());

		tca.ret_dataptr(dta); dta = NULL;
		tcb.ret_dataptr(dtb1); dtb1 = NULL;
		tcb_ref.ret_dataptr(dtb2); dtb2 = NULL;
	}

	bta.set_immutable(); btb_ref.set_immutable();

	// Evaluate the expression

	letter i, j;
	btb(i|j) = 1.5*bta(j|i);

	// Compare against the reference

	compare_ref<2>::compare(testname, btb, btb_ref, 1e-15);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


void labeled_btensor_test::test_expr_copy_4() throw(libtest::test_exception) {

	// b(i|j) = -a(i|j)

	static const char *testname =
		"labeled_btensor_test::test_expr_copy_4()";

	try {

	bispace<1> sp_i(4), sp_j(4);
	bispace<2> sp_ij(sp_i&sp_j);
	btensor<2> bta(sp_ij), btb(sp_ij), btb_ref(sp_ij);

	block_tensor_ctrl<2, double> btctrla(bta), btctrlb(btb),
		btctrlb_ref(btb_ref);
	index<2> i0;
	tensor_i<2, double> &ta = btctrla.req_block(i0);
	tensor_i<2, double> &tb = btctrlb.req_block(i0);
	tensor_i<2, double> &tb_ref = btctrlb_ref.req_block(i0);

	dimensions<2> dims(ta.get_dims());

	{
		tensor_ctrl<2, double> tca(ta), tcb(tb), tcb_ref(tb_ref);
		double *dta = tca.req_dataptr();
		double *dtb1 = tcb.req_dataptr();
		double *dtb2 = tcb_ref.req_dataptr();

		// Fill in random data

		abs_index<2> aida(dims);
		do {
			size_t i = aida.get_abs_index();
			dta[i] = drand48();
			dtb2[i] = -dta[i];
			dtb1[i] = drand48();
		} while(aida.inc());

		tca.ret_dataptr(dta); dta = NULL;
		tcb.ret_dataptr(dtb1); dtb1 = NULL;
		tcb_ref.ret_dataptr(dtb2); dtb2 = NULL;
	}

	bta.set_immutable(); btb_ref.set_immutable();

	// Evaluate the expression

	letter i, j;
	btb(i|j) = -bta(i|j);

	// Compare against the reference

	compare_ref<2>::compare(testname, btb, btb_ref, 1e-15);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


void labeled_btensor_test::test_expr_add_1() throw(libtest::test_exception) {

	// c(i|j) = a(i|j) + b(i|j)

	static const char *testname =
		"labeled_btensor_test::test_expr_add_1()";

	try {

	bispace<1> sp_i(4), sp_j(4);
	bispace<2> sp_ij(sp_i&sp_j);
	btensor<2> bta(sp_ij), btb(sp_ij), btc(sp_ij), btc_ref(sp_ij);

	block_tensor_ctrl<2, double> btctrla(bta), btctrlb(btb),
		btctrlc(btc), btctrlc_ref(btc_ref);
	index<2> i0;
	tensor_i<2, double> &ta = btctrla.req_block(i0);
	tensor_i<2, double> &tb = btctrlb.req_block(i0);
	tensor_i<2, double> &tc = btctrlc.req_block(i0);
	tensor_i<2, double> &tc_ref = btctrlc_ref.req_block(i0);

	dimensions<2> dims(ta.get_dims());

	{
		tensor_ctrl<2, double> tca(ta), tcb(tb), tcc(tc),
			tcc_ref(tc_ref);
		double *dta = tca.req_dataptr();
		double *dtb = tcb.req_dataptr();
		double *dtc1 = tcc.req_dataptr();
		double *dtc2 = tcc_ref.req_dataptr();

		// Fill in random data

		abs_index<2> aida(dims);
		do {
			size_t i = aida.get_abs_index();
			dta[i] = drand48();
			dtb[i] = drand48();
			dtc1[i] = drand48();
			dtc2[i] = dta[i] + dtb[i];
		} while(aida.inc());

		tca.ret_dataptr(dta); dta = NULL;
		tcb.ret_dataptr(dtb); dtb = NULL;
		tcc.ret_dataptr(dtc1); dtc1 = NULL;
		tcc_ref.ret_dataptr(dtc2); dtc2 = NULL;
	}

	bta.set_immutable();
	btb.set_immutable();
	btc_ref.set_immutable();

	// Evaluate the expression

	letter i, j;
	btc(i|j) = bta(i|j) + btb(i|j);

	// Compare against the reference

	compare_ref<2>::compare(testname, btc, btc_ref, 1e-15);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


void labeled_btensor_test::test_expr_add_2() throw(libtest::test_exception) {

	// c(i|j) = -a(i|j) + 3.0*b(i|j)

	static const char *testname =
		"labeled_btensor_test::test_expr_add_2()";

	try {

	bispace<1> sp_i(4), sp_j(4);
	bispace<2> sp_ij(sp_i&sp_j);
	btensor<2> bta(sp_ij), btb(sp_ij), btc(sp_ij), btc_ref(sp_ij);

	block_tensor_ctrl<2, double> btctrla(bta), btctrlb(btb),
		btctrlc(btc), btctrlc_ref(btc_ref);
	index<2> i0;
	tensor_i<2, double> &ta = btctrla.req_block(i0);
	tensor_i<2, double> &tb = btctrlb.req_block(i0);
	tensor_i<2, double> &tc = btctrlc.req_block(i0);
	tensor_i<2, double> &tc_ref = btctrlc_ref.req_block(i0);

	dimensions<2> dims(ta.get_dims());

	{
		tensor_ctrl<2, double> tca(ta), tcb(tb), tcc(tc),
			tcc_ref(tc_ref);
		double *dta = tca.req_dataptr();
		double *dtb = tcb.req_dataptr();
		double *dtc1 = tcc.req_dataptr();
		double *dtc2 = tcc_ref.req_dataptr();

		// Fill in random data

		abs_index<2> aida(dims);
		do {
			size_t i = aida.get_abs_index();
			dta[i] = drand48();
			dtb[i] = drand48();
			dtc1[i] = drand48();
			dtc2[i] = -dta[i] + 3.0*dtb[i];
		} while(aida.inc());

		tca.ret_dataptr(dta); dta = NULL;
		tcb.ret_dataptr(dtb); dtb = NULL;
		tcc.ret_dataptr(dtc1); dtc1 = NULL;
		tcc_ref.ret_dataptr(dtc2); dtc2 = NULL;
	}

	bta.set_immutable();
	btb.set_immutable();
	btc_ref.set_immutable();

	// Evaluate the expression

	letter i, j;
	btc(i|j) = -bta(i|j) + 3.0*btb(i|j);

	// Compare against the reference

	compare_ref<2>::compare(testname, btc, btc_ref, 1e-15);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


void labeled_btensor_test::test_expr_add_3() throw(libtest::test_exception) {

	// c(i|j) = a(i|j) - b(i|j)

	static const char *testname =
		"labeled_btensor_test::test_expr_add_3()";

	try {

	bispace<1> sp_i(4), sp_j(4);
	bispace<2> sp_ij(sp_i&sp_j);
	btensor<2> bta(sp_ij), btb(sp_ij), btc(sp_ij), btc_ref(sp_ij);

	block_tensor_ctrl<2, double> btctrla(bta), btctrlb(btb),
		btctrlc(btc), btctrlc_ref(btc_ref);
	index<2> i0;
	tensor_i<2, double> &ta = btctrla.req_block(i0);
	tensor_i<2, double> &tb = btctrlb.req_block(i0);
	tensor_i<2, double> &tc = btctrlc.req_block(i0);
	tensor_i<2, double> &tc_ref = btctrlc_ref.req_block(i0);

	dimensions<2> dims(ta.get_dims());

	{
		tensor_ctrl<2, double> tca(ta), tcb(tb), tcc(tc),
			tcc_ref(tc_ref);
		double *dta = tca.req_dataptr();
		double *dtb = tcb.req_dataptr();
		double *dtc1 = tcc.req_dataptr();
		double *dtc2 = tcc_ref.req_dataptr();

		// Fill in random data

		abs_index<2> aida(dims);
		do {
			size_t i = aida.get_abs_index();
			dta[i] = drand48();
			dtb[i] = drand48();
			dtc1[i] = drand48();
			dtc2[i] = dta[i] - dtb[i];
		} while(aida.inc());

		tca.ret_dataptr(dta); dta = NULL;
		tcb.ret_dataptr(dtb); dtb = NULL;
		tcc.ret_dataptr(dtc1); dtc1 = NULL;
		tcc_ref.ret_dataptr(dtc2); dtc2 = NULL;
	}

	bta.set_immutable();
	btb.set_immutable();
	btc_ref.set_immutable();

	// Evaluate the expression

	letter i, j;
	btc(i|j) = bta(i|j) - btb(i|j);

	// Compare against the reference

	compare_ref<2>::compare(testname, btc, btc_ref, 1e-15);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


void labeled_btensor_test::test_expr_add_4() throw(libtest::test_exception) {

	// c(i|j) = 4.0*a(i|j) - 0.5*b(j|i)

	static const char *testname =
		"labeled_btensor_test::test_expr_add_4()";

	try {

	bispace<1> sp_i(4), sp_j(4);
	bispace<2> sp_ij(sp_i&sp_j);
	btensor<2> bta(sp_ij), btb(sp_ij), btc(sp_ij), btc_ref(sp_ij);

	block_tensor_ctrl<2, double> btctrla(bta), btctrlb(btb),
		btctrlc(btc), btctrlc_ref(btc_ref);
	index<2> i0;
	tensor_i<2, double> &ta = btctrla.req_block(i0);
	tensor_i<2, double> &tb = btctrlb.req_block(i0);
	tensor_i<2, double> &tc = btctrlc.req_block(i0);
	tensor_i<2, double> &tc_ref = btctrlc_ref.req_block(i0);

	dimensions<2> dims(ta.get_dims());

	{
		tensor_ctrl<2, double> tca(ta), tcb(tb), tcc(tc),
			tcc_ref(tc_ref);
		double *dta = tca.req_dataptr();
		double *dtb = tcb.req_dataptr();
		double *dtc1 = tcc.req_dataptr();
		double *dtc2 = tcc_ref.req_dataptr();

		// Fill in random data

		abs_index<2> aida(dims);
		permutation<2> p; p.permute(0, 1);
		do {
			index<2> idb(aida.get_index());
			idb.permute(p);
			abs_index<2> aidb(idb, dims);
			size_t i = aida.get_abs_index();
			size_t j = aidb.get_abs_index();
			dta[i] = drand48();
			dtb[j] = drand48();
			dtc1[i] = drand48();
			dtc2[i] = 4.0*dta[i] - 0.5*dtb[j];
		} while(aida.inc());

		tca.ret_dataptr(dta); dta = NULL;
		tcb.ret_dataptr(dtb); dtb = NULL;
		tcc.ret_dataptr(dtc1); dtc1 = NULL;
		tcc_ref.ret_dataptr(dtc2); dtc2 = NULL;
	}

	bta.set_immutable();
	btb.set_immutable();
	btc_ref.set_immutable();

	// Evaluate the expression

	letter i, j;
	btc(i|j) = 4.0*bta(i|j) - 0.5*btb(j|i);

	// Compare against the reference

	compare_ref<2>::compare(testname, btc, btc_ref, 1e-15);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


void labeled_btensor_test::test_expr_add_5() throw(libtest::test_exception) {

	// d(i|j) = a(i|j) + b(i|j) + c(i|j)

	static const char *testname =
		"labeled_btensor_test::test_expr_add_5()";

	try {

	bispace<1> sp_i(4), sp_j(4);
	bispace<2> sp_ij(sp_i&sp_j);
	btensor<2> bta(sp_ij), btb(sp_ij), btc(sp_ij),
		btd(sp_ij), btd_ref(sp_ij);

	//	Fill in random data

	btod_random<2>().perform(bta);
	btod_random<2>().perform(btb);
	btod_random<2>().perform(btc);

	//	Compute the reference

	btod_add<2> add(bta);
	add.add_op(btb);
	add.add_op(btc);
	add.perform(btd_ref);

	//	Evaluate the expression

	letter i, j;
	btd(i|j) = bta(i|j) + btb(i|j) + btc(i|j);

	//	Compare against the reference

	compare_ref<2>::compare(testname, btd, btd_ref, 1e-15);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


} // namespace libtensor
