#include <libtensor/btod/btod_random.h>
#include <libtensor/iface/iface.h>
#include "../compare_ref.h"
#include "contract_test.h"

namespace libtensor {


void contract_test::perform() throw(libtest::test_exception) {

	allocator<double>::vmm().init(16, 16, 16777216, 16777216);

	try {

		test_subexpr_labels_1();
		test_contr_bld_1();
		test_contr_bld_2();
		test_tt_1();
		test_tt_2();
		test_tt_3();
		test_tt_4();
		test_tt_5();
		test_tt_6();
		test_tt_7();
		test_tt_8();
		test_te_1();
		test_te_2();
		test_te_3();
		test_te_4();
		test_et_1();
		test_et_2();
		test_et_3();
		test_ee_1();
		test_ee_2();

	} catch(...) {
		allocator<double>::vmm().shutdown();
		throw;
	}

	allocator<double>::vmm().shutdown();
}


namespace contract_test_ns {

using labeled_btensor_expr::expr;
using labeled_btensor_expr::core_contract;
using labeled_btensor_expr::contract_subexpr_labels;

template<size_t N, size_t M, size_t K, typename T, typename E1, typename E2>
void test_subexpr_labels_tpl(
	expr<N + M, T, core_contract<N, M, K, T, E1, E2> > e,
	letter_expr<N + M> label_c) {

	contract_subexpr_labels<N, M, K, T, E1, E2> subexpr_labels(e, label_c);
}

} // namespace contract_test_ns
namespace ns = contract_test_ns;


void contract_test::test_subexpr_labels_1() throw(libtest::test_exception) {

	const char *testname = "contract_test::test_subexpr_labels_1()";

	try {

	bispace<1> spi(4), spa(5);
	bispace<4> spijab(spi&spi|spa&spa);
	btensor<4> ta(spijab), tb(spijab);
	letter i, j, k, l, a, b;
	ns::test_subexpr_labels_tpl(
		contract(a|b, ta(i|j|a|b), tb(k|l|a|b)),
		i|j|k|l);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void contract_test::test_contr_bld_1() throw(libtest::test_exception) {

	const char *testname = "contract_test::test_contr_bld_1()";

	try {

	letter i, j, k, l, a;
	letter_expr<4> label_c(i|j|k|l);
	letter_expr<1> label_contr(a);
	letter_expr<2> label_a(a|i);
	letter_expr<4> label_b(j|k|l|a);

	permutation<2> perm_a;
	permutation<4> perm_b;

	labeled_btensor_expr::contract_contraction2_builder<1, 3, 1> bld(
		label_a, perm_a, label_b, perm_b, label_c, label_contr);
	const contraction2<1, 3, 1> &contr = bld.get_contr();

	permutation<4> perm_c;
	contraction2<1, 3, 1> contr_ref(perm_c);
	contr_ref.contract(0, 3);

	for(size_t i = 0; i < 10; i++) {
		if(contr.get_conn().at(i) != contr_ref.get_conn().at(i)) {
			std::ostringstream ss;
			ss << "Incorrect connection at position " << i << ": "
				<< contr.get_conn().at(i) << " vs. "
				<< contr_ref.get_conn().at(i) << " (ref).";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void contract_test::test_contr_bld_2() throw(libtest::test_exception) {

	const char *testname = "contract_test::test_contr_bld_2()";

	try {

	letter i, j, k, l, a;
	letter_expr<4> label_c(i|j|k|l);
	letter_expr<1> label_contr(a);
	letter_expr<4> label_a(i|k|l|a);
	letter_expr<2> label_b(a|j);

	permutation<4> perm_a;
	permutation<2> perm_b;

	labeled_btensor_expr::contract_contraction2_builder<3, 1, 1> bld(
		label_a, perm_a, label_b, perm_b, label_c, label_contr);
	const contraction2<3, 1, 1> &contr = bld.get_contr();

	permutation<4> perm_c; perm_c.permute(1, 2).permute(1, 3);
	contraction2<3, 1, 1> contr_ref(perm_c);
	contr_ref.contract(3, 0);

	for(size_t i = 0; i < 10; i++) {
		if(contr.get_conn().at(i) != contr_ref.get_conn().at(i)) {
			std::ostringstream ss;
			ss << "Incorrect connection at position " << i << ": "
				<< contr.get_conn().at(i) << " vs. "
				<< contr_ref.get_conn().at(i) << " (ref).";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void contract_test::test_tt_1() throw(libtest::test_exception) {

	const char *testname = "contract_test::test_tt_1()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<2> sp_ia(sp_i|sp_a);
	bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
	bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

	btensor<2> t1(sp_ia);
	btensor<4> t2(sp_bcdi);
	btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

	btod_random<2>().perform(t1);
	btod_random<4>().perform(t2);
	t1.set_immutable();
	t2.set_immutable();

	contraction2<1, 3, 1> contr;
	contr.contract(0, 3);
	btod_contract2<1, 3, 1> op(contr, t1, t2);
	op.perform(t3_ref);

	letter a, b, c, d, i;
	t3(a|b|c|d) = contract(i, t1(i|a), t2(b|c|d|i));

	compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void contract_test::test_tt_2() throw(libtest::test_exception) {

	const char *testname = "contract_test::test_tt_2()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<2> sp_ia(sp_i|sp_a);
	bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
	bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

	btensor<2> t1(sp_ia);
	btensor<4> t2(sp_bcdi);
	btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

	btod_random<2>().perform(t1);
	btod_random<4>().perform(t2);
	t1.set_immutable();
	t2.set_immutable();

	permutation<4> perm_c; perm_c.permute(0, 1);
	contraction2<1, 3, 1> contr(perm_c);
	contr.contract(0, 3);
	btod_contract2<1, 3, 1> op(contr, t1, t2);
	op.perform(t3_ref);

	letter a, b, c, d, i;
	t3(a|b|c|d) = contract(i, t1(i|b), t2(a|c|d|i));

	compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void contract_test::test_tt_3() throw(libtest::test_exception) {

	const char *testname = "contract_test::test_tt_3()";

	try {

	bispace<1> sp_i(10), sp_a(20), sp_b(19);
	bispace<2> sp_ib(sp_i|sp_b);
	bispace<4> sp_acdi(sp_a|sp_a|sp_a|sp_i);
	bispace<4> sp_abcd(sp_a|sp_b|sp_a|sp_a);

	btensor<4> t1(sp_acdi);
	btensor<2> t2(sp_ib);
	btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

	btod_random<4>().perform(t1);
	btod_random<2>().perform(t2);
	t1.set_immutable();
	t2.set_immutable();

	permutation<4> perm_c; perm_c.permute(1, 2).permute(1, 3);
	contraction2<3, 1, 1> contr(perm_c);
	contr.contract(3, 0);
	btod_contract2<3, 1, 1> op(contr, t1, t2);
	op.perform(t3_ref);

	letter a, b, c, d, i;
	t3(a|b|c|d) = contract(i, t1(a|c|d|i), t2(i|b));

	compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void contract_test::test_tt_4() throw(libtest::test_exception) {

	const char *testname = "contract_test::test_tt_4()";

	try {

	bispace<1> sp_i(10), sp_a(20), sp_b(19);
	bispace<2> sp_ib(sp_i|sp_b);
	bispace<4> sp_acdi(sp_a|sp_a|sp_a|sp_i);
	bispace<4> sp_abcd(sp_a|sp_b|sp_a|sp_a);

	btensor<4> t1(sp_acdi);
	btensor<2> t2(sp_ib);
	btensor<4> t3(sp_abcd), t3_ref_tmp(sp_abcd), t3_ref(sp_abcd);

	btod_random<4>().perform(t1);
	btod_random<2>().perform(t2);
	t1.set_immutable();
	t2.set_immutable();

	permutation<4> perm_c; perm_c.permute(1, 2).permute(1, 3);
	contraction2<3, 1, 1> contr(perm_c);
	contr.contract(3, 0);
	btod_contract2<3, 1, 1> op(contr, t1, t2);
	op.perform(t3_ref_tmp);
	btod_copy<4>(t3_ref_tmp, -1.0).perform(t3_ref);

	letter a, b, c, d, i;
	t3(a|b|c|d) = -contract(i, t1(a|c|d|i), t2(i|b));

	compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void contract_test::test_tt_5() throw(libtest::test_exception) {

	const char *testname = "contract_test::test_tt_5()";

	try {

	bispace<1> sp_i(10), sp_a(20), sp_b(19);
	bispace<2> sp_ib(sp_i|sp_b);
	bispace<4> sp_acdi(sp_a|sp_a|sp_a|sp_i);
	bispace<4> sp_abcd(sp_a|sp_b|sp_a|sp_a);

	btensor<4> t1(sp_acdi);
	btensor<2> t2(sp_ib);
	btensor<4> t3(sp_abcd);
	btensor<4> t4(sp_abcd), t4_ref_tmp(sp_abcd), t4_ref(sp_abcd);

	btod_random<4>().perform(t1);
	btod_random<2>().perform(t2);
	btod_random<4>().perform(t3);
	t1.set_immutable();
	t2.set_immutable();
	t3.set_immutable();

	permutation<4> perm_c; perm_c.permute(1, 2).permute(1, 3);
	contraction2<3, 1, 1> contr(perm_c);
	contr.contract(3, 0);
	btod_contract2<3, 1, 1> op(contr, t1, t2);
	op.perform(t4_ref_tmp);
	btod_add<4> add(t3);
	add.add_op(t4_ref_tmp, -1.0);
	add.perform(t4_ref);

	letter a, b, c, d, i;
	t4(a|b|c|d) = t3(a|b|c|d) - contract(i, t1(a|c|d|i), t2(i|b));

	compare_ref<4>::compare(testname, t4, t4_ref, 1e-14);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void contract_test::test_tt_6() throw(libtest::test_exception) {

	const char *testname = "contract_test::test_tt_6()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<2> sp_ia(sp_i|sp_a), sp_ab(sp_a|sp_a);
	bispace<4> sp_iabc(sp_i|sp_a|sp_a|sp_a);

	btensor<4> t1(sp_iabc);
	btensor<2> t2(sp_ia);
	btensor<2> t3(sp_ab), t3_ref(sp_ab);

	btod_random<4>().perform(t1);
	btod_random<2>().perform(t2);
	t1.set_immutable();
	t2.set_immutable();

	contraction2<2, 0, 2> contr;
	contr.contract(0, 0);
	contr.contract(3, 1);
	btod_contract2<2, 0, 2>(contr, t1, t2).perform(t3_ref);

	letter a, b, c, i;
	t3(a|b) = contract(i|c, t1(i|a|b|c), t2(i|c));

	compare_ref<2>::compare(testname, t3, t3_ref, 1e-14);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void contract_test::test_tt_7() throw(libtest::test_exception) {

	const char *testname = "contract_test::test_tt_7()";

	try {

	bispace<1> sp_i(13), sp_a(7);
	bispace<4> sp_ijab(sp_i&sp_i|sp_a&sp_a), sp_iabc(sp_i|sp_a&sp_a&sp_a);

	btensor<4> t1(sp_iabc);
	btensor<4> t2(sp_ijab);
	btensor<4> t3(sp_iabc), t3_ref(sp_iabc);

	btod_random<4>().perform(t1);
	btod_random<4>().perform(t2);
	t1.set_immutable();
	t2.set_immutable();

	//	iabc = kcad ikbd
	//	caib->iabc
	contraction2<2, 2, 2> contr(permutation<4>().permute(0, 2).
		permute(2, 3));
	contr.contract(0, 1);
	contr.contract(3, 3);

	btod_contract2<2, 2, 2>(contr, t1, t2).perform(t3_ref, 1.0);

	letter i, k, a, b, c, d;
	t3(i|a|b|c) = contract(k|d, t1(k|c|a|d), t2(i|k|b|d));

	compare_ref<4>::compare(testname, t3, t3_ref, 1e-14);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void contract_test::test_tt_8() throw(libtest::test_exception) {

	const char *testname = "contract_test::test_tt_8()";

	try {

	bispace<1> sp_i(10), sp_a(3);
	sp_i.split(5);
	sp_a.split(2);
	bispace<2> sp_ab(sp_a&sp_a);
	bispace<4> sp_ijka(sp_i&sp_i&sp_i|sp_a);

	btensor<4> t1(sp_ijka), t2(sp_ijka);
	btensor<2> t3(sp_ab), t3_ref(sp_ab);

	{
		block_tensor_ctrl<4, double> c1(t1), c2(t2);
		c1.req_symmetry().insert(se_perm<4, double>(
			permutation<4>().permute(0, 1), false));
		c2.req_symmetry().insert(se_perm<4, double>(
			permutation<4>().permute(0, 1), false));
	}

	btod_random<4>().perform(t1);
	btod_random<4>().perform(t2);
	t1.set_immutable();
	t2.set_immutable();

	contraction2<1, 1, 3> contr(permutation<2>().permute(0, 1));
	contr.contract(0, 0);
	contr.contract(1, 1);
	contr.contract(2, 2);

	btod_contract2<1, 1, 3>(contr, t1, t2).perform(t3_ref, -0.5);

	letter i, j, k, a, b;
        t3(a|b) =  0.5 * (-contract(i|j|k, t1(k|j|i|b), t2(k|j|i|a)));

	compare_ref<2>::compare(testname, t3, t3_ref, 1e-14);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void contract_test::test_te_1() throw(libtest::test_exception) {

	const char *testname = "contract_test::test_te_1()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<2> sp_ia(sp_i|sp_a);
	bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
	bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

	btensor<2> t1(sp_ia);
	btensor<4> t2a(sp_bcdi), t2b(sp_bcdi), t2(sp_bcdi);
	btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

	btod_random<2>().perform(t1);
	btod_random<4>().perform(t2a);
	btod_random<4>().perform(t2b);
	t1.set_immutable();
	t2a.set_immutable();
	t2b.set_immutable();

	btod_add<4> op_add(t2a);
	op_add.add_op(t2b);
	op_add.perform(t2);
	t2.set_immutable();

	permutation<4> perm_c; perm_c.permute(0, 1);
	contraction2<1, 3, 1> contr(perm_c);
	contr.contract(0, 3);
	btod_contract2<1, 3, 1> op(contr, t1, t2);
	op.perform(t3_ref);

	letter a, b, c, d, i;
	t3(a|b|c|d) = contract(i, t1(i|b), t2a(a|c|d|i) + t2b(a|c|d|i));

	compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void contract_test::test_te_2() throw(libtest::test_exception) {

	const char *testname = "contract_test::test_te_2()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<2> sp_ia(sp_i|sp_a);
	bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
	bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

	btensor<2> t1(sp_ia);
	btensor<4> t2a(sp_bcdi), t2b(sp_bcdi), t2(sp_bcdi);
	btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

	btod_random<2>().perform(t1);
	btod_random<4>().perform(t2a);
	btod_random<4>().perform(t2b);
	t1.set_immutable();
	t2a.set_immutable();
	t2b.set_immutable();

	btod_add<4> op_add(t2a);
	op_add.add_op(t2b, -0.5);
	op_add.perform(t2);
	t2.set_immutable();

	permutation<4> perm_c; perm_c.permute(0, 1);
	contraction2<1, 3, 1> contr(perm_c);
	contr.contract(0, 3);
	btod_contract2<1, 3, 1> op(contr, t1, t2);
	op.perform(t3_ref);

	letter a, b, c, d, i;
	t3(a|b|c|d) = contract(i, t1(i|b), t2a(a|c|d|i) - 0.5*t2b(a|c|d|i));

	compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void contract_test::test_te_3() throw(libtest::test_exception) {

	const char *testname = "contract_test::test_te_3()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<2> sp_ia(sp_i|sp_a);
	bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
	bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

	btensor<2> t1(sp_ia);
	btensor<4> t2a(sp_bcdi), t2b(sp_bcdi), t2(sp_bcdi);
	btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

	btod_random<2>().perform(t1);
	btod_random<4>().perform(t2a);
	btod_random<4>().perform(t2b);
	t1.set_immutable();
	t2a.set_immutable();
	t2b.set_immutable();

	btod_add<4> op_add(t2a);
	op_add.add_op(t2b);
	op_add.perform(t2);
	t2.set_immutable();

	permutation<4> perm_c; perm_c.permute(0, 1);
	contraction2<1, 3, 1> contr(perm_c);
	contr.contract(0, 3);
	btod_contract2<1, 3, 1> op(contr, t1, t2);
	op.perform(t3_ref);

	letter a, b, c, d, i;
	t3(a|b|c|d) = contract(i, t1(i|b), t2a(a|c|d|i)) +
		contract(i, t1(i|b), t2b(a|c|d|i));

	compare_ref<4>::compare(testname, t3, t3_ref, 1e-14);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void contract_test::test_te_4() throw(libtest::test_exception) {

	const char *testname = "contract_test::test_te_4()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<2> sp_ia(sp_i|sp_a);
	bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
	bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

	btensor<2> t1(sp_ia);
	btensor<4> t2a(sp_bcdi), t2b(sp_bcdi), t2(sp_bcdi);
	btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

	btod_random<2>().perform(t1);
	btod_random<4>().perform(t2a);
	btod_random<4>().perform(t2b);
	t1.set_immutable();
	t2a.set_immutable();
	t2b.set_immutable();

	btod_add<4> op_add(t2a);
	op_add.add_op(t2b, -0.5);
	op_add.perform(t2);
	t2.set_immutable();

	permutation<4> perm_c; perm_c.permute(0, 1);
	contraction2<1, 3, 1> contr(perm_c);
	contr.contract(0, 3);
	btod_contract2<1, 3, 1> op(contr, t1, t2);
	op.perform(t3_ref);

	letter a, b, c, d, i;
	t3(a|b|c|d) = contract(i, t1(i|b), t2a(a|c|d|i)) -
		0.5*contract(i, t1(i|b), t2b(a|c|d|i));

	compare_ref<4>::compare(testname, t3, t3_ref, 1e-14);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void contract_test::test_et_1() throw(libtest::test_exception) {

	const char *testname = "contract_test::test_et_1()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<2> sp_ia(sp_i|sp_a);
	bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
	bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

	btensor<4> t1a(sp_bcdi), t1b(sp_bcdi), t1(sp_bcdi);
	btensor<2> t2(sp_ia);
	btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

	btod_random<4>().perform(t1a);
	btod_random<4>().perform(t1b);
	btod_random<2>().perform(t2);
	t1a.set_immutable();
	t1b.set_immutable();
	t2.set_immutable();

	btod_add<4> op_add(t1a);
	op_add.add_op(t1b);
	op_add.perform(t1);
	t1.set_immutable();

	permutation<4> perm_c; perm_c.permute(1, 2).permute(1, 3);
	contraction2<3, 1, 1> contr(perm_c);
	contr.contract(3, 0);
	btod_contract2<3, 1, 1> op(contr, t1, t2);
	op.perform(t3_ref);

	letter a, b, c, d, i;
	t3(a|b|c|d) = contract(i, t1a(a|c|d|i) + t1b(a|c|d|i), t2(i|b));

	compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void contract_test::test_et_2() throw(libtest::test_exception) {

	const char *testname = "contract_test::test_et_2()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<2> sp_ia(sp_i|sp_a);
	bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
	bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

	btensor<4> t1a(sp_bcdi), t1b(sp_bcdi), t1(sp_bcdi);
	btensor<2> t2(sp_ia);
	btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

	btod_random<4>().perform(t1a);
	btod_random<4>().perform(t1b);
	btod_random<2>().perform(t2);
	t1a.set_immutable();
	t1b.set_immutable();
	t2.set_immutable();

	btod_add<4> op_add(t1a);
	op_add.add_op(t1b, 1.5);
	op_add.perform(t1);
	t1.set_immutable();

	permutation<4> perm_c; perm_c.permute(1, 2).permute(1, 3);
	contraction2<3, 1, 1> contr(perm_c);
	contr.contract(3, 0);
	btod_contract2<3, 1, 1> op(contr, t1, t2);
	op.perform(t3_ref);

	letter a, b, c, d, i;
	t3(a|b|c|d) = contract(i, t1a(a|c|d|i) + 1.5*t1b(a|c|d|i), t2(i|b));

	compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void contract_test::test_et_3() throw(libtest::test_exception) {

	const char *testname = "contract_test::test_et_3()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<2> sp_ia(sp_i|sp_a);
	bispace<2> sp_ac(sp_a|sp_a), sp_di(sp_a|sp_i);
	bispace<4> sp_acdi(sp_a|sp_a|sp_a|sp_i);
	bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

	btensor<2> t1a(sp_ac), t1b(sp_di);
	btensor<4> t1(sp_acdi);
	btensor<2> t2(sp_ia);
	btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

	btod_random<2>().perform(t1a);
	btod_random<2>().perform(t1b);
	btod_random<2>().perform(t2);
	t1a.set_immutable();
	t1b.set_immutable();
	t2.set_immutable();

	contraction2<2, 2, 0> contr_t1;
	btod_contract2<2, 2, 0>(contr_t1, t1a, t1b).perform(t1);
	t1.set_immutable();

	permutation<4> perm_c; perm_c.permute(1, 2).permute(1, 3);
	contraction2<3, 1, 1> contr(perm_c);
	contr.contract(3, 0);
	btod_contract2<3, 1, 1> op(contr, t1, t2);
	op.perform(t3_ref);

	letter a, b, c, d, i;
	t3(a|b|c|d) = contract(i, t1a(a|c)*t1b(d|i), t2(i|b));

	compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void contract_test::test_ee_1() throw(libtest::test_exception) {

	const char *testname = "contract_test::test_ee_1()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<2> sp_ia(sp_i|sp_a);
	bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
	bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

	btensor<4> t1a(sp_bcdi), t1b(sp_bcdi), t1(sp_bcdi);
	btensor<2> t2a(sp_ia), t2b(sp_ia), t2(sp_ia);
	btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

	btod_random<4>().perform(t1a);
	btod_random<4>().perform(t1b);
	btod_random<2>().perform(t2a);
	btod_random<2>().perform(t2b);
	t1a.set_immutable();
	t1b.set_immutable();
	t2a.set_immutable();
	t2b.set_immutable();

	btod_add<4> op_add1(t1a);
	op_add1.add_op(t1b);
	op_add1.perform(t1);
	t1.set_immutable();

	btod_add<2> op_add2(t2a);
	op_add2.add_op(t2b);
	op_add2.perform(t2);
	t2.set_immutable();

	permutation<4> perm_c; perm_c.permute(1, 2).permute(1, 3);
	contraction2<3, 1, 1> contr(perm_c);
	contr.contract(3, 0);
	btod_contract2<3, 1, 1> op(contr, t1, t2);
	op.perform(t3_ref);

	letter a, b, c, d, i;
	t3(a|b|c|d) = contract(
		i, t1a(a|c|d|i) + t1b(a|c|d|i), t2a(i|b) + t2b(i|b));

	compare_ref<4>::compare(testname, t3, t3_ref, 1e-14);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void contract_test::test_ee_2() throw(libtest::test_exception) {

	const char *testname = "contract_test::test_ee_2()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<2> sp_ia(sp_i|sp_a);
	bispace<2> sp_ai(sp_a|sp_i);
	bispace<4> sp_acdi(sp_a|sp_a|sp_a|sp_i);
	bispace<4> sp_aicd(sp_a|sp_i|sp_a|sp_a);
	bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

	btensor<4> t1a(sp_acdi), t1b(sp_aicd), t1(sp_acdi);
	btensor<2> t2a(sp_ia), t2b(sp_ai), t2(sp_ia);
	btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

	btod_random<4>().perform(t1a);
	btod_random<4>().perform(t1b);
	btod_random<2>().perform(t2a);
	btod_random<2>().perform(t2b);
	t1a.set_immutable();
	t1b.set_immutable();
	t2a.set_immutable();
	t2b.set_immutable();

	permutation<4> perm_t1b;
	perm_t1b.permute(1, 2).permute(2, 3); // aicd->acid->acdi
	btod_add<4> op_add1(t1a);
	op_add1.add_op(t1b, perm_t1b);
	op_add1.perform(t1);
	t1.set_immutable();

	permutation<2> perm_t2b;
	perm_t2b.permute(0, 1); // bi->ib
	btod_add<2> op_add2(t2a);
	op_add2.add_op(t2b, perm_t2b);
	op_add2.perform(t2);
	t2.set_immutable();

	permutation<4> perm_c; perm_c.permute(1, 2).permute(1, 3);
	contraction2<3, 1, 1> contr(perm_c);
	contr.contract(3, 0);
	btod_contract2<3, 1, 1> op(contr, t1, t2);
	op.perform(t3_ref);

	letter a, b, c, d, i;
	t3(a|b|c|d) = contract(
		i, t1a(a|c|d|i) + t1b(a|i|c|d), t2a(i|b) + t2b(b|i));

	compare_ref<4>::compare(testname, t3, t3_ref, 1e-14);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
