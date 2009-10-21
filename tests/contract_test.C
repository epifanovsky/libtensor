#include <libtensor.h>
#include "compare_ref.h"
#include "contract_test.h"

namespace libtensor {


void contract_test::perform() throw(libtest::test_exception) {

	test_subexpr_labels_1();
	test_contr_bld_1();
	test_1();
	test_2();

	bispace<1> sp_i(10), sp_j(10), sp_a(20), sp_b(20);
	bispace<2> sp_ij(sp_i&sp_j), sp_ab(sp_a&sp_b);
	bispace<2> sp_ji(sp_j&sp_i), sp_ba(sp_b&sp_a);
	bispace<4> sp_ijab(sp_ij|sp_ab), sp_jiab(sp_ji|sp_ab);
	bispace<4> sp_abcd(sp_ab&sp_ab);
	btensor<4> t1_ijab(sp_ijab), t2_ijab(sp_ijab);
	btensor<4> t3_jiab(sp_jiab), t4_jiab(sp_jiab);
	btensor<4> t5_abcd(sp_abcd), t6_abcd(sp_abcd), t7_abcd(sp_abcd);
	bispace<2> sp_ia(sp_i|sp_a);
	bispace<4> sp_bcdi(sp_a&sp_a&sp_a|sp_i);
	btensor<2> t8_ia(sp_ia);
	btensor<4> t9_bcdi(sp_bcdi);

	letter i, j, k, l, a, b, c, d;

	try {
	contract(i, t1_ijab(i|j|a|b), t2_ijab(i|k|c|d));
	t5_abcd(a|b|c|d) = contract(i|j, t1_ijab(i|j|a|b), t2_ijab(i|j|c|d));
	t6_abcd(a|b|c|d) = contract(i|j, t1_ijab(i|j|a|b) + t3_jiab(j|i|a|b),
		t2_ijab(i|j|c|d));
	t7_abcd(a|b|c|d) = t5_abcd(a|b|c|d) + t6_abcd(a|b|c|d);
	t7_abcd(a|b|c|d) = contract(i|j, t1_ijab(i|j|a|b), t2_ijab(i|j|c|d)) +
		contract(i|j, t1_ijab(i|j|a|b) + t3_jiab(j|i|a|b),
			t2_ijab(i|j|c|d)) + t6_abcd(a|b|c|d);
	contract(i|j, t1_ijab(i|j|a|b), t2_ijab(i|j|c|d) + t3_jiab(j|i|c|d));
	} catch(exception &e) {
		fail_test("contract_test::perform()", __FILE__, __LINE__,
			e.what());
	}
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


void contract_test::test_1() throw(libtest::test_exception) {

	const char *testname = "contract_test::test_1()";

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


void contract_test::test_2() throw(libtest::test_exception) {

	const char *testname = "contract_test::test_2()";

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


} // namespace libtensor
