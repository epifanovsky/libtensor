#include <libtensor.h>
#include "compare_ref.h"
#include "sym_contract_test.h"

namespace libtensor {


void sym_contract_test::perform() throw(libtest::test_exception) {

	test_tt_1();
	test_ee_1();

}


void sym_contract_test::test_tt_1() throw(libtest::test_exception) {

	const char *testname = "sym_contract_test::test_tt_1()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<2> sp_ia(sp_i|sp_a);
	bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
	bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

	btensor<2> t1(sp_ia);
	btensor<4> t2(sp_bcdi);
	btensor<4> t3(sp_abcd), t3_ref(sp_abcd), t3_ref_tmp(sp_abcd);

	btod_random<2>().perform(t1);
	btod_random<4>().perform(t2);
	t1.set_immutable();
	t2.set_immutable();

	contraction2<1, 3, 1> contr;
	contr.contract(0, 3);
	btod_contract2<1, 3, 1> op(contr, t1, t2);
	op.perform(t3_ref_tmp);
	permutation<4> perm; perm.permute(0, 1);
	btod_copy<4>(t3_ref_tmp).perform(t3_ref);
	btod_copy<4>(t3_ref_tmp, perm).perform(t3_ref, 1.0);

	letter a, b, c, d, i;
	t3(a|b|c|d) = sym_contract(a|b, i, t1(i|a), t2(b|c|d|i));

	compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void sym_contract_test::test_ee_1() throw(libtest::test_exception) {

	const char *testname = "sym_contract_test::test_ee_1()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<2> sp_ia(sp_i|sp_a);
	bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
	bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

	btensor<4> t1a(sp_bcdi), t1b(sp_bcdi), t1(sp_bcdi);
	btensor<2> t2a(sp_ia), t2b(sp_ia), t2(sp_ia);
	btensor<4> t3(sp_abcd), t3_ref_tmp(sp_abcd), t3_ref(sp_abcd);

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
	op.perform(t3_ref_tmp);

	permutation<4> perm_sym; perm_sym.permute(0, 1);
	btod_add<4> op_sym(t3_ref_tmp);
	op_sym.add_op(t3_ref_tmp, perm_sym);
	op_sym.perform(t3_ref);

	letter a, b, c, d, i;
	t3(a|b|c|d) = sym_contract(
		a|b, i, t1a(a|c|d|i) + t1b(a|c|d|i), t2a(i|b) + t2b(i|b));

	compare_ref<4>::compare(testname, t3, t3_ref, 1e-14);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
