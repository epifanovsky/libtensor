#include <libtensor.h>
#include "anon_eval_test.h"
#include "compare_ref.h"

namespace libtensor {


void anon_eval_test::perform() throw(libtest::test_exception) {

	test_copy_1();
	test_copy_2();
	test_copy_3();
	test_copy_4();
	test_copy_5();
	test_copy_6();
}


template<size_t N, typename T, typename Core>
void anon_eval_test::invoke_eval(
	const char *testname,
	const labeled_btensor_expr::expr<N, T, Core> &expr,
	const letter_expr<N> &label, block_tensor_i<N, T> &ref)
	throw(libtest::test_exception) {

	labeled_btensor_expr::anon_eval<N, T, Core> ev(expr, label);
	ev.evaluate();
	compare_ref<N>::compare(testname, ev.get_btensor(), ref, 1e-15);
}


void anon_eval_test::test_copy_1() throw(libtest::test_exception) {

	//
	//	Simple copy, no symmetry
	//	q(i|j|a|b) = p(i|j|a|b)
	//

	static const char *testname = "anon_eval_test::test_copy_1()";
	typedef libvmm::std_allocator<double> allocator_t;

	try {

	bispace<1> si(5), sj(5), sa(10), sb(10);
	si.split(3);
	sj.split(3);
	sa.split(6);
	sb.split(6);
	bispace<4> sijab(si|sj|sa|sb);

	btensor<4> tp(sijab);
	btod_random<4>().perform(tp);

	block_tensor<4, double, allocator_t> tp_ref(sijab.get_bis());
	btod_copy<4>(tp).perform(tp_ref);

	letter i, j, a, b;
	invoke_eval(testname, 1.0*tp(i|j|a|b), i|j|a|b, tp_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void anon_eval_test::test_copy_2() throw(libtest::test_exception) {

	//
	//	Permuted copy, no symmetry
	//	q(i|a|j|b) = p(i|j|a|b)
	//

	static const char *testname = "anon_eval_test::test_copy_2()";
	typedef libvmm::std_allocator<double> allocator_t;

	try {

	bispace<1> si(5), sj(5), sa(10), sb(10);
	si.split(3);
	sj.split(3);
	sa.split(6);
	sb.split(6);
	bispace<4> sijab(si|sj|sa|sb), siajb(si|sa|sj|sb);

	btensor<4> tp(sijab);
	btod_random<4>().perform(tp);

	block_tensor<4, double, allocator_t> tp_ref(siajb.get_bis());
	permutation<4> perm;
	perm.permute(1, 2); // ijab -> iajb
	btod_copy<4>(tp, perm).perform(tp_ref);

	letter i, j, a, b;
	invoke_eval(testname, 1.0*tp(i|j|a|b), i|a|j|b, tp_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void anon_eval_test::test_copy_3() throw(libtest::test_exception) {

	//
	//	Scaled copy, no symmetry
	//	q(i|j|a|b) = p(i|j|a|b)*1.5
	//

	static const char *testname = "anon_eval_test::test_copy_3()";
	typedef libvmm::std_allocator<double> allocator_t;

	try {

	bispace<1> si(5), sj(5), sa(10), sb(10);
	si.split(3);
	sj.split(3);
	sa.split(6);
	sb.split(6);
	bispace<4> sijab(si|sj|sa|sb);

	btensor<4> tp(sijab);
	btod_random<4>().perform(tp);

	block_tensor<4, double, allocator_t> tp_ref(sijab.get_bis());
	btod_copy<4>(tp, 1.5).perform(tp_ref);

	letter i, j, a, b;
	invoke_eval(testname, tp(i|j|a|b)*1.5, i|j|a|b, tp_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void anon_eval_test::test_copy_4() throw(libtest::test_exception) {

	//
	//	Scaled permuted copy, no symmetry
	//	q(i|a|j|b) = -1.5*p(i|j|a|b)
	//

	static const char *testname = "anon_eval_test::test_copy_4()";
	typedef libvmm::std_allocator<double> allocator_t;

	try {

	bispace<1> si(5), sj(5), sa(10), sb(10);
	si.split(3);
	sj.split(3);
	sa.split(6);
	sb.split(6);
	bispace<4> sijab(si|sj|sa|sb), siajb(si|sa|sj|sb);

	btensor<4> tp(sijab);
	btod_random<4>().perform(tp);

	block_tensor<4, double, allocator_t> tp_ref(siajb.get_bis());
	permutation<4> perm;
	perm.permute(1, 2); // ijab -> iajb
	btod_copy<4>(tp, perm, -1.5).perform(tp_ref);

	letter i, j, a, b;
	invoke_eval(testname, -1.5*tp(i|j|a|b), i|a|j|b, tp_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void anon_eval_test::test_copy_5() throw(libtest::test_exception) {

	//
	//	Simple copy, permutational symmetry
	//	q(i|j|a|b) = p(i|j|a|b)
	//

	static const char *testname = "anon_eval_test::test_copy_5()";
	typedef libvmm::std_allocator<double> allocator_t;

	try {

	bispace<1> si(5), sj(5), sa(10), sb(10);
	si.split(3);
	sj.split(3);
	sa.split(6);
	sb.split(6);
	bispace<4> sijab(si|sj|sa|sb);

	btensor<4> tp(sijab);
	mask<4> msk1, msk2;
	msk1[0] = true; msk1[1] = true;
	msk2[2] = true; msk2[3] = true;
	symel_cycleperm<4, double> cycle1(2, msk1), cycle2(2, msk2);
	block_tensor_ctrl<4, double> ctrl(tp);
	ctrl.req_sym_add_element(cycle1);
	ctrl.req_sym_add_element(cycle2);
	btod_random<4>().perform(tp);

	block_tensor<4, double, allocator_t> tp_ref(sijab.get_bis());
	btod_copy<4>(tp).perform(tp_ref);

	letter i, j, a, b;
	invoke_eval(testname, 1.0*tp(i|j|a|b), i|j|a|b, tp_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void anon_eval_test::test_copy_6() throw(libtest::test_exception) {

	//
	//	Permuted copy, permutational symmetry
	//	q(i|a|j|b) = p(i|j|a|b)
	//

	static const char *testname = "anon_eval_test::test_copy_6()";
	typedef libvmm::std_allocator<double> allocator_t;

	try {

	bispace<1> si(5), sj(5), sa(10), sb(10);
	si.split(3);
	sj.split(3);
	sa.split(6);
	sb.split(6);
	bispace<4> sijab(si|sj|sa|sb), siajb(si|sa|sj|sb);

	btensor<4> tp(sijab);
	mask<4> msk1, msk2;
	msk1[0] = true; msk1[1] = true;
	msk2[2] = true; msk2[3] = true;
	symel_cycleperm<4, double> cycle1(2, msk1), cycle2(2, msk2);
	block_tensor_ctrl<4, double> ctrl(tp);
	ctrl.req_sym_add_element(cycle1);
	ctrl.req_sym_add_element(cycle2);
	btod_random<4>().perform(tp);

	block_tensor<4, double, allocator_t> tp_ref(siajb.get_bis());
	permutation<4> perm;
	perm.permute(1, 2); // ijab -> iajb
	btod_copy<4>(tp, perm).perform(tp_ref);

	letter i, j, a, b;
	invoke_eval(testname, 1.0*tp(i|j|a|b), i|a|j|b, tp_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
