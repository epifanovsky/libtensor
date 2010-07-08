#include <libtensor/btod/btod_random.h>
#include <libtensor/iface/iface.h>
#include "expr_test.h"
#include "compare_ref.h"

namespace libtensor {


void expr_test::perform() throw(libtest::test_exception) {

	libvmm::vm_allocator<double>::vmm().init(
		16, 16, 16777216, 16777216, 0.90, 0.05);

	try {

		test_1();
		test_2();
		test_3();
		test_4();
		test_5();
		test_6();

	} catch(...) {
		libvmm::vm_allocator<double>::vmm().shutdown();
		throw;
	}

	libvmm::vm_allocator<double>::vmm().shutdown();
}


void expr_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "expr_test::test_1()";

	try {

	bispace<1> so(13); so.split(3).split(7).split(10);
	bispace<1> sv(7); sv.split(2).split(3).split(5);

	bispace<2> sov(so|sv);

	btensor<2> t1(sov), t2(sov), t3(sov), t3_ref(sov);

	btod_random<2>().perform(t1);
	btod_random<2>().perform(t2);
	t1.set_immutable();
	t2.set_immutable();

	btod_add<2> op(t1);
	op.add_op(t2, -1.0);
	op.perform(t3_ref);

	letter i, a;

	t3(i|a) = t1(i|a) - t2(i|a);

	compare_ref<2>::compare(testname, t3, t3_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void expr_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "expr_test::test_2()";

	try {

	bispace<1> so(13); so.split(3).split(7).split(10);
	bispace<1> sv(7); sv.split(2).split(3).split(5);

	bispace<2> sov(so|sv);
	bispace<4> sooov(so&so&so|sv), soovv(so&so|sv&sv), sovvv(so|sv&sv&sv),
		svvvv(sv&sv&sv&sv);

	bispace<1> so1(so), so2(so), sv1(sv), sv2(sv);
	bispace<4> sovov(so1|sv1|so2|sv2, so1&so2|sv1&sv2);

	btensor<2> t1(sov);
	btensor<4> t2(soovv);
	btensor<2> f_ov(sov);
	btensor<4> i_ooov(sooov), i_oovv(soovv), i_ovov(sovov), i_ovvv(sovvv);
	btensor<4> i3_ovvv(sovvv), i5_vvvv(svvvv);

	btod_random<2>().perform(t1);
	btod_random<4>().perform(t2);
	btod_random<2>().perform(f_ov);
	btod_random<4>().perform(i_ooov);
	btod_random<4>().perform(i_oovv);
	btod_random<4>().perform(i_ovov);
	btod_random<4>().perform(i_ovvv);
	btod_random<4>().perform(i5_vvvv);

	letter i, j, k, l, a, b, c, d;

	i3_ovvv(i|a|b|c) =
		  i_ovvv(i|a|b|c)
		+ asymm(b|c, contract(j,
			t1(j|c),
			i_ovov(j|b|i|a)
			- contract(k|d, t2(i|k|b|d), i_oovv(j|k|a|d))))
		- asymm(b|c, contract(k|d, i_ovvv(k|c|a|d), t2(i|k|b|d)));

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void expr_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "expr_test::test_3()";

	try {

	bispace<1> so(10); so.split(5);
	bispace<1> sv(4); sv.split(2);

	bispace<2> sov(so|sv);
	bispace<4> sooov(so&so&so|sv), soovv(so&so|sv&sv), sovvv(so|sv&sv&sv),
		svvvv(sv&sv&sv&sv);

	bispace<1> so1(so), so2(so), sv1(sv), sv2(sv);
	bispace<4> sovov(so1|sv1|so2|sv2, so1&so2|sv1&sv2);

	btensor<2> t1(sov);
	btensor<4> t2(soovv);
	btensor<4> i1_ovov(sovov);

	{
		block_tensor_ctrl<4, double> c_t2(t2);
		symmetry<4, double> sym_t2(t2.get_bis());
		sym_t2.insert(se_perm<4, double>(permutation<4>().
			permute(0, 1), false));
		sym_t2.insert(se_perm<4, double>(permutation<4>().
			permute(2, 3), false));
		so_copy<4, double>(sym_t2).perform(c_t2.req_symmetry());
	}

	btod_random<2>().perform(t1);
	btod_random<4>().perform(t2);
	btod_random<4>().perform(i1_ovov);

	letter i, j, k, l, a, b, c, d;

	i1_ovov(i|b|k|c) = t2(i|k|b|c) - t1(i|c) * t1(k|b);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void expr_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "expr_test::test_4()";

	try {

	bispace<1> so(10); so.split(5);
	bispace<1> sv(4); sv.split(2);

	bispace<2> sov(so|sv);
	bispace<4> sooov(so&so&so|sv), soovv(so&so|sv&sv), sovvv(so|sv&sv&sv),
		svvvv(sv&sv&sv&sv);

	bispace<1> so1(so), so2(so), sv1(sv), sv2(sv);
	bispace<4> sovov(so1|sv1|so2|sv2, so1&so2|sv1&sv2);

	btensor<4> i_oovv(soovv), i_ovov(sovov), t2(soovv);
	btensor<4> i2_oovv(soovv);

	{
		block_tensor_ctrl<4, double> c_i_oovv(i_oovv), c_t2(t2);
		symmetry<4, double> sym_t2(t2.get_bis());
		sym_t2.insert(se_perm<4, double>(permutation<4>().
			permute(0, 1), false));
		sym_t2.insert(se_perm<4, double>(permutation<4>().
			permute(2, 3), false));
		so_copy<4, double>(sym_t2).perform(c_t2.req_symmetry());
		so_copy<4, double>(sym_t2).perform(c_i_oovv.req_symmetry());
	}
	{
		block_tensor_ctrl<4, double> c_i_ovov(i_ovov);
		symmetry<4, double> sym_i_ovov(i_ovov.get_bis());
		sym_i_ovov.insert(se_perm<4, double>(permutation<4>().
			permute(0, 2).permute(1, 3), true));
		so_copy<4, double>(sym_i_ovov).perform(c_i_ovov.req_symmetry());
	}

	btod_random<4>().perform(i_oovv);
	btod_random<4>().perform(i_ovov);
	btod_random<4>().perform(t2);

	letter i, j, k, l, a, b, c, d;

	i2_oovv(j|k|a|b) =
		i_ovov(k|a|j|b) - contract(l|c, i_oovv(k|l|b|c), t2(j|l|a|c));

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void expr_test::test_5() throw(libtest::test_exception) {

	static const char *testname = "expr_test::test_5()";

	try {

	bispace<1> so(10); so.split(5);
	bispace<1> sv(4); sv.split(2);

	bispace<2> soo(so&so);
	bispace<4> sooov(so&so&so|sv), soovv(so&so|sv&sv), sovvv(so|sv&sv&sv),
		svvvv(sv&sv&sv&sv);

	btensor<4> t1_oovv(soovv), t2_oovv(soovv);
	btensor<2> t3_oo(soo), t3_oo_ref(soo);

	{
		block_tensor_ctrl<4, double> c_t1_oovv(t1_oovv);
		c_t1_oovv.req_symmetry().insert(se_perm<4, double>(
			permutation<4>().permute(0, 1), false));
		c_t1_oovv.req_symmetry().insert(se_perm<4, double>(
			permutation<4>().permute(2, 3), false));
	}

	btod_random<4>().perform(t1_oovv);
	btod_random<4>().perform(t2_oovv);

	letter i, j, k, l, a, b, c, d;

	t3_oo(i|j) = contract(k|a|b, t1_oovv(j|k|a|b), t2_oovv(i|k|a|b));

	contraction2<1, 1, 3> contr(permutation<2>().permute(0, 1));
	contr.contract(1, 1);
	contr.contract(2, 2);
	contr.contract(3, 3);
	btod_contract2<1, 1, 3>(contr, t1_oovv, t2_oovv).perform(t3_oo_ref);
	compare_ref<2>::compare(testname, t3_oo, t3_oo_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void expr_test::test_6() throw(libtest::test_exception) {

	static const char *testname = "expr_test::test_6()";

	try {

	bispace<1> so(10); so.split(5);
	bispace<1> sv(4); sv.split(2);

	bispace<2> sov(so|sv);
	bispace<4> sooov(so&so&so|sv), soovv(so&so|sv&sv), sovvv(so|sv&sv&sv),
		svvvv(sv&sv&sv&sv);

	btensor<2> t1(sov), y1_ov(sov), t3_ov(sov), t3_ov_ref(sov);
	btensor<4> t2(soovv), tt_oovv(soovv), tt_oovv_ref(soovv);

	{
		block_tensor_ctrl<4, double> c_t2(t2);
		c_t2.req_symmetry().insert(se_perm<4, double>(
			permutation<4>().permute(0, 1), false));
		c_t2.req_symmetry().insert(se_perm<4, double>(
			permutation<4>().permute(2, 3), false));
	}

	btod_random<2>().perform(t1);
	btod_random<4>().perform(t2);
	btod_random<2>().perform(y1_ov);

	letter i, j, k, l, a, b, c, d;

	tt_oovv(i|j|a|b) = t2(i|j|a|b) - t1(j|a)*t1(i|b);
	t3_ov(i|a) = contract(j|b, t2(i|j|a|b) - t1(j|a)*t1(i|b), y1_ov(j|b));

	btod_copy<4>(t2).perform(tt_oovv_ref);
	contraction2<2, 2, 0> contr1(permutation<4>().
		permute(1, 2).permute(0, 1));
	btod_contract2<2, 2, 0>(contr1, t1, t1).perform(tt_oovv_ref, -1.0);

	contraction2<2, 0, 2> contr2;
	contr2.contract(1, 0);
	contr2.contract(3, 1);
	btod_contract2<2, 0, 2>(contr2, tt_oovv_ref, y1_ov).perform(t3_ov_ref);

	compare_ref<4>::compare(testname, tt_oovv, tt_oovv_ref, 1e-15);
	compare_ref<2>::compare(testname, t3_ov, t3_ov_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
