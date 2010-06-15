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


} // namespace libtensor
