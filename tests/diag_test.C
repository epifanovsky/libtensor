#include <libtensor/btod/btod_diag.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/iface/iface.h>
#include "compare_ref.h"
#include "diag_test.h"

namespace libtensor {


void diag_test::perform() throw(libtest::test_exception) {

	libvmm::vm_allocator<double>::vmm().init(
		16, 16, 16777216, 16777216, 0.90, 0.05);

	try {

		test_t_1();
		test_t_2();
		test_t_3();

	} catch(...) {
		libvmm::vm_allocator<double>::vmm().shutdown();
		throw;
	}

	libvmm::vm_allocator<double>::vmm().shutdown();
}


void diag_test::test_t_1() throw(libtest::test_exception) {

	static const char *testname = "diag_test::test_t_1()";

	try {

	bispace<1> sp_i(10);
	bispace<2> sp_ij(sp_i&sp_i);

	btensor<2> t1(sp_ij);
	btensor<1> t2(sp_i), t2_ref(sp_i);

	btod_random<2>().perform(t1);
	t1.set_immutable();

	mask<2> msk;
	msk[0] = true; msk[1] = true;
	permutation<1> perm;
	btod_diag<2, 2>(t1, msk, perm).perform(t2_ref);

	letter i, j, k;
	t2(k) = diag(k, i|j, t1(j|i));

	compare_ref<1>::compare(testname, t2, t2_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void diag_test::test_t_2() throw(libtest::test_exception) {

	static const char *testname = "diag_test::test_t_2()";

	try {

	bispace<1> sp_i(10), sp_a(11);
	bispace<2> sp_ia(sp_i|sp_a);
	bispace<3> sp_ija(sp_i&sp_i|sp_a);

	btensor<3> t1(sp_ija);
	btensor<2> t2(sp_ia), t2_ref(sp_ia);

	btod_random<3>().perform(t1);
	t1.set_immutable();

	mask<3> msk;
	msk[0] = true; msk[1] = true;
	permutation<2> perm;
	btod_diag<3, 2>(t1, msk, perm).perform(t2_ref);

	letter i, j, a;
	t2(i|a) = diag(i, i|j, t1(i|j|a));

	compare_ref<2>::compare(testname, t2, t2_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void diag_test::test_t_3() throw(libtest::test_exception) {

	static const char *testname = "diag_test::test_t_3()";

	try {

	bispace<1> sp_i(10), sp_a(11), sp_j(sp_i);
	bispace<2> sp_ai(sp_a|sp_i);
	bispace<3> sp_iaj(sp_i|sp_a|sp_j, sp_i&sp_j|sp_a);

	btensor<3> t1(sp_iaj);
	btensor<2> t2(sp_ai), t2_ref(sp_ai);

	btod_random<3>().perform(t1);
	t1.set_immutable();

	mask<3> msk;
	msk[0] = true; msk[2] = true;
	permutation<2> perm;
	perm.permute(0, 1); // ia->ai
	btod_diag<3, 2>(t1, msk, perm).perform(t2_ref);

	letter i, j, a;
	t2(a|i) = diag(i, i|j, t1(i|a|j));

	compare_ref<2>::compare(testname, t2, t2_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
