#include <libvmm/std_allocator.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/btod/btod_set_diag.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/tod/tod_btconv.h>
#include "btod_set_diag_test.h"
#include "../compare_ref.h"

namespace libtensor {


void btod_set_diag_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_4();
}


void btod_set_diag_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "btod_set_diag_test::test_1()";

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	symmetry<2, double> sym(bis);

	test_generic(testname, bis, sym, 0.0);
	test_generic(testname, bis, sym, 11.5);
}


void btod_set_diag_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "btod_set_diag_test::test_2()";

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);

	mask<2> m;
	m[0] = true; m[1] = true;
	bis.split(m, 3);

	symmetry<2, double> sym(bis);

	test_generic(testname, bis, sym, 0.0);
	test_generic(testname, bis, sym, 11.6);
}


void btod_set_diag_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "btod_set_diag_test::test_3()";

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);

	mask<2> m;
	m[0] = true; m[1] = true;
	bis.split(m, 3);
	bis.split(m, 8);

	symmetry<2, double> sym(bis);

	test_generic(testname, bis, sym, 0.0);
	test_generic(testname, bis, sym, 11.7);
}


void btod_set_diag_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "btod_set_diag_test::test_4()";

	index<4> i1, i2;
	i2[0] = 10; i2[1] = 10; i2[2] = 10; i2[3] = 10;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);

	mask<4> m;
	m[0] = true; m[1] = true; m[2] = true; m[3] = true;
	bis.split(m, 3);
	bis.split(m, 8);

	symmetry<4, double> sym(bis);
	se_perm<4, double> elem1(permutation<4>().permute(0, 1).permute(1, 2).
		permute(2, 3), true);
	se_perm<4, double> elem2(permutation<4>().permute(0, 1), true);
	sym.insert(elem1);
	sym.insert(elem2);

	test_generic(testname, bis, sym, -2.0);
	test_generic(testname, bis, sym, 0.12);
}


template<size_t N>
void btod_set_diag_test::test_generic(const char *testname,
	const block_index_space<N> &bis, const symmetry<N, double> &sym,
	double d) throw(libtest::test_exception) {

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	block_tensor<N, double, allocator_t> bt(bis);
	tensor<N, double, allocator_t> t(bis.get_dims()), t_ref(bis.get_dims());

	//	Fill in random data & make reference

	{
		block_tensor_ctrl<N, double> ctrl(bt);
		so_copy<N, double>(sym).perform(ctrl.req_symmetry());
	}
	btod_random<N>().perform(bt);
	tod_btconv<N>(bt).perform(t_ref);
	tod_set_diag<N>(d).perform(t_ref);

	//	Perform the operation

	btod_set_diag<N>(d).perform(bt);
	tod_btconv<N>(bt).perform(t);

	//	Compare against the reference

	compare_ref<N>::compare(testname, t, t_ref, 0.0);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
