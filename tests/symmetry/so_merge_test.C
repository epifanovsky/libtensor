#include <libtensor/btod/scalar_transf_double.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_merge.h>
#include "../compare_ref.h"
#include "so_merge_test.h"

namespace libtensor {


void so_merge_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
}


/**	\test Invokes merge of 2 dimensions of C1 in 4-space onto 3-space.
		Expects C1 in 3-space.
 **/
void so_merge_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_merge_test::test_1()";

	try {

	index<4> i1a, i1b;
	i1b[0] = 5; i1b[1] = 5; i1b[2] = 10; i1b[3] = 10;
	index<3> i2a, i2b;
	i2b[0] = 5; i2b[1] = 5; i2b[2] = 10;
	block_index_space<4> bis1(dimensions<4>(index_range<4>(i1a, i1b)));
	block_index_space<3> bis2(dimensions<3>(index_range<3>(i2a, i2b)));

	symmetry<4, double> sym1(bis1);
	symmetry<3, double> sym2(bis2);
	symmetry<3, double> sym2_ref(bis2);
	mask<4> msk; msk[2] = true; msk[3] = true;
	sequence<4, size_t> seq(0);
	so_merge<4, 1, double>(sym1, msk, seq).perform(sym2);

	symmetry<3, double>::iterator i = sym2.begin();
	if(i != sym2.end()) {
		fail_test(testname, __FILE__, __LINE__, "i != sym2.end()");
	}
	compare_ref<3>::compare(testname, sym2, sym2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Invokes a merge of 3 dim in S5(+) in 5-space onto 3-space.
		Expects S3(+) in 3-space.
 **/
void so_merge_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "so_merge_test::test_2()";

	try {

	index<5> i1a, i1b;
	i1b[0] = 5; i1b[1] = 5; i1b[2] = 10; i1b[3] = 10; i1b[4] = 10;
	index<3> i2a, i2b;
	i2b[0] = 5; i2b[1] = 5; i2b[2] = 10;
	block_index_space<5> bis1(dimensions<5>(index_range<5>(i1a, i1b)));
	block_index_space<3> bis2(dimensions<3>(index_range<3>(i2a, i2b)));

	symmetry<5, double> sym1(bis1);
	permutation<5> p1a, p1b;
	p1a.permute(0, 1).permute(1, 2).permute(2, 3).permute(3, 4);
	p1b.permute(0, 1);
	scalar_transf<double> tr0;
	sym1.insert(se_perm<5, double>(p1a, tr0));
	sym1.insert(se_perm<5, double>(p1b, tr0));

	symmetry<3, double> sym2(bis2);
	symmetry<3, double> sym2_ref(bis2);
	mask<5> msk;
	msk[2] = true; msk[3] = true; msk[4] = true;
	sequence<5, size_t> seq(0);
	so_merge<5, 2, double>(sym1, msk, seq).perform(sym2);

	permutation<3> p2a, p2b;
	p2a.permute(0, 1).permute(1, 2);
	p2b.permute(0, 1);
	sym2_ref.insert(se_perm<3, double>(p2a, tr0));
	sym2_ref.insert(se_perm<3, double>(p2b, tr0));

	symmetry<3, double>::iterator i = sym2.begin();
	if(i == sym2.end()) {
		fail_test(testname, __FILE__, __LINE__, "i == sym2.end()");
	}
	compare_ref<3>::compare(testname, sym2, sym2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}

/**	\test Invokes merge of 2 dims of S2(+) onto 1-space.
 **/
void so_merge_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "so_merge_test::test_3()";

	try {

	index<2> i2a, i2b;
	i2b[0] = 5; i2b[1] = 5;
	index<1> i1a, i1b;
	i1b[0] = 5;
	block_index_space<2> bis1(dimensions<2>(index_range<2>(i2a, i2b)));
	block_index_space<1> bis2(dimensions<1>(index_range<1>(i1a, i1b)));

	mask<2> m;
	m[0] = true; m[1] = true;
	sequence<2, size_t> seq(0);
	symmetry<2, double> sym1(bis1);
	symmetry<1, double> sym2(bis2);
	so_merge<2, 1, double>(sym1, m, seq).perform(sym2);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}

/**	\test Invokes a projection of S2(+) onto 2-space.
 **/
void so_merge_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "so_merge_test::test_4()";

	try {

	index<2> i2a, i2b;
	i2b[0] = 5; i2b[1] = 5;
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	block_index_space<2> bis1(dims2);

	mask<2> msk;
	msk[0] = true;
	sequence<2, size_t> seq(0);
	symmetry<2, double> sym1(bis1);
	symmetry<2, double> sym2(bis1);
	so_merge<2, 0, double>(sym1, msk, seq).perform(sym2);

	symmetry<2, double>::iterator i = sym2.begin();
	if(i == sym2.end()) {
		fail_test(testname, __FILE__, __LINE__, "i == sym2.end()");
	}
	compare_ref<2>::compare(testname, sym2, sym1);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}

} // namespace libtensor
