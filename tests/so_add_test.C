#include <typeinfo>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_add.h>
#include <libtensor/btod/transf_double.h>
#include "so_add_test.h"
#include "compare_ref.h"

namespace libtensor {


void so_add_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_4();
	test_5();
	test_6();
}


/**	\test Empty + empty %symmetries in 4-space.
 **/
void so_add_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_add_test::test_1()";

	try {

	index<4> i1, i2;
	i2[0] = 5; i2[1] = 5; i2[2] = 10; i2[3] = 10;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));

	symmetry<4, double> sym1(bis), sym2(bis), sym3(bis);
	permutation<4> perm1, perm2;

	so_add<4, double>(sym1, perm1, sym2, perm2).perform(sym3);

	symmetry<4, double>::iterator j3 = sym3.begin();
	if(j3 != sym3.end()) {
		fail_test(testname, __FILE__, __LINE__, "j3 != sym3.end()");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Empty + non-empty perm %symmetry in 4-space.
 **/
void so_add_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "so_add_test::test_2()";

	typedef symmetry_element_set<4, double> symmetry_element_set_t;

	try {

	index<4> i1, i2;
	i2[0] = 5; i2[1] = 5; i2[2] = 10; i2[3] = 10;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));

	symmetry<4, double> sym1(bis), sym2(bis), sym3(bis);
	permutation<4> perm0;

	permutation<4> p1; p1.permute(0, 1);
	se_perm<4, double> e1(p1, true);
	sym2.insert(e1);

	so_add<4, double>(sym1, perm0, sym2, perm0).perform(sym3);

	symmetry<4, double>::iterator j3 = sym3.begin();
	if(j3 != sym3.end()) {
		fail_test(testname, __FILE__, __LINE__, "j3 != sym3.end()");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Non-empty + empty perm %symmetry in 4-space.
 **/
void so_add_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "so_add_test::test_3()";

	try {

	index<4> i1, i2;
	i2[0] = 5; i2[1] = 5; i2[2] = 10; i2[3] = 10;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));

	symmetry<4, double> sym1(bis), sym2(bis), sym3(bis);
	permutation<4> perm0;

	permutation<4> p1; p1.permute(0, 1);
	se_perm<4, double> e1(p1, true);
	sym1.insert(e1);

	so_add<4, double>(sym1, perm0, sym2, perm0).perform(sym3);

	symmetry<4, double>::iterator j3 = sym3.begin();
	if(j3 != sym3.end()) {
		fail_test(testname, __FILE__, __LINE__, "j3 != sym3.end()");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Non-empty + non-empty non-overlapping perm %symmetry in 4-space.
 **/
void so_add_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "so_add_test::test_4()";

	try {

	index<4> i1, i2;
	i2[0] = 5; i2[1] = 5; i2[2] = 10; i2[3] = 10;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));

	symmetry<4, double> sym1(bis), sym2(bis), sym3(bis);
	permutation<4> perm0;

	permutation<4> p1, p2;
	p1.permute(0, 1);
	p2.permute(2, 3);
	se_perm<4, double> e1(p1, true), e2(p2, true);
	sym1.insert(e1);
	sym2.insert(e2);

	so_add<4, double>(sym1, perm0, sym2, perm0).perform(sym3);

	symmetry<4, double>::iterator j3 = sym3.begin();
	if(j3 != sym3.end()) {
		fail_test(testname, __FILE__, __LINE__, "j3 != sym3.end()");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Non-empty + non-empty overlapping perm %symmetry in 4-space.
 **/
void so_add_test::test_5() throw(libtest::test_exception) {

	static const char *testname = "so_add_test::test_5()";

	typedef se_perm<4, double> se_perm_t;
	typedef symmetry_element_set<4, double> symmetry_element_set_t;

	try {

	index<4> i1, i2;
	i2[0] = 5; i2[1] = 5; i2[2] = 5; i2[3] = 5;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));

	symmetry<4, double> sym1(bis), sym2(bis), sym3(bis);
	permutation<4> perm0;

	permutation<4> p1, p2, p3;
	p1.permute(0, 1).permute(1, 2).permute(2, 3);
	p2.permute(2, 3);
	p3.permute(1, 2);
	se_perm<4, double> e1(p1, true), e2(p2, true), e3(p3, true);
	sym1.insert(e1);
	sym1.insert(e2);
	sym2.insert(e3);

	so_add<4, double>(sym1, perm0, sym2, perm0).perform(sym3);

	symmetry<4, double>::iterator j3 = sym3.begin();
	if(j3 == sym3.end()) {
		fail_test(testname, __FILE__, __LINE__, "j3 == sym3.end()");
	}

	const symmetry_element_set_t &subset31 = sym3.get_subset(j3);
	symmetry_element_set_t::const_iterator jj3 = subset31.begin();
	if(jj3 == subset31.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"jj3 == subset31.end()");
	}
	try {
		const se_perm_t &elem1i = dynamic_cast<const se_perm_t&>(
			subset31.get_elem(jj3));
		if(!elem1i.get_perm().equals(p3)) {
			fail_test(testname, __FILE__, __LINE__,
				"Bad permutation in elem1i");
		}
	} catch(std::bad_cast &e) {
		fail_test(testname, __FILE__, __LINE__, "bad_cast for elem1i");
	}

	jj3++;
	if(jj3 != subset31.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"jj3 != subset31.end()");
	}

	j3++;
	if(j3 != sym3.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one subset in sym3.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test S2*S2 + permuted S2*S2 in 4-space. Expected result --
		identity group.
 **/
void so_add_test::test_6() throw(libtest::test_exception) {

	static const char *testname = "so_add_test::test_6()";

	try {

	index<4> i1, i2;
	i2[0] = 5; i2[1] = 5; i2[2] = 5; i2[3] = 5;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
	mask<4> m;
	m[0] = true; m[1] = true; m[2] = true; m[3] = true;
	bis.split(m, 2);
	bis.split(m, 3);

	symmetry<4, double> sym1(bis), sym2(bis), sym2_ref(bis);

	sym1.insert(se_perm<4, double>(permutation<4>().permute(0, 1), true));
	sym1.insert(se_perm<4, double>(permutation<4>().permute(2, 3), true));

	so_add<4, double>(sym1, permutation<4>(), sym1,
		permutation<4>().permute(0, 2)).perform(sym2);

	compare_ref<4>::compare(testname, sym2, sym2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


} // namespace libtensor
