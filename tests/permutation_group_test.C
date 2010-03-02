#include <libtensor/btod/transf_double.h>
#include <libtensor/symmetry/permutation_group.h>
#include "permutation_group_test.h"

namespace libtensor {


void permutation_group_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2a();
	test_2b();
	test_3();
	test_4();
	test_5();
	test_6();

	test_project_down_1();
	test_project_down_2();
}


/**	\test Tests the C1 group in a 4-space
 **/
void permutation_group_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "permutation_group_test::test_1()";

	typedef se_perm<4, double> se_perm_t;

	try {

	permutation_group<4, double> pg;
	symmetry_element_set<4, double> set(se_perm_t::k_sym_type);
	pg.convert(set);
	if(!set.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "!set.is_empty()");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the S2(+) group in a 2-space
 **/
void permutation_group_test::test_2a() throw(libtest::test_exception) {

	static const char *testname = "permutation_group_test::test_2a()";

	typedef se_perm<2, double> se_perm_t;

	try {

	permutation<2> perm1; perm1.permute(0, 1);

	symmetry_element_set<2, double> set1(se_perm_t::k_sym_type),
		set2(se_perm_t::k_sym_type);

	set1.insert(se_perm_t(perm1, true));
	permutation_group<2, double> pg(set1);
	
	pg.convert(set2);
	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "set2.is_empty()");
	}

	typedef symmetry_element_set_adapter<2, double, se_perm_t> adapter_t;
	adapter_t adapter(set2);
	adapter_t::iterator i = adapter.begin();
	const se_perm_t &e1 = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one element.");
	}

	permutation<2> p1; p1.permute(0, 1);
	if(!e1.is_symm()) {
		fail_test(testname, __FILE__, __LINE__, "!e1.is_symm()");
	}
	if(!p1.equals(e1.get_perm())) {
		fail_test(testname, __FILE__, __LINE__, "p1 != e1.get_perm()");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the S2(-) group in a 2-space
 **/
void permutation_group_test::test_2b() throw(libtest::test_exception) {

	static const char *testname = "permutation_group_test::test_2b()";

	typedef se_perm<2, double> se_perm_t;

	try {

	permutation<2> perm1; perm1.permute(0, 1);

	symmetry_element_set<2, double> set1(se_perm_t::k_sym_type),
		set2(se_perm_t::k_sym_type);

	set1.insert(se_perm_t(perm1, false));
	permutation_group<2, double> pg(set1);
	
	pg.convert(set2);
	if(set2.is_empty()) {
		fail_test(testname, __FILE__, __LINE__, "set2.is_empty()");
	}

	typedef symmetry_element_set_adapter<2, double, se_perm_t> adapter_t;
	adapter_t adapter(set2);
	adapter_t::iterator i = adapter.begin();
	const se_perm_t &e1 = adapter.get_elem(i);
	i++;
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one element.");
	}

	permutation<2> p1; p1.permute(0, 1);
	if(e1.is_symm()) {
		fail_test(testname, __FILE__, __LINE__, "e1.is_symm()");
	}
	if(!p1.equals(e1.get_perm())) {
		fail_test(testname, __FILE__, __LINE__, "p1 != e1.get_perm()");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the S3(+) group in a 3-space. The group is created using
		[012->120] and [012->102].
 **/
void permutation_group_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "permutation_group_test::test_3()";

	//~ typedef se_perm<3, double> se_perm_t;

	//~ try {

	//~ permutation_group<3, double> pg;

	//~ permutation<3> perm1; perm1.permute(0, 1).permute(1, 2);
	//~ permutation<3> perm2; perm2.permute(0, 1);
	//~ pg.join_permutation(perm1, true);
	//~ pg.join_permutation(perm2, true);

	//~ symmetry_element_set<3, double> set(se_perm_t::k_sym_type);
	//~ pg.convert(set);
	//~ if(set.is_empty()) {
		//~ fail_test(testname, __FILE__, __LINE__, "set.is_empty()");
	//~ }

	//~ typedef symmetry_element_set_adapter<3, double, se_perm_t> adapter_t;
	//~ adapter_t adapter(set);
	//~ adapter_t::iterator i = adapter.begin();
	//~ const se_perm_t &e1 = adapter.get_elem(i);
	//~ i++;
	//~ if(i == adapter.end()) {
		//~ fail_test(testname, __FILE__, __LINE__,
			//~ "Expected two elements, only one found.");
	//~ }
	//~ i++;
	//~ const se_perm_t &e2 = adapter.get_elem(i);
	//~ if(i != adapter.end()) {
		//~ fail_test(testname, __FILE__, __LINE__,
			//~ "Expected two elements, found more.");
	//~ }

	//~ permutation<3> p1; p1.permute(0, 1).permute(1, 2);
	//~ permutation<3> p2; p2.permute(0, 1);
	//~ if(!e1.is_symm()) {
		//~ fail_test(testname, __FILE__, __LINE__, "!e1.is_symm()");
	//~ }
	//~ if(!e2.is_symm()) {
		//~ fail_test(testname, __FILE__, __LINE__, "!e2.is_symm()");
	//~ }
	//~ if(p1.equals(e1.get_perm())) {
		//~ if(!p2.equals(e2.get_perm())) {
			//~ fail_test(testname, __FILE__, __LINE__,
				//~ "p1 == e1.get_perm(), p2 != e2.get_perm()");
		//~ }
	//~ } else {
	//~ if(p1.equals(e2.get_perm())) {
		//~ if(!p2.equals(e1.get_perm())) {
			//~ fail_test(testname, __FILE__, __LINE__,
				//~ "p1 == e2.get_perm(), p2 != e1.get_perm()");
		//~ }
	//~ } else {
		//~ fail_test(testname, __FILE__, __LINE__,
			//~ "p1 != e1.get_perm(), p1 != e2.get_perm()");
	//~ }
	//~ }

	//~ } catch(exception &e) {
		//~ fail_test(testname, __FILE__, __LINE__, e.what());
	//~ }
}


/**	\test Tests the A4(+) group in a 4-space
 **/
void permutation_group_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "permutation_group_test::test_4()";

	typedef se_perm<4, double> se_perm_t;

	try {

	permutation<4> perm1; perm1.permute(0, 1).permute(2, 3);
	permutation<4> perm2; perm2.permute(0, 1).permute(1, 2);

	symmetry_element_set<4, double> set1(se_perm_t::k_sym_type),
		set2(se_perm_t::k_sym_type);

	set1.insert(se_perm_t(perm1, true));
	set1.insert(se_perm_t(perm2, true));
	permutation_group<4, double> pg(set1);

	if(!pg.is_member(true, perm1)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg.is_member(true, perm1)");
	}
	if(!pg.is_member(true, perm2)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg.is_member(true, perm2)");
	}
	
	permutation<4> perm3; perm3.permute(0, 2).permute(1, 3);
	if(!pg.is_member(true, perm3)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg.is_member(true, perm3)");
	}

	permutation<4> perm4; perm4.permute(1, 2);
	if(pg.is_member(true, perm4)) {
		fail_test(testname, __FILE__, __LINE__,
			"pg.is_member(true, perm4)");
	}

	permutation<4> perm5; perm5.permute(0, 2);
	if(pg.is_member(true, perm5)) {
		fail_test(testname, __FILE__, __LINE__,
			"pg.is_member(true, perm5)");
	}


	//~ pg.convert(set2);
	//~ if(set2.is_empty()) {
		//~ fail_test(testname, __FILE__, __LINE__, "set2.is_empty()");
	//~ }

	//~ typedef symmetry_element_set_adapter<2, double, se_perm_t> adapter_t;
	//~ adapter_t adapter(set2);
	//~ adapter_t::iterator i = adapter.begin();
	//~ const se_perm_t &e1 = adapter.get_elem(i);
	//~ i++;
	//~ if(i != adapter.end()) {
		//~ fail_test(testname, __FILE__, __LINE__,
			//~ "Expected only one element.");
	//~ }

	//~ permutation<2> p1; p1.permute(0, 1);
	//~ if(!e1.is_symm()) {
		//~ fail_test(testname, __FILE__, __LINE__, "!e1.is_symm()");
	//~ }
	//~ if(!p1.equals(e1.get_perm())) {
		//~ fail_test(testname, __FILE__, __LINE__, "p1 != e1.get_perm()");
	//~ }

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the S2(+)*S2(+) group in a 4-space
 **/
void permutation_group_test::test_5() throw(libtest::test_exception) {

	static const char *testname = "permutation_group_test::test_5()";

	typedef se_perm<4, double> se_perm_t;

	try {

	permutation<4> perm1; perm1.permute(0, 1).permute(2, 3);
	permutation<4> perm2; perm2.permute(0, 1);

	symmetry_element_set<4, double> set1(se_perm_t::k_sym_type),
		set2(se_perm_t::k_sym_type);

	set1.insert(se_perm_t(perm1, true));
	set1.insert(se_perm_t(perm2, true));
	permutation_group<4, double> pg(set1);

	if(!pg.is_member(true, perm1)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg.is_member(true, perm1)");
	}
	if(!pg.is_member(true, perm2)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg.is_member(true, perm2)");
	}
	
	permutation<4> perm3; perm3.permute(2, 3);
	if(!pg.is_member(true, perm3)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg.is_member(true, perm3)");
	}

	permutation<4> perm4; perm4.permute(1, 2);
	if(pg.is_member(true, perm4)) {
		fail_test(testname, __FILE__, __LINE__,
			"pg.is_member(true, perm4)");
	}

	permutation<4> perm5; perm5.permute(0, 2);
	if(pg.is_member(true, perm5)) {
		fail_test(testname, __FILE__, __LINE__,
			"pg.is_member(true, perm5)");
	}

	permutation<4> perm6; perm6.permute(2, 3).permute(1, 2).permute(0, 1);
	if(pg.is_member(true, perm6)) {
		fail_test(testname, __FILE__, __LINE__,
			"pg.is_member(true, perm6)");
	}


	//~ pg.convert(set2);
	//~ if(set2.is_empty()) {
		//~ fail_test(testname, __FILE__, __LINE__, "set2.is_empty()");
	//~ }

	//~ typedef symmetry_element_set_adapter<2, double, se_perm_t> adapter_t;
	//~ adapter_t adapter(set2);
	//~ adapter_t::iterator i = adapter.begin();
	//~ const se_perm_t &e1 = adapter.get_elem(i);
	//~ i++;
	//~ if(i != adapter.end()) {
		//~ fail_test(testname, __FILE__, __LINE__,
			//~ "Expected only one element.");
	//~ }

	//~ permutation<2> p1; p1.permute(0, 1);
	//~ if(!e1.is_symm()) {
		//~ fail_test(testname, __FILE__, __LINE__, "!e1.is_symm()");
	//~ }
	//~ if(!p1.equals(e1.get_perm())) {
		//~ fail_test(testname, __FILE__, __LINE__, "p1 != e1.get_perm()");
	//~ }

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the S4(+) group in a 4-space
 **/
void permutation_group_test::test_6() throw(libtest::test_exception) {

	static const char *testname = "permutation_group_test::test_6()";

	typedef se_perm<4, double> se_perm_t;

	try {

	permutation<4> perm1; perm1.permute(0, 1).permute(1, 2).permute(2, 3);
	permutation<4> perm2; perm2.permute(0, 1);

	symmetry_element_set<4, double> set1(se_perm_t::k_sym_type),
		set2(se_perm_t::k_sym_type);

	set1.insert(se_perm_t(perm1, true));
	set1.insert(se_perm_t(perm2, true));
	permutation_group<4, double> pg(set1);

	if(!pg.is_member(true, perm1)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg.is_member(true, perm1)");
	}
	if(!pg.is_member(true, perm2)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg.is_member(true, perm2)");
	}
	
	permutation<4> perm3; perm3.permute(0, 2).permute(1, 3);
	if(!pg.is_member(true, perm3)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg.is_member(true, perm3)");
	}

	permutation<4> perm4; perm4.permute(1, 2);
	if(!pg.is_member(true, perm4)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg.is_member(true, perm4)");
	}

	permutation<4> perm5; perm5.permute(0, 2);
	if(!pg.is_member(true, perm5)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg.is_member(true, perm5)");
	}

	permutation<4> perm6; perm6.permute(2, 3).permute(1, 2).permute(0, 1);
	if(!pg.is_member(true, perm6)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg.is_member(true, perm6)");
	}

	//	0123->3210
	permutation<4> perm7; perm7.permute(0, 3).permute(1, 2);
	if(!pg.is_member(true, perm7)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg.is_member(true, perm7)");
	}


	//~ pg.convert(set2);
	//~ if(set2.is_empty()) {
		//~ fail_test(testname, __FILE__, __LINE__, "set2.is_empty()");
	//~ }

	//~ typedef symmetry_element_set_adapter<2, double, se_perm_t> adapter_t;
	//~ adapter_t adapter(set2);
	//~ adapter_t::iterator i = adapter.begin();
	//~ const se_perm_t &e1 = adapter.get_elem(i);
	//~ i++;
	//~ if(i != adapter.end()) {
		//~ fail_test(testname, __FILE__, __LINE__,
			//~ "Expected only one element.");
	//~ }

	//~ permutation<2> p1; p1.permute(0, 1);
	//~ if(!e1.is_symm()) {
		//~ fail_test(testname, __FILE__, __LINE__, "!e1.is_symm()");
	//~ }
	//~ if(!p1.equals(e1.get_perm())) {
		//~ fail_test(testname, __FILE__, __LINE__, "p1 != e1.get_perm()");
	//~ }

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the projection of the S4(+) group in a 4-space onto
		a 2-space, S2(+)
 **/
void permutation_group_test::test_project_down_1()
	throw(libtest::test_exception) {

	static const char *testname =
		"permutation_group_test::test_project_down_1()";

	typedef se_perm<4, double> se_perm_t;

	try {

	permutation<4> perm1; perm1.permute(0, 1).permute(1, 2).permute(2, 3);
	permutation<4> perm2; perm2.permute(0, 1);

	symmetry_element_set<4, double> set1(se_perm_t::k_sym_type),
		set2(se_perm_t::k_sym_type);

	set1.insert(se_perm_t(perm1, true));
	set1.insert(se_perm_t(perm2, true));
	permutation_group<4, double> pg4(set1);

	permutation_group<2, double> pg2;
	mask<4> msk; msk[1] = true; msk[3] = true;
	pg4.project_down(msk, pg2);

	permutation<2> p2_1, p2_2; p2_2.permute(0, 1);

	if(!pg2.is_member(true, p2_1)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg2.is_member(true, p2_1)");
	}
	if(!pg2.is_member(true, p2_2)) {
		fail_test(testname, __FILE__, __LINE__,
			"!pg2.is_member(true, p2_2)");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the projection of the C4(+) group in a 4-space onto
		a 2-space, C1
 **/
void permutation_group_test::test_project_down_2()
	throw(libtest::test_exception) {

	static const char *testname =
		"permutation_group_test::test_project_down_2()";

	typedef se_perm<4, double> se_perm_t;

	try {

	permutation<4> perm1; perm1.permute(0, 1).permute(1, 2).permute(2, 3);

	symmetry_element_set<4, double> set1(se_perm_t::k_sym_type);
	set1.insert(se_perm_t(perm1, true));
	permutation_group<4, double> pg4(set1);

	permutation_group<2, double> pg2;
	mask<4> msk; msk[0] = true; msk[1] = true;
	pg4.project_down(msk, pg2);

	permutation<2> p2_1, p2_2; p2_2.permute(0, 1);

	if(pg2.is_member(true, p2_1)) {
		fail_test(testname, __FILE__, __LINE__,
			"pg2.is_member(true, p2_1)");
	}
	if(pg2.is_member(true, p2_2)) {
		fail_test(testname, __FILE__, __LINE__,
			"pg2.is_member(true, p2_2)");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
