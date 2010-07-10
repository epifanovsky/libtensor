#include <libtensor/btod/transf_double.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/se_label.h>
#include "se_label_test.h"

namespace libtensor {


void se_label_test::perform() throw(libtest::test_exception) {

	// Setup point_group_table and product_table_container

	try {

	point_group_table s6(4);
	point_group_table::label_t ag = 0, eg = 1, au = 2, eu = 3;
	s6.set_product(ag, ag, 0, ag);
	s6.set_product(eg, ag, 0, eg);
	s6.set_product(au, ag, 0, au);
	s6.set_product(eu, ag, 0, au);
	s6.set_product(eg, eg, 0, ag);
	s6.set_product(eg, eg, 1, eg);
	s6.set_product(au, eg, 0, eu);
	s6.set_product(eu, eg, 0, au);
	s6.set_product(eu, eg, 1, eu);
	s6.set_product(au, au, 0, ag);
	s6.set_product(eu, au, 0, eg);
	s6.set_product(eu, eu, 0, ag);
	s6.set_product(eu, eu, 1, eg);
	s6.check();
	product_table_container::get_instance().add(s6);

	} catch (exception &e) {
		fail_test("se_label_test::perform()", __FILE__, __LINE__,
				e.what());
	}

	try {

	test_1();
	test_2();
	test_3();
	test_4();

	product_table_container::get_instance().erase(point_group_table::k_id);

	} catch (libtest::test_exception) {
		product_table_container::get_instance().erase(point_group_table::k_id);
		throw;
	}
}


/**	\test Two blocks, all labeled (2-dim)
 **/
void se_label_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "se_label_test::test_1()";

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	mask<2> m11;
	m11[0] = true; m11[1] = true;
	bis.split(m11, 5);

	se_label<2, double> elem1(bis.get_block_index_dims(),
			point_group_table::k_id);
	elem1.assign(m11, 0, 0); // ag
	elem1.assign(m11, 1, 2); // au
	elem1.set_target(0);

	se_label<2, double> elem2(elem1), elem3(elem1), elem4(elem1);
	elem2.set_target(2);
	elem3.set_target(1);
	elem4.set_target(3);

	index<2> i00, i01, i10, i11;
	i01[0] = 0; i01[1] = 1;
	i10[0] = 1; i10[1] = 0;
	i11[0] = 1; i11[1] = 1;

	if(!elem1.is_allowed(i00)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem1.is_allowed(i00)");
	}
	if(elem1.is_allowed(i01)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem1.is_allowed(i01)");
	}
	if(elem1.is_allowed(i10)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem1.is_allowed(i10)");
	}
	if(!elem1.is_allowed(i11)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem1.is_allowed(i11)");
	}

	if(elem2.is_allowed(i00)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem2.is_allowed(i00)");
	}
	if(!elem2.is_allowed(i01)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem2.is_allowed(i01)");
	}
	if(!elem2.is_allowed(i10)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem2.is_allowed(i10)");
	}
	if(elem2.is_allowed(i11)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem2.is_allowed(i11)");
	}
	
	if(elem3.is_allowed(i00)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem3.is_allowed(i00)");
	}
	if(elem3.is_allowed(i01)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem3.is_allowed(i01)");
	}
	if(elem3.is_allowed(i10)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem3.is_allowed(i10)");
	}
	if(elem3.is_allowed(i11)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem3.is_allowed(i11)");
	}
	if(elem4.is_allowed(i00)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem4.is_allowed(i00)");
	}
	if(elem4.is_allowed(i01)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem4.is_allowed(i01)");
	}
	if(elem4.is_allowed(i10)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem4.is_allowed(i10)");
	}
	if(elem4.is_allowed(i11)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem4.is_allowed(i11)");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Four blocks, all labeled, different index types (2-dim)
 **/
void se_label_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "se_label_test::test_2()";

	try {

	index<2> i1, i2;
	i2[0] = 12; i2[1] = 24;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	mask<2> m01, m10;
	m10[0] = true; m01[1] = true;
	bis.split(m01, 6);
	bis.split(m01, 12);
	bis.split(m01, 18);
	bis.split(m10, 3);
	bis.split(m10, 6);
	bis.split(m10, 9);

	se_label<2, double> elem1(bis.get_block_index_dims(),
			point_group_table::k_id);
	for (size_t i = 0; i < 4; i++) elem1.assign(m10, i, i);
	elem1.assign(m01, 0, 0); // ag
	elem1.assign(m01, 1, 2); // au
	elem1.assign(m01, 2, 1); // eg
	elem1.assign(m01, 3, 3); // eu
	elem1.set_target(0);

	se_label<2, double> elem2(elem1), elem3(elem1), elem4(elem1);
	elem2.set_target(1);
	elem3.set_target(2);
	elem4.set_target(3);

	index<2> i12, i13, i32, i33;
	i12[0] = 1; i12[1] = 2;
	i13[0] = 1; i13[1] = 3;
	i32[0] = 3; i32[1] = 2;
	i33[0] = 3; i33[1] = 3;

	if(! elem1.is_allowed(i12)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem1.is_allowed(i12)");
	}
	if(! elem2.is_allowed(i12)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem2.is_allowed(i12)");
	}
	if(elem3.is_allowed(i12)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem3.is_allowed(i12)");
	}
	if(elem4.is_allowed(i12)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem4.is_allowed(i12)");
	}

	if(elem1.is_allowed(i13)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem1.is_allowed(i13)");
	}
	if(elem2.is_allowed(i13)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem2.is_allowed(i13)");
	}
	if(! elem3.is_allowed(i13)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem3.is_allowed(i13)");
	}
	if(! elem4.is_allowed(i13)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem4.is_allowed(i13)");
	}

	if(elem1.is_allowed(i32)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem1.is_allowed(i32)");
	}
	if(elem2.is_allowed(i32)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem2.is_allowed(i32)");
	}
	if(! elem3.is_allowed(i32)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem3.is_allowed(i32)");
	}
	if(! elem4.is_allowed(i32)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem4.is_allowed(i32)");
	}

	if(! elem1.is_allowed(i33)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem1.is_allowed(i33)");
	}
	if(! elem2.is_allowed(i33)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem2.is_allowed(i33)");
	}
	if(elem3.is_allowed(i33)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem3.is_allowed(i33)");
	}
	if(elem4.is_allowed(i33)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem4.is_allowed(i33)");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Four blocks, all labeled, different index types (4-dim)
 **/
void se_label_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "se_label_test::test_3()";

	try {

	index<4> i1, i2;
	i2[0] = 12; i2[1] = 12, i2[2] = 16, i2[3] = 16;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
	mask<4> m1, m2;
	m1[0] = true; m1[1] = true;
	m2[2] = true; m2[3] = true;
	bis.split(m1, 4);
	bis.split(m1, 8);
	bis.split(m2, 4);
	bis.split(m2, 8);
	bis.split(m2, 12);

	se_label<4, double> elem1(bis.get_block_index_dims(),
			point_group_table::k_id);
	for (size_t i = 0; i < 4; i++) elem1.assign(m2, i, i);
	elem1.assign(m1, 1, 1); // au
	elem1.assign(m1, 2, 2); // eg
	elem1.set_target(0);

	permutation<4> p12;
	p12.permute(1, 2);

	elem1.permute(p12);
	bis.permute(p12);

	if (! elem1.is_valid_bis(bis)) {
		fail_test(testname, __FILE__, __LINE__,
			"Block index space invalid after permutation.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Four blocks, one dim unlabeled, different index types (2-dim)
 **/
void se_label_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "se_label_test::test_4()";

	try {

	index<2> i1, i2;
	i2[0] = 12; i2[1] = 20;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	mask<2> m10, m01;
	m10[0] = true; m01[1] = true;
	bis.split(m10, 3);
	bis.split(m10, 6);
	bis.split(m10, 9);
	bis.split(m01, 10);

	se_label<2, double> elem1(bis.get_block_index_dims(),
			point_group_table::k_id);
	for (size_t i = 0; i < 4; i++) elem1.assign(m10, i, i);
	elem1.set_target(0);

	se_label<2, double> elem2(elem1), elem3(elem1), elem4(elem1);
	elem2.set_target(1);
	elem3.set_target(2);
	elem4.set_target(3);

	index<2> i00, i10, i21, i31;
	i10[0] = 1;
	i21[0] = 2; i21[1] = 1;
	i31[0] = 3; i31[1] = 1;

	if(! elem1.is_allowed(i00)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem1.is_allowed(i00)");
	}
	if(! elem2.is_allowed(i00)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem2.is_allowed(i00)");
	}
	if(! elem3.is_allowed(i00)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem3.is_allowed(i00)");
	}
	if(! elem4.is_allowed(i00)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem4.is_allowed(i00)");
	}

	if(! elem1.is_allowed(i10)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem1.is_allowed(i10)");
	}
	if(! elem2.is_allowed(i10)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem2.is_allowed(i10)");
	}
	if(! elem3.is_allowed(i10)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem3.is_allowed(i10)");
	}
	if(! elem4.is_allowed(i10)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem4.is_allowed(i10)");
	}

	if(! elem1.is_allowed(i21)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem1.is_allowed(i21)");
	}
	if(! elem2.is_allowed(i21)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem2.is_allowed(i21)");
	}
	if(! elem3.is_allowed(i21)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem3.is_allowed(i21)");
	}
	if(! elem4.is_allowed(i21)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem4.is_allowed(i21)");
	}

	if(! elem1.is_allowed(i31)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem1.is_allowed(i31)");
	}
	if(! elem2.is_allowed(i31)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem2.is_allowed(i31)");
	}
	if(! elem3.is_allowed(i31)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem3.is_allowed(i31)");
	}
	if(! elem4.is_allowed(i31)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem4.is_allowed(i31)");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

} // namespace libtensor
