#include <libtensor/btod/transf_double.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/se_label.h>
#include "se_label_test.h"

namespace libtensor {

const char *se_label_test::table_id = "point_group";

void se_label_test::perform() throw(libtest::test_exception) {

	// Setup point_group_table and product_table_container

	try {

	point_group_table s6(table_id, 4);
	point_group_table::label_t ag = 0, eg = 1, au = 2, eu = 3;
	s6.add_product(ag, ag, ag);
	s6.add_product(ag, eg, eg);
	s6.add_product(ag, au, au);
	s6.add_product(ag, eu, eu);
	s6.add_product(eg, eg, ag);
	s6.add_product(eg, eg, eg);
	s6.add_product(eg, au, eu);
	s6.add_product(eg, eu, au);
	s6.add_product(eg, eu, eu);
	s6.add_product(au, au, ag);
	s6.add_product(au, eu, eg);
	s6.add_product(eu, eu, ag);
	s6.add_product(eu, eu, eg);
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
	test_5();
	test_6();
	test_7();
	test_8();

	product_table_container::get_instance().erase(table_id);

	} catch (libtest::test_exception) {
		product_table_container::get_instance().erase(table_id);
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

	se_label<2, double> elem1(bis.get_block_index_dims(), table_id);
	elem1.assign(m11, 0, 0); // ag
	elem1.assign(m11, 1, 2); // au
	elem1.add_target(0);

	se_label<2, double> elem2(elem1), elem3(elem1), elem4(elem1);
	elem2.delete_target();
	elem2.add_target(2);
	elem3.delete_target();
	elem3.add_target(1);
	elem4.delete_target();
	elem4.add_target(3);

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

	se_label<2, double> elem1(bis.get_block_index_dims(), table_id);
	for (size_t i = 0; i < 4; i++) elem1.assign(m10, i, i);
	elem1.assign(m01, 0, 0); // ag
	elem1.assign(m01, 1, 2); // au
	elem1.assign(m01, 2, 1); // eg
	elem1.assign(m01, 3, 3); // eu
	elem1.add_target(0);

	se_label<2, double> elem2(elem1), elem3(elem1), elem4(elem1);
	elem2.add_target(1);
	elem3.delete_target();
	elem3.add_target(2);
	elem4.delete_target();
	elem4.add_target(3);

	index<2> i02, i12, i13, i21, i32, i33;
	i02[0] = 0; i02[1] = 2;
	i12[0] = 1; i12[1] = 2;
	i13[0] = 1; i13[1] = 3;
	i21[0] = 2; i21[1] = 1;
	i32[0] = 3; i32[1] = 2;
	i33[0] = 3; i33[1] = 3;

	if(elem1.is_allowed(i02)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem1.is_allowed(i02)");
	}
	if(! elem1.is_allowed(i12)) {
		fail_test(testname, __FILE__, __LINE__,
			"! elem1.is_allowed(i12)");
	}
	if(elem1.is_allowed(i13)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem1.is_allowed(i13)");
	}
	if(! elem1.is_allowed(i21)) {
		fail_test(testname, __FILE__, __LINE__,
			"! elem1.is_allowed(i21)");
	}
	if(elem1.is_allowed(i32)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem1.is_allowed(i32)");
	}
	if(! elem1.is_allowed(i33)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem1.is_allowed(i33)");
	}

	if(! elem2.is_allowed(i02)) {
		fail_test(testname, __FILE__, __LINE__,
			"! elem2.is_allowed(i02)");
	}
	if(! elem2.is_allowed(i12)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem2.is_allowed(i12)");
	}
	if(elem2.is_allowed(i13)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem2.is_allowed(i13)");
	}
	if(! elem2.is_allowed(i21)) {
		fail_test(testname, __FILE__, __LINE__,
			"! elem1.is_allowed(i21)");
	}
	if(elem2.is_allowed(i32)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem2.is_allowed(i32)");
	}
	if(! elem2.is_allowed(i33)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem2.is_allowed(i33)");
	}

	if(elem3.is_allowed(i12)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem3.is_allowed(i12)");
	}
	if(! elem3.is_allowed(i13)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem3.is_allowed(i13)");
	}
	if(! elem3.is_allowed(i32)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem3.is_allowed(i32)");
	}
	if(elem3.is_allowed(i33)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem3.is_allowed(i33)");
	}

	if(elem4.is_allowed(i12)) {
		fail_test(testname, __FILE__, __LINE__,
			"elem4.is_allowed(i12)");
	}
	if(! elem4.is_allowed(i13)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem4.is_allowed(i13)");
	}
	if(! elem4.is_allowed(i32)) {
		fail_test(testname, __FILE__, __LINE__,
			"!elem4.is_allowed(i32)");
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

	se_label<4, double> elem1(bis.get_block_index_dims(), table_id);
	for (size_t i = 0; i < 4; i++) elem1.assign(m2, i, i);
	elem1.assign(m1, 1, 1); // au
	elem1.assign(m1, 2, 2); // eg
	elem1.add_target(0);

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

	se_label<2, double> elem1(bis.get_block_index_dims(), table_id);
	for (size_t i = 0; i < 4; i++) elem1.assign(m10, i, i);
	elem1.add_target(0);

	se_label<2, double> elem2(elem1), elem3(elem1), elem4(elem1);
	elem2.add_target(1);
	elem3.add_target(2);
	elem4.add_target(3);

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

/**	\test Adding removing labels, matching splits
 **/
void se_label_test::test_5() throw(libtest::test_exception) {

	static const char *testname = "se_label_test::test_5()";

	try {

	index<4> i1, i2;
	i2[0] = 12; i2[1] = 12; i2[2] = 20; i2[3] = 20;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
	mask<4> m0001, m0011, m1100, m1111;
	m0001[3] = true;
	m0011[2] = true; m0011[3] = true;
	m1100[0] = true; m1100[1] = true;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	bis.split(m1100, 3);
	bis.split(m1100, 6);
	bis.split(m1100, 9);
	bis.split(m0011, 5);
	bis.split(m0011, 10);
	bis.split(m0011, 15);

	se_label<4, double> elem1(bis.get_block_index_dims(), table_id);

	// assign labels
	for (size_t i = 0; i < 4; i++) elem1.assign(m1111, i, i);

	// assign different labels (two label types)

	elem1.assign(m1100, 0, 1);
	elem1.assign(m1100, 1, 0);

	if (elem1.get_dim_type(0) == elem1.get_dim_type(2))
		fail_test(testname, __FILE__, __LINE__,
			"elem1.get_dim_type(0)==elem1.get_dim_type(2)");


	// remove previously assigned labels

	elem1.remove(m0001, 0);

	if (elem1.get_dim_type(0) == elem1.get_dim_type(2))
		fail_test(testname, __FILE__, __LINE__,
			"elem1.get_dim_type(0)==elem1.get_dim_type(2)");
	if (elem1.get_dim_type(2) == elem1.get_dim_type(3))
		fail_test(testname, __FILE__, __LINE__,
			"elem1.get_dim_type(2)==elem1.get_dim_type(3)");
	if (elem1.get_dim_type(0) == elem1.get_dim_type(3))
		fail_test(testname, __FILE__, __LINE__,
			"elem1.get_dim_type(0)==elem1.get_dim_type(3)");

	// reassign removed label

	elem1.assign(m0001, 0, 0);

	// match labels

	elem1.match_labels();

	if (elem1.get_dim_type(2) != elem1.get_dim_type(3))
		fail_test(testname, __FILE__, __LINE__,
			"elem1.get_dim_type(2)!=elem1.get_dim_type(3)");

	// assign labels so that all dimensions re equal again.

	elem1.assign(m1100, 0, 0);
	elem1.assign(m1100, 1, 1);

	elem1.match_labels();

	if (elem1.get_dim_type(0) != elem1.get_dim_type(2))
		fail_test(testname, __FILE__, __LINE__,
			"elem1.get_dim_type(0)!=elem1.get_dim_type(2)");

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}
/**	\test Exception tests
 **/
void se_label_test::test_6() throw(libtest::test_exception) {

	static const char *testname = "se_label_test::test_6()";

	index<4> i1, i2;
	i2[0] = 12; i2[1] = 12; i2[2] = 20; i2[3] = 20;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
	mask<4> m0001, m0011, m1100, m1111;
	m0001[3] = true;
	m0011[2] = true; m0011[3] = true;
	m1100[0] = true; m1100[1] = true;
	m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
	bis.split(m1100, 3);
	bis.split(m1100, 6);
	bis.split(m1100, 9);
	bis.split(m0011, 5);
	bis.split(m0011, 10);
	bis.split(m0011, 15);

	se_label<4, double> elem1(bis.get_block_index_dims(), table_id);

	// assign labels
	for (size_t i = 0; i < 4; i++) elem1.assign(m1111, i, i);

	// assign different labels (two label types)

	elem1.assign(m1100, 0, 1);
	elem1.assign(m1100, 1, 0);

	// test invalid mask exception for remove()
	bool failed = false;
	try {

	elem1.remove(m1111, 2);

	} catch (bad_parameter &e) {
		failed = true;
	}

	if (! failed)
		fail_test(testname, __FILE__, __LINE__,
			"Invalid mask in remove(mask<N>, size_t) not recognized.");

	failed = false;
	try {

	elem1.assign(m1111, 0, 0);

	} catch (bad_parameter &e) {
		failed = true;
	}

	if (! failed)
		fail_test(testname, __FILE__, __LINE__,
			"Invalid mask in assign(mask<N>, size_t, label_t) not recognized.");

}

/**	\test Four blocks, all labeled, no target / full target (2-dim)
 **/
void se_label_test::test_7() throw(libtest::test_exception) {

	static const char *testname = "se_label_test::test_7()";

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	mask<2> m11;
	m11[0] = true; m11[1] = true;
	bis.split(m11, 2);
	bis.split(m11, 5);
	bis.split(m11, 7);

	se_label<2, double> elem1(bis.get_block_index_dims(), table_id);
	for (size_t i = 0; i < 4; i++) elem1.assign(m11, i, i);

	se_label<2, double> elem2(elem1);
	for (size_t i = 0; i < 4; i++) elem2.add_target(i);

	index<2> idx;
	for (size_t i = 0; i < 4; i++)
	for (size_t j = 0; j < 4; j++) {

		idx[0] = i; idx[1] = j;

		if(elem1.is_allowed(idx)) {
			std::ostringstream oss;
			oss << "elem1.is_allowed(i" << i << j << ")";
			fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
		}
		if(! elem2.is_allowed(idx)) {
			std::ostringstream oss;
			oss << "! elem2.is_allowed(i" << i << j << ")";
			fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
		}
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\test Four blocks, all labeled, single target, varying dimensions, permute.
 **/
void se_label_test::test_8() throw(libtest::test_exception) {

	static const char *testname = "se_label_test::test_8()";

	try {

	index<3> i1, i2;
	i2[0] = 3; i2[1] = 5; i2[2] = 2;
	dimensions<3> dim(index_range<3>(i1, i2));
	mask<3> m100, m010, m001;
	m100[0] = true; m010[1] = true; m001[2] = true;

	se_label<3, double> elem1(dim, table_id);
	size_t mapa[6], mapb[3];
	mapa[0] = 1; mapa[1] = 0; mapa[2] = 0;
	mapa[3] = 1; mapa[4] = 2; mapa[5] = 3;
	mapb[0] = 2; mapb[1] = 2; mapb[2] = 3;

	for (size_t i = 0; i < 4; i++) elem1.assign(m100, i, i);
	for (size_t i = 0; i < 6; i++) elem1.assign(m010, i, mapa[i]);
	for (size_t i = 0; i < 3; i++) elem1.assign(m001, i, mapb[i]);

	permutation<3> perm; perm.permute(0, 1).permute(1, 2);

	dim.permute(perm);
	elem1.permute(perm);


	if(! dim.equals(elem1.get_block_index_dims())) {
		fail_test(testname, __FILE__, __LINE__, "Wrong dim.");
	}
	if(elem1.get_label(elem1.get_dim_type(0), 0) != 1) {
		fail_test(testname, __FILE__, __LINE__, "Wrong label.");
	}
	if(elem1.get_label(elem1.get_dim_type(1), 0) != 2) {
		fail_test(testname, __FILE__, __LINE__, "Wrong label.");
	}
	if(elem1.get_label(elem1.get_dim_type(2), 0) != 0) {
		fail_test(testname, __FILE__, __LINE__, "Wrong label.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

} // namespace libtensor
