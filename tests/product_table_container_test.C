#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include "product_table_container_test.h"

namespace libtensor {


void product_table_container_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
}


/**	\test Add tables to prodcut_table_container
 **/
void product_table_container_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "product_table_container_test::test_1()";

	try {

	typedef point_group_table::label_t label_t;
	typedef point_group_table::label_group label_group;

	product_table_container &ptc = product_table_container::get_instance();


	point_group_table pg1(4), pg2(2);
	label_t ag = 0, bg = 1, au = 2, bu = 3, g = 0, u = 1;
	pg1.set_product(ag, ag, 0, ag);
	pg1.set_product(ag, bg, 0, bg);
	pg1.set_product(ag, au, 0, au);
	pg1.set_product(ag, bu, 0, bu);
	pg1.set_product(bg, bg, 0, ag);
	pg1.set_product(bg, au, 0, bu);
	pg1.set_product(bg, bu, 0, au);
	pg1.set_product(au, au, 0, ag);
	pg1.set_product(au, bu, 0, bg);
	pg1.set_product(bu, bu, 0, ag);
	pg1.check();

	pg2.set_product(g, g, 0, g);
	pg2.set_product(g, u, 0, u);
	pg2.set_product(u, u, 0, g);
	pg2.check();

	ptc.add(pg1);
	bool failed = false;
	try {

	ptc.add(pg2);

	} catch(exception &e) {
		failed = true;
	}

	if (! failed) {
		fail_test(testname, __FILE__, __LINE__,
				"Adding twice the same type of product table.");
	}

	ptc.erase(point_group_table::k_id);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Requesting and returning tables
 **/
void product_table_container_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "product_table_container_test::test_2()";

	try {

	typedef point_group_table::label_t label_t;
	typedef point_group_table::label_group label_group;

	product_table_container &ptc = product_table_container::get_instance();

	{ // Setup point group table
	point_group_table pg(2);
	label_t g = 0, u = 1;
	pg.set_product(g, g, 0, g);
	pg.set_product(g, u, 0, u);
	pg.set_product(u, u, 0, g);
	pg.check();

	ptc.add(pg);
	}

	product_table_i &pt1 = ptc.req_table(point_group_table::k_id);

	bool failed = false;
	try {

	product_table_i &pt2 = ptc.req_table(point_group_table::k_id);

	} catch(exception &e) {
		failed = true;
	}
	if (! failed) {
		fail_test(testname, __FILE__, __LINE__,
				"Table requested twice for writing.");
	}

	failed = false;
	try {

	const product_table_i &pt2 = ptc.req_const_table(point_group_table::k_id);

	} catch(exception &e) {
		failed = true;
	}
	if (! failed) {
		fail_test(testname, __FILE__, __LINE__,
				"Table requested for reading, while already checked out for writing.");
	}

	ptc.ret_table(point_group_table::k_id);

	failed = false;
	try {

	ptc.ret_table(point_group_table::k_id);

	} catch(exception &e) {
		failed = true;
	}
	if (! failed) {
		fail_test(testname, __FILE__, __LINE__, "Table returned twice.");
	}

	const product_table_i &pt_r1 = ptc.req_const_table(point_group_table::k_id);
	const product_table_i &pt_r2 = ptc.req_const_table(point_group_table::k_id);
	ptc.ret_table(point_group_table::k_id);
	ptc.ret_table(point_group_table::k_id);

	failed = false;
	try {

	const product_table_i &pt2 = ptc.req_table("What so ever.");

	} catch(exception &e) {
		failed = true;
	}
	if (! failed) {
		fail_test(testname, __FILE__, __LINE__, "Unknown table requested.");
	}

	failed = false;
	try {

	ptc.ret_table("What so ever.");

	} catch(exception &e) {
		failed = true;
	}
	if (! failed) {
		fail_test(testname, __FILE__, __LINE__, "Unknown table returned.");
	}

	ptc.erase(point_group_table::k_id);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Deleting tables
 **/
void product_table_container_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "product_table_container_test::test_3()";

	try {

	typedef point_group_table::label_t label_t;
	typedef point_group_table::label_group label_group;

	product_table_container &ptc = product_table_container::get_instance();

	{ // Setup point group table
	point_group_table pg(2);
	label_t g = 0, u = 1;
	pg.set_product(g, g, 0, g);
	pg.set_product(g, u, 0, u);
	pg.set_product(u, u, 0, g);
	pg.check();

	ptc.add(pg);
	}

	product_table_i &pt1 = ptc.req_table(point_group_table::k_id);

	bool failed = false;
	try {

	ptc.erase(point_group_table::k_id);

	} catch(exception &e) {
		failed = true;
	}
	if (! failed) {
		fail_test(testname, __FILE__, __LINE__, "Checked out table deleted.");
	}

	ptc.ret_table(point_group_table::k_id);

	const product_table_i &pt2 = ptc.req_const_table(point_group_table::k_id);

	failed = false;
	try {

	ptc.erase(point_group_table::k_id);

	} catch(exception &e) {
		failed = true;
	}
	if (! failed) {
		fail_test(testname, __FILE__, __LINE__, "Checked out table deleted.");
	}

	ptc.ret_table(point_group_table::k_id);

	ptc.erase(point_group_table::k_id);

	failed = false;
	try {

	ptc.erase(point_group_table::k_id);

	} catch(exception &e) {
		failed = true;
	}
	if (! failed) {
		fail_test(testname, __FILE__, __LINE__, "Table deleted twice.");
	}

	failed = false;
	try {

	ptc.erase("What so ever");

	} catch(exception &e) {
		failed = true;
	}
	if (! failed) {
		fail_test(testname, __FILE__, __LINE__, "Unknown table deleted.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}



} // namespace libtensor
