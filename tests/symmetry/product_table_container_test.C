#include <libtensor/symmetry/label/point_group_table.h>
#include <libtensor/symmetry/label/product_table_container.h>
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


	point_group_table pg1(testname, 4), pg2(testname, 2);
	label_t ag = 0, bg = 1, au = 2, bu = 3, g = 0, u = 1;
	pg1.add_product(ag, ag, ag);
	pg1.add_product(ag, bg, bg);
	pg1.add_product(ag, au, au);
	pg1.add_product(ag, bu, bu);
	pg1.add_product(bg, ag, bg);
	pg1.add_product(bg, bg, ag);
	pg1.add_product(bg, au, bu);
	pg1.add_product(bg, bu, au);
	pg1.add_product(au, ag, au);
	pg1.add_product(au, bg, bu);
	pg1.add_product(au, au, ag);
	pg1.add_product(au, bu, bg);
	pg1.add_product(bu, ag, bu);
	pg1.add_product(bu, bg, au);
	pg1.add_product(bu, au, bg);
	pg1.add_product(bu, bu, ag);
	pg1.check();

	pg2.add_product(g, g, g);
	pg2.add_product(g, u, u);
	pg2.add_product(u, g, u);
	pg2.add_product(u, u, g);
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

	ptc.erase(testname);

	// adding incomplete table
	pg2.delete_product(u, g);

	failed = false;
	try {

	ptc.add(pg2);

	} catch(exception &e) {
		failed = true;
	}

	if (! failed) {
		fail_test(testname, __FILE__, __LINE__,
				"Adding incomplete product table.");
	}


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
	point_group_table pg(testname, 2);
	label_t g = 0, u = 1;
	pg.add_product(g, g, g);
	pg.add_product(g, u, u);
	pg.add_product(u, g, u);
	pg.add_product(u, u, g);
	pg.check();

	ptc.add(pg);
	}

	product_table_i &pt1 = ptc.req_table(testname);

	bool failed = false;
	try {

	product_table_i &pt2 = ptc.req_table(testname);

	} catch(exception &e) {
		failed = true;
	}
	if (! failed) {
		fail_test(testname, __FILE__, __LINE__,
				"Table requested twice for writing.");
	}

	failed = false;
	try {

	const product_table_i &pt2 = ptc.req_const_table(testname);

	} catch(exception &e) {
		failed = true;
	}
	if (! failed) {
		fail_test(testname, __FILE__, __LINE__,
				"Table requested for reading, while already checked out for writing.");
	}

	ptc.ret_table(testname);

	failed = false;
	try {

	ptc.ret_table(testname);

	} catch(exception &e) {
		failed = true;
	}
	if (! failed) {
		fail_test(testname, __FILE__, __LINE__, "Table returned twice.");
	}

	const product_table_i &pt_r1 = ptc.req_const_table(testname);
	const product_table_i &pt_r2 = ptc.req_const_table(testname);
	ptc.ret_table(testname);
	ptc.ret_table(testname);

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

	ptc.erase(testname);

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
	point_group_table pg(testname, 2);
	label_t g = 0, u = 1;
	pg.add_product(g, g, g);
	pg.add_product(g, u, u);
	pg.add_product(u, g, u);
	pg.add_product(u, u, g);
	pg.check();

	ptc.add(pg);
	}

	product_table_i &pt1 = ptc.req_table(testname);

	bool failed = false;
	try {

	ptc.erase(testname);

	} catch(exception &e) {
		failed = true;
	}
	if (! failed) {
		fail_test(testname, __FILE__, __LINE__, "Checked out table deleted.");
	}

	ptc.ret_table(testname);

	const product_table_i &pt2 = ptc.req_const_table(testname);

	failed = false;
	try {

	ptc.erase(testname);

	} catch(exception &e) {
		failed = true;
	}
	if (! failed) {
		fail_test(testname, __FILE__, __LINE__, "Checked out table deleted.");
	}

	ptc.ret_table(testname);

	ptc.erase(testname);

	failed = false;
	try {

	ptc.erase(testname);

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
