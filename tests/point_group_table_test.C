#include <libtensor/symmetry/point_group_table.h>
#include "point_group_table_test.h"

namespace libtensor {


void point_group_table_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
}


/**	\test Point group C2h.
 **/
void point_group_table_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "point_group_table_test::test_1()";

	try {

	typedef point_group_table::label_t label_t;
	typedef point_group_table::label_group label_group;

	point_group_table pg(testname, 4);
	label_t ag = 0, bg = 1, au = 2, bu = 3;
	pg.set_product(ag, ag, 0, ag);
	pg.set_product(ag, bg, 0, bg);
	pg.set_product(ag, au, 0, au);
	pg.set_product(ag, bu, 0, bu);
	pg.set_product(bg, bg, 0, ag);
	pg.set_product(bg, au, 0, bu);
	pg.set_product(bg, bu, 0, au);
	pg.set_product(au, au, 0, ag);
	pg.set_product(au, bu, 0, bg);
	pg.set_product(bu, bu, 0, ag);
	pg.check();

	std::string id(pg.get_id());
	if (id.compare(testname) != 0)
		fail_test(testname, __FILE__, __LINE__, "Wrong id.");

	if (pg.nlabels() != 4)
		fail_test(testname, __FILE__, __LINE__, "Wrong number of labels.");

	label_t any = pg.invalid();
	if (pg.is_valid(any))
		fail_test(testname, __FILE__, __LINE__,
				"Invalid label not recognized.");

	label_group lg(2, any);
	lg[0] = ag; lg[1] = ag;
	if (! pg.is_in_product(lg, ag))
		fail_test(testname, __FILE__, __LINE__,
				"Wrong result of product.");

	lg[0] = au; lg[1] = bg;
	if (! pg.is_in_product(lg, bu))
		fail_test(testname, __FILE__, __LINE__,
				"Wrong result of product.");

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Wrongly setup product table
 **/
void point_group_table_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "point_group_table_test::test_2()";

	try {

	typedef point_group_table::label_t label_t;
	typedef point_group_table::label_group label_group;

	point_group_table pg(testname, 2);
	label_t g = 0, u = 1;

	bool failed = false;
	try {
	pg.set_product(g, 2, 0, u);
	} catch(exception &e) {
		failed = true;
	}
	if (! failed) {
		fail_test(testname, __FILE__, __LINE__,
				"Illegal call of set_product not catched.");
	}

	pg.set_product(g, g, 0, g);
	pg.set_product(g, u, 0, u);
	pg.set_product(u, u, 1, u);

	failed = false;
	try {
	pg.check();
	} catch(exception &e) {
		failed = true;
	}
	if (! failed) {
		fail_test(testname, __FILE__, __LINE__,
				"Consistency check failed.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Clone method
 **/
void point_group_table_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "point_group_table_test::test_3()";

	try {

	typedef point_group_table::label_t label_t;
	typedef point_group_table::label_group label_group;

	point_group_table pg(testname, 4);
	label_t ag = 0, bg = 1, au = 2, bu = 3;
	pg.set_product(ag, ag, 0, ag);
	pg.set_product(ag, bg, 0, bg);
	pg.set_product(ag, au, 0, au);
	pg.set_product(ag, bu, 0, bu);
	pg.set_product(bg, bg, 0, ag);
	pg.set_product(bg, au, 0, bu);
	pg.set_product(bg, bu, 0, au);
	pg.set_product(au, au, 0, ag);
	pg.set_product(au, bu, 0, bg);
	pg.set_product(bu, bu, 0, ag);
	pg.check();

	point_group_table *pg_copy = pg.clone();
	pg_copy->check();
	delete pg_copy;

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
