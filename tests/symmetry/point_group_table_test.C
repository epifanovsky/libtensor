#include <libtensor/symmetry/label/point_group_table.h>
#include "point_group_table_test.h"

namespace libtensor {


void point_group_table_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
    test_3();
    test_4();
}


/**	\test Point group C2h.
 **/
void point_group_table_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "point_group_table_test::test_1()";

    try {

        typedef point_group_table::irrep_label_t irrep_label_t;
        typedef point_group_table::irrep_set_t irrep_set_t;
        typedef point_group_table::irrep_group_t irrep_group_t;
        typedef point_group_table::irrep_map_t irrep_map_t;

        irrep_label_t ag = 0, bg = 1, au = 2, bu = 3;
        irrep_map_t irrep_names;
        irrep_names[ag] = "Ag"; irrep_names[bg] = "Bg";
        irrep_names[au] = "Au"; irrep_names[bu] = "Bu";

        point_group_table pg(testname, irrep_names, ag);
        pg.add_product(ag, ag, ag);
        pg.add_product(ag, bg, bg);
        pg.add_product(ag, au, au);
        pg.add_product(ag, bu, bu);
        pg.add_product(bg, ag, bg);
        pg.add_product(bg, bg, ag);
        pg.add_product(bg, au, bu);
        pg.add_product(bg, bu, au);
        pg.add_product(au, ag, au);
        pg.add_product(au, bg, bu);
        pg.add_product(au, au, ag);
        pg.add_product(au, bu, bg);
        pg.add_product(bu, ag, bu);
        pg.add_product(bu, bg, au);
        pg.add_product(bu, au, bg);
        pg.add_product(bu, bu, ag);
        pg.check();

        std::string id(pg.get_id());
        if (id.compare(testname) != 0)
            fail_test(testname, __FILE__, __LINE__, "Wrong id.");

        irrep_set_t irs = pg.get_complete_set();
        if (irs.size() != irrep_names.size())
            fail_test(testname, __FILE__, __LINE__, "Wrong number of labels.");

        irrep_group_t lg;
        lg.insert(ag); lg.insert(ag);
        if (! pg.is_in_product(lg, ag))
            fail_test(testname, __FILE__, __LINE__,
                    "Wrong result of product.");

        lg.clear();
        lg.insert(au); lg.insert(bg);
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

        typedef point_group_table::irrep_label_t irrep_label_t;
        typedef point_group_table::irrep_set_t irrep_set_t;
        typedef point_group_table::irrep_group_t irrep_group_t;
        typedef point_group_table::irrep_map_t irrep_map_t;

        irrep_label_t g = 0, u = 1;
        irrep_map_t irrep_names;
        irrep_names[g] = "g"; irrep_names[u] = "u";
        point_group_table pg(testname, irrep_names, g);

        bool failed = false;
        try {

            pg.add_product(g, 2, u);

        } catch(exception &e) {
            failed = true;
        }
        if (! failed) {
            fail_test(testname, __FILE__, __LINE__,
                    "Illegal call of add_product not catched.");
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

        typedef point_group_table::irrep_label_t irrep_label_t;
        typedef point_group_table::irrep_set_t irrep_set_t;
        typedef point_group_table::irrep_group_t irrep_group_t;
        typedef point_group_table::irrep_map_t irrep_map_t;

        irrep_label_t ag = 0, bg = 1, au = 2, bu = 3;
        irrep_map_t irrep_names;
        irrep_names[ag] = "Ag"; irrep_names[bg] = "Bg";
        irrep_names[au] = "Au"; irrep_names[bu] = "Bu";

        point_group_table pg(testname, irrep_names, ag);
        pg.add_product(bg, au, bu);
        pg.add_product(bg, bu, au);
        pg.add_product(au, bu, bg);
        pg.check();

        point_group_table *pg_copy = pg.clone();
        pg_copy->check();
        delete pg_copy;

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/**	\test Product evaluation for product table
 **/
void point_group_table_test::test_4() throw(libtest::test_exception) {

    static const char *testname = "point_group_table_test::test_4()";

    try {

        typedef point_group_table::irrep_label_t irrep_label_t;
        typedef point_group_table::irrep_set_t irrep_set_t;
        typedef point_group_table::irrep_group_t irrep_group_t;
        typedef point_group_table::irrep_map_t irrep_map_t;

        irrep_label_t ag = 0, eg = 1, au = 2, eu = 3;
        irrep_map_t irrep_names;
        irrep_names[ag] = "Ag"; irrep_names[eg] = "Eg";
        irrep_names[au] = "Au"; irrep_names[eu] = "Eu";
        point_group_table pg(testname, irrep_names, ag);
        pg.add_product(eg, eg, eg);
        pg.add_product(eg, au, eu);
        pg.add_product(eg, eu, au);
        pg.add_product(eg, eu, eu);
        pg.add_product(au, eu, eg);
        pg.add_product(eu, eu, eg);
        pg.check();

        // test ag x eu = eu and eu x eg = au + eu
        irrep_group_t lg1, lg2;
        lg1.insert(ag); lg1.insert(eu);
        lg2.insert(eg); lg2.insert(eu);

        if (! pg.is_in_product(lg1, eu))
            fail_test(testname, __FILE__, __LINE__,
                    "Eu is not in product Ag x Eu.");
        if (! pg.is_in_product(lg2, eu))
            fail_test(testname, __FILE__, __LINE__,
                    "Eu is not in product Eu x Eg.");
        if (! pg.is_in_product(lg2, au))
            fail_test(testname, __FILE__, __LINE__,
                    "Au is not in product Eu x Eg.");

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

} // namespace libtensor
