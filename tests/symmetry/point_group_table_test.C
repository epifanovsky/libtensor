#include <libtensor/symmetry/point_group_table.h>
#include "point_group_table_test.h"

namespace libtensor {


void point_group_table_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
    test_3();
    test_4();
}


/** \test Point group C2h.
 **/
void point_group_table_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "point_group_table_test::test_1()";

    try {

        std::vector<std::string> irreps(4);
        point_group_table::label_t ag = 0, bg = 1, au = 2, bu = 3;
        irreps[ag] = "Ag"; irreps[bg] = "Bg";
        irreps[au] = "Au"; irreps[bu] = "Bu";

        point_group_table pg(testname, irreps, "Ag");
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

        if (pg.get_n_labels() != 4)
            fail_test(testname, __FILE__, __LINE__, "Wrong number of labels.");

        if (pg.get_irrep_name(point_group_table::k_identity) != "Ag")
            fail_test(testname, __FILE__, __LINE__, "Name of identity irrep.");
        if (pg.get_label("Ag") != point_group_table::k_identity)
            fail_test(testname, __FILE__, __LINE__, "Label of identity irrep.");

        for (size_t i = 1; i < 4; i++) {
            point_group_table::label_t il = pg.get_label(irreps[i]);
            if (pg.get_irrep_name(il) != irreps[i])
                fail_test(testname, __FILE__, __LINE__,
                        "Label-name assignment.");
        }

        product_table_i::label_group_t lg;
        lg.push_back(ag); lg.push_back(ag);
        if (! pg.is_in_product(lg, ag))
            fail_test(testname, __FILE__, __LINE__,
                    "Wrong result of product.");

        lg.clear();
        lg.push_back(au); lg.push_back(bg);
        if (! pg.is_in_product(lg, bu))
            fail_test(testname, __FILE__, __LINE__,
                    "Wrong result of product.");

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Wrongly setup product table
 **/
void point_group_table_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "point_group_table_test::test_2()";

    try {

        point_group_table::label_t g = 0, u = 1;
        std::vector<std::string> irreps(2);
        irreps[g] = "g"; irreps[u] = "u";
        point_group_table pg(testname, irreps, "g");
        pg.add_product(u, u, g);

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


/** \test Clone method
 **/
void point_group_table_test::test_3() throw(libtest::test_exception) {

    static const char *testname = "point_group_table_test::test_3()";

    try {

        point_group_table::label_t ag = 0, bg = 1, au = 2, bu = 3;
        std::vector<std::string> irreps(4);
        irreps[ag] = "Ag"; irreps[bg] = "Bg";
        irreps[au] = "Au"; irreps[bu] = "Bu";

        point_group_table pg(testname, irreps, "Ag");
        pg.add_product(bg, bg, ag);
        pg.add_product(bg, au, bu);
        pg.add_product(bg, bu, au);
        pg.add_product(au, au, ag);
        pg.add_product(au, bu, bg);
        pg.add_product(bu, bu, ag);
        pg.check();

        point_group_table *pg_copy = pg.clone();
        pg_copy->check();
        delete pg_copy;

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Product evaluation for product table
 **/
void point_group_table_test::test_4() throw(libtest::test_exception) {

    static const char *testname = "point_group_table_test::test_4()";

    try {

        point_group_table::label_t ag = 0, eg = 1, au = 2, eu = 3;
        std::vector<std::string> irreps(4);
        irreps[ag] = "Ag"; irreps[eg] = "Eg";
        irreps[au] = "Au"; irreps[eu] = "Eu";
        point_group_table pg(testname, irreps, "Ag");
        pg.add_product(eg, eg, ag);
        pg.add_product(eg, eg, eg);
        pg.add_product(eg, au, eu);
        pg.add_product(eg, eu, au);
        pg.add_product(eg, eu, eu);
        pg.add_product(au, au, ag);
        pg.add_product(au, eu, eg);
        pg.add_product(eu, eu, ag);
        pg.add_product(eu, eu, eg);
        pg.check();

        // test ag x eu = eu and eu x eg = au + eu
        product_table_i::label_set_t::const_iterator ils;

        product_table_i::label_set_t ls1 = pg.product(ag, eu);
        if (ls1.size() != 1)
            fail_test(testname, __FILE__, __LINE__,
                    "Wrong number of labels in product");
        if (*(ls1.begin()) != eu)
            fail_test(testname, __FILE__, __LINE__,
                    "Eu is not in product Ag x Eu.");

        product_table_i::label_set_t ls2 = pg.product(eu, eg);
        if (ls2.size() != 2)
            fail_test(testname, __FILE__, __LINE__,
                    "Wrong number of labels in product");

        ils = ls2.begin();
        if (*ils != au)
            fail_test(testname, __FILE__, __LINE__,
                    "Au is not in product Eg x Eu.");
        ils++;
        if (*ils != eu)
            fail_test(testname, __FILE__, __LINE__,
                    "Eu is not in product Eg x Eu.");

        product_table_i::label_set_t ls3a, ls3;
        ls3a.insert(au); ls3a.insert(eu);
        ls3 = pg.product(eg, ls3a);
        if (ls3.size() != 2)
            fail_test(testname, __FILE__, __LINE__,
                    "Wrong number of labels in product");

        ils = ls3.begin();
        if (*ils != au)
            fail_test(testname, __FILE__, __LINE__,
                    "Au is not in product Eg x (Au + Eu).");
        ils++;
        if (*ils != eu)
            fail_test(testname, __FILE__, __LINE__,
                    "Eu is not in product Eg x (Au + Eu).");

        product_table_i::label_set_t ls4a, ls4;
        ls4a.insert(ag); ls4a.insert(au);
        ls4 = pg.product(ls4a, au);
        if (ls4.size() != 2)
            fail_test(testname, __FILE__, __LINE__,
                    "Wrong number of labels in product");

        ils = ls4.begin();
        if (*ils != ag)
            fail_test(testname, __FILE__, __LINE__,
                    "Ag is not in product (Ag + Au) x Au.");
        ils++;
        if (*ils != au)
            fail_test(testname, __FILE__, __LINE__,
                    "Au is not in product (Ag + Au) x Au.");

        product_table_i::label_set_t ls5a, ls5b, ls5;
        ls5a.insert(ag); ls5a.insert(au);
        ls5b.insert(eg); ls5b.insert(au);
        ls5 = pg.product(ls5a, ls5b);
        if (ls5.size() != 4)
            fail_test(testname, __FILE__, __LINE__,
                    "Wrong number of labels in product");

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
