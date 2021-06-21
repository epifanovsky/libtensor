#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include "product_table_container_test.h"

namespace libtensor {


void product_table_container_test::perform() {

    test_1();
    test_2();
    test_3();
}


/** \test Add tables to prodcut_table_container
 **/
void product_table_container_test::test_1() {

    static const char *testname = "product_table_container_test::test_1()";

    try {

        product_table_container &ptc = product_table_container::get_instance();

        point_group_table::label_t ag = 0, bg = 1, au = 2, bu = 3, g = 0, u = 1;
        std::vector<std::string> in1(4), in2(2);
        in1[ag] = "Ag"; in1[bg] = "Bg"; in1[au] = "Au"; in1[bu] = "Bu";
        in2[g] = "g"; in2[u] = "u";
        point_group_table pg1(testname, in1, "Ag"), pg2(testname, in2, "g");
        pg1.add_product(bg, bg, ag);
        pg1.add_product(bg, au, bu);
        pg1.add_product(bg, bu, au);
        pg1.add_product(au, au, ag);
        pg1.add_product(au, bu, bg);
        pg1.add_product(bu, bu, ag);
        pg1.check();

        pg2.add_product(u, u, g);
        pg2.check();

        ptc.add(pg2);
        bool failed = false;
        try {

            ptc.add(pg1);

        } catch(exception &e) {
            failed = true;
        }

        if (! failed) {
            fail_test(testname, __FILE__, __LINE__,
                    "Adding twice the same type of product table.");
        }

        ptc.erase(testname);

        // adding incomplete table
        pg1.reset();
        pg1.add_product(bg, au, bu);
        pg1.add_product(bg, bu, au);

        failed = false;
        try {

            ptc.add(pg1);

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


/** \test Requesting and returning tables
 **/
void product_table_container_test::test_2() {

    static const char *testname = "product_table_container_test::test_2()";

    try {

        product_table_container &ptc = product_table_container::get_instance();

        { // Setup point group table
            point_group_table::label_t g = 0, u = 1;
            std::vector<std::string> irreps(2);
            irreps[g] = "g"; irreps[u] = "u";
            point_group_table pg(testname, irreps, "g");
            pg.add_product(u, u, g);
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
                    "Table requested for reading, "
                    "while already checked out for writing.");
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
        const point_group_table &pt_r2 =
                ptc.req_const_table<point_group_table>(testname);
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


/** \test Deleting tables
 **/
void product_table_container_test::test_3() {

    static const char *testname = "product_table_container_test::test_3()";

    try {

        product_table_container &ptc = product_table_container::get_instance();

        { // Setup point group table
            point_group_table::label_t g = 0, u = 1;
            std::vector<std::string> irnames(2);
            irnames[g] = "g"; irnames[u] = "u";
            point_group_table pg(testname, irnames, "g");
            pg.add_product(u, u, g);
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
            fail_test(testname, __FILE__, __LINE__,
                    "Checked out table deleted.");
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
            fail_test(testname, __FILE__, __LINE__,
                    "Checked out table deleted.");
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
