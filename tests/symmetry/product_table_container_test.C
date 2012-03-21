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

        typedef point_group_table::irrep_label_t irrep_label_t;
        typedef point_group_table::irrep_group_t irrep_group_t;
        typedef point_group_table::irrep_map_t irrep_map_t;

        product_table_container &ptc = product_table_container::get_instance();

        irrep_label_t ag = 0, bg = 1, au = 2, bu = 3, g = 0, u = 1;
        irrep_map_t in1, in2;
        in1[ag] = "Ag"; in1[bg] = "Bg"; in1[au] = "Au"; in1[bu] = "Bu";
        in2[g] = "g"; in2[u] = "u";
        point_group_table pg1(testname, in1, ag), pg2(testname, in2, g);
        pg1.add_product(bg, au, bu);
        pg1.add_product(bg, bu, au);
        pg1.add_product(au, bu, bg);
        pg1.check();
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


/**	\test Requesting and returning tables
 **/
void product_table_container_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "product_table_container_test::test_2()";

    try {

        typedef point_group_table::irrep_label_t irrep_label_t;
        typedef point_group_table::irrep_group_t irrep_group_t;
        typedef point_group_table::irrep_map_t irrep_map_t;

        product_table_container &ptc = product_table_container::get_instance();

        { // Setup point group table
            irrep_label_t g = 0, u = 1;
            irrep_map_t irnames;
            irnames[g] = "g"; irnames[u] = "u";
            point_group_table pg(testname, irnames, g);
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

        typedef point_group_table::irrep_label_t irrep_label_t;
        typedef point_group_table::irrep_group_t irrep_group_t;
        typedef point_group_table::irrep_map_t irrep_map_t;

        product_table_container &ptc = product_table_container::get_instance();

        { // Setup point group table
            irrep_label_t g = 0, u = 1;
            irrep_map_t irnames;
            irnames[g] = "g"; irnames[u] = "u";
            point_group_table pg(testname, irnames, g);
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
