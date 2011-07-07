#include <libtensor/btod/transf_double.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/symmetry/label/point_group_table.h>
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

        test_empty();
        test_set_1();
        test_set_2();
        test_set_3();
        test_set_4();
        test_permute();

        product_table_container::get_instance().erase(table_id);

    } catch (libtest::test_exception) {
        product_table_container::get_instance().erase(table_id);
        throw;
    }
}


/**	\test Empty se_label<N, T>
 **/
void se_label_test::test_empty() throw(libtest::test_exception) {

    static const char *testname = "se_label_test::test_empty()";

    try {

        index<2> i1, i2;
        i2[0] = 9; i2[1] = 9;
        block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
        mask<2> m11;
        m11[0] = true; m11[1] = true;
        bis.split(m11, 5);

        dimensions<2> bidims = bis.get_block_index_dims();
        se_label<2, double> elem1(bidims);

        se_label<2, double>::iterator it = elem1.begin();
        if (it != elem1.end()) {
            fail_test(testname, __FILE__, __LINE__, "Subset found.");
        }

        abs_index<2> ai(bidims);
        do {

            if (! elem1.is_allowed(ai.get_index())) {
                std::ostringstream oss;
                oss << "! elem1.is_allowed(" << ai.get_index() << ")";
                fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
            }
        } while (ai.inc());

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/**	\test One subset, full mask
 **/
void se_label_test::test_set_1() throw(libtest::test_exception) {

    static const char *testname = "se_label_test::test_set_1()";

    try {

        index<2> i1, i2;
        i2[0] = 3; i2[1] = 5;
        block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
        mask<2> m01, m10, m11;
        m10[0] = true; m01[1] = true;
        bis.split(m01, 1);
        bis.split(m01, 3);
        bis.split(m01, 5);
        bis.split(m10, 1);
        bis.split(m10, 2);
        bis.split(m10, 3);

        dimensions<2> bidims = bis.get_block_index_dims();
        se_label<2, double> elem1(bidims);

        label_set<2> &subset1 = elem1.create_subset(table_id);
        for (size_t i = 0; i < 4; i++) subset1.assign(m10, i, i);
        subset1.assign(m01, 0, 0); // ag
        subset1.assign(m01, 1, 2); // au
        subset1.assign(m01, 2, 1); // eg
        subset1.assign(m01, 3, 3); // eu
        subset1.add_intrinsic(0);

        size_t nss = 0;
        for (se_label<2, double>::const_iterator it = elem1.begin();
                it != elem1.end(); it++) nss++;
        if (nss != 1) {
            fail_test(testname, __FILE__, __LINE__,
                    "Wrong number of subsets.");
        }

        se_label<2, double> elem2(elem1), elem3(elem1), elem4(elem1);
        nss = 0;
        for (se_label<2, double>::const_iterator it = elem2.begin();
                it != elem2.end(); it++) nss++;
        if (nss != 1) {
            fail_test(testname, __FILE__, __LINE__,
                    "Wrong number of subsets.");
        }

        se_label<2, double>::iterator it2 = elem2.begin();
        label_set<2> &subset2 = elem2.get_subset(it2);
        label_set<2> &subset3 = elem3.get_subset(elem3.begin());
        label_set<2> &subset4 = elem4.get_subset(elem4.begin());
        subset2.add_intrinsic(1);
        subset3.add_intrinsic(1);
        subset3.add_intrinsic(2);
        subset3.add_intrinsic(3);
        subset4.clear_intrinsic();

        std::vector<bool> r1(bidims.get_size(), false);
        std::vector<bool> r2(bidims.get_size(), false);
        r1[0] = r1[6] = r1[9] = r1[15] = true;
        r2[0] = r2[2] = r2[4] = r2[6] =
                r2[9] = r2[11] = r2[13] = r2[15] = true;

        abs_index<2> ai(bidims);
        do {
            if (elem1.is_allowed(ai.get_index()) != r1[ai.get_abs_index()]) {
                std::ostringstream oss;
                oss << (r1[ai.get_abs_index()] ? "!" : "") <<
                        "elem1.is_allowed(" << ai.get_index() << ")";
                fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
            }
            if (elem2.is_allowed(ai.get_index()) != r2[ai.get_abs_index()]) {
                std::ostringstream oss;
                oss << (r2[ai.get_abs_index()] ? "!" : "") <<
                        "elem2.is_allowed(" << ai.get_index() << ")";
                fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
            }
            if (! elem3.is_allowed(ai.get_index())) {
                std::ostringstream oss;
                oss << "! elem3.is_allowed(" << ai.get_index() << ")";
                fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
            }
            if (elem4.is_allowed(ai.get_index())) {
                std::ostringstream oss;
                oss << "elem4.is_allowed(" << ai.get_index() << ")";
                fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
            }

        } while (ai.inc());

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/**	\test One subset, partial evaluation mask
 **/
void se_label_test::test_set_2() throw(libtest::test_exception) {

    static const char *testname = "se_label_test::test_set_2()";

    try {

        index<2> i1, i2;
        i2[0] = 3; i2[1] = 5;
        block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
        mask<2> m01, m10;
        m10[0] = true; m01[1] = true;
        bis.split(m01, 1);
        bis.split(m01, 3);
        bis.split(m01, 5);
        bis.split(m10, 1);
        bis.split(m10, 2);
        bis.split(m10, 3);

        dimensions<2> bidims = bis.get_block_index_dims();
        se_label<2, double> elem1(bidims);

        label_set<2> &subset1 = elem1.create_subset(table_id);
        subset1.assign(m01, 0, 0); // ag
        subset1.assign(m01, 1, 2); // au
        subset1.assign(m01, 2, 1); // eg
        subset1.assign(m01, 3, 3); // eu
        subset1.add_intrinsic(0);
        subset1.set_mask(m01);

        se_label<2, double> elem2(elem1), elem3(elem1), elem4(elem1);
        label_set<2> &subset2 = elem2.get_subset(elem2.begin());
        label_set<2> &subset3 = elem3.get_subset(elem3.begin());
        label_set<2> &subset4 = elem4.get_subset(elem4.begin());
        subset2.add_intrinsic(1);
        subset3.add_intrinsic(1);
        subset3.add_intrinsic(2);
        subset3.add_intrinsic(3);
        subset4.clear_intrinsic();

        std::vector<bool> r1(bidims.get_size(), false);
        std::vector<bool> r2(bidims.get_size(), false);
        r1[0] = r1[4] = r1[8] = r1[12] = true;
        r2[0] = r2[2] = r2[4] = r2[6] =
                r2[8] = r2[10] = r2[12] = r2[14] = true;

        abs_index<2> ai(bidims);
        do {
            if (elem1.is_allowed(ai.get_index()) != r1[ai.get_abs_index()]) {
                std::ostringstream oss;
                oss << (r1[ai.get_abs_index()] ? "!" : "") <<
                        " elem1.is_allowed(" << ai.get_index() << ")";
                fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
            }
            if (elem2.is_allowed(ai.get_index()) != r2[ai.get_abs_index()]) {
                std::ostringstream oss;
                oss << (r2[ai.get_abs_index()] ? "!" : "") <<
                        " elem2.is_allowed(" << ai.get_index() << ")";
                fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
            }
            if (! elem3.is_allowed(ai.get_index())) {
                std::ostringstream oss;
                oss << "! elem3.is_allowed(" << ai.get_index() << ")";
                fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
            }
            if (elem4.is_allowed(ai.get_index())) {
                std::ostringstream oss;
                oss << "elem4.is_allowed(" << ai.get_index() << ")";
                fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
            }

        } while (ai.inc());

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/**	\test Two subsets, no unmasked dimensions
 **/
void se_label_test::test_set_3() throw(libtest::test_exception) {

    static const char *testname = "se_label_test::test_set_3()";

    try {

        index<2> i1, i2;
        i2[0] = 3; i2[1] = 5;
        block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
        mask<2> m01, m10;
        m10[0] = true; m01[1] = true;
        bis.split(m01, 1);
        bis.split(m01, 3);
        bis.split(m01, 5);
        bis.split(m10, 1);
        bis.split(m10, 2);
        bis.split(m10, 3);

        dimensions<2> bidims = bis.get_block_index_dims();
        se_label<2, double> elem1(bidims);
        label_set<2> &subset1a = elem1.create_subset(table_id);
        subset1a.assign(m01, 0, 0); // ag
        subset1a.assign(m01, 1, 2); // au
        subset1a.assign(m01, 2, 1); // eg
        subset1a.assign(m01, 3, 3); // eu
        subset1a.add_intrinsic(0);
        subset1a.set_mask(m01);

        label_set<2> &subset1b = elem1.create_subset(table_id);
        for (size_t i = 0; i < 4; i++) subset1b.assign(m10, i, i);
        subset1b.add_intrinsic(1);
        subset1b.set_mask(m10);

        std::vector<bool> r1(bidims.get_size(), false);
        r1[0] = r1[4] = r1[5] = r1[6] = r1[7] = r1[8] = r1[12] = true;

        abs_index<2> ai(bidims);
        do {
            if (elem1.is_allowed(ai.get_index()) != r1[ai.get_abs_index()]) {
                std::ostringstream oss;
                oss << (r1[ai.get_abs_index()] ? "!" : "") <<
                        "elem1.is_allowed(" << ai.get_index() << ")";
                fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
            }

        } while (ai.inc());

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/**	\test Two subsets, unmasked dimensions
 **/
void se_label_test::test_set_4() throw(libtest::test_exception) {

    static const char *testname = "se_label_test::test_set_4()";

    try {

        index<3> i1, i2;
        i2[0] = 3; i2[1] = 3; i2[2] = 5;
        block_index_space<3> bis(dimensions<3>(index_range<3>(i1, i2)));
        mask<3> m001, m110, m100;
        m100[0] = m110[0] = m110[1] = m001[2] = true;
        bis.split(m110, 1);
        bis.split(m110, 2);
        bis.split(m110, 3);
        bis.split(m001, 1);
        bis.split(m001, 3);
        bis.split(m001, 5);

        dimensions<3> bidims = bis.get_block_index_dims();
        se_label<3, double> elem1(bidims);
        label_set<3> &subset1a = elem1.create_subset(table_id);
        subset1a.assign(m001, 0, 0); // ag
        subset1a.assign(m001, 1, 2); // au
        subset1a.assign(m001, 2, 1); // eg
        subset1a.assign(m001, 3, 3); // eu
        subset1a.add_intrinsic(0);
        subset1a.set_mask(m001);

        label_set<3> &subset1b = elem1.create_subset(table_id);
        for (size_t i = 0; i < 4; i++) subset1b.assign(m100, i, i);
        subset1b.add_intrinsic(1);
        subset1b.set_mask(m110);

        abs_index<3> ai(bidims);
        do {
            if (! elem1.is_allowed(ai.get_index())) {
                std::ostringstream oss;
                oss << "! elem1.is_allowed(" << ai.get_index() << ")";
                fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
            }

        } while (ai.inc());

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}

/**	\test Two subsets, no unmasked dimensions, permutation
 **/
void se_label_test::test_permute() throw(libtest::test_exception) {

    static const char *testname = "se_label_test::test_permute()";

    try {

        index<4> i1, i2;
        i2[0] = 3; i2[1] = 3; i2[2] = 5; i2[3] = 5;
        block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
        mask<4> m0001, m0011, m0100, m0101, m1010, m1100;
        m0100[1] = true; m0001[3] = true;
        m0101[1] = true; m0101[3] = true;
        m1010[0] = true; m1010[2] = true;
        m0011[2] = true; m0011[3] = true;
        m1100[0] = true; m1100[1] = true;
        bis.split(m1100, 1);
        bis.split(m1100, 2);
        bis.split(m1100, 3);
        bis.split(m0011, 1);
        bis.split(m0011, 3);
        bis.split(m0011, 5);

        dimensions<4> bidims = bis.get_block_index_dims();
        se_label<4, double> elem1(bidims);

        label_set<4> &subset1a = elem1.create_subset(table_id);
        label_set<4> &subset1b = elem1.create_subset(table_id);

        // assign labels
        for (size_t i = 0; i < 4; i++) {
            subset1a.assign(m0001, i, i);
            subset1b.assign(m1010, i, i);
        }

        // assign different labels (two label types)
        subset1a.assign(m0100, 0, 0);
        subset1a.assign(m0100, 1, 2);
        subset1a.assign(m0100, 2, 3);
        subset1a.assign(m0100, 3, 1);

        subset1a.add_intrinsic(0);
        subset1b.add_intrinsic(1);
        subset1a.set_mask(m0101);
        subset1b.set_mask(m1010);

        permutation<4> p; p.permute(0, 1).permute(1, 3);
        elem1.permute(p);
        bidims.permute(p);

        std::vector<bool> r1(bidims.get_size(), false);
        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < 4; j++) {
                r1[i *  4 + j           ] = r1[i *  4 + j      +  96] = true;
                r1[i *  4 + j      + 176] = r1[i *  4 + j      + 208] = true;
                r1[i * 64 + j * 16 +   4] = true;
                r1[i * 64 + j * 16 +   1] = r1[i * 64 + j * 16 +   5] = true;
                r1[i * 64 + j * 16 +  14] = true;
                r1[i * 64 + j * 16 +  11] = r1[i * 64 + j * 16 +  15] = true;
            }
        }

        abs_index<4> ai(bidims);
        do {
            if (elem1.is_allowed(ai.get_index()) != r1[ai.get_abs_index()]) {
                std::ostringstream oss;
                oss << (r1[ai.get_abs_index()] ? "!" : "") <<
                        "elem1.is_allowed(" << ai.get_index() << ")";
                fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
            }

        } while (ai.inc());

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


} // namespace libtensor
