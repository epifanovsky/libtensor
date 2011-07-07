#include <libtensor/core/abs_index.h>
#include <libtensor/core/block_index_space.h>
#include <libtensor/symmetry/label/label_set.h>
#include <libtensor/symmetry/label/point_group_table.h>
#include "label_set_test.h"

namespace libtensor {

const char *label_set_test::table_id = "point_group";

void label_set_test::perform() throw(libtest::test_exception) {

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
        fail_test("label_set_test::perform()", __FILE__, __LINE__,
                e.what());
    }

    try {

        test_basic_1();
        test_basic_2();
        test_set_1();
        test_set_2();
        test_set_3();
        test_set_4();
        test_permute_1();

        product_table_container::get_instance().erase(table_id);

    } catch (libtest::test_exception) {
        product_table_container::get_instance().erase(table_id);
        throw;
    }
}

/** \test Tests assigning labels, setting evaluation mask, adding intrinsic
        labels, and clearing them
 **/
void label_set_test::test_basic_1() throw(libtest::test_exception) {

    static const char *testname = "label_set_test::test_basic_1()";

    try {

        index<4> i1, i2;
        i2[0] = 3; i2[1] = 3; i2[2] = 3; i2[3] = 3;
        block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
        mask<4> m0011, m1100;
        m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;
        bis.split(m1100, 2);
        bis.split(m0011, 1);
        bis.split(m0011, 2);
        bis.split(m0011, 3);

        mask<4> m0100, m0111;
        m0100[1] = true; m0111[1] = true; m0111[2] = true; m0111[3] = true;
        label_set<4> el(bis.get_block_index_dims(), table_id);

        // Check assignment
        el.assign(m0011, 0, 0); // ag
        el.assign(m0011, 1, 2); // au
        el.assign(m0011, 2, 1); // eg
        el.assign(m0011, 3, 3); // eu
        el.assign(m0100, 0, 0); // ag

        if (el.get_dim_type(1) == el.get_dim_type(2)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Equal types for dims 1 and 2.");
        }
        if (el.get_dim_type(2) != el.get_dim_type(3)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Non-equal types for dims 2 and 3.");
        }
        if (el.get_label(el.get_dim_type(1), 0) != 0) {
            fail_test(testname, __FILE__, __LINE__,
                    "Unexpected label at (1, 0).");
        }
        if (el.get_label(el.get_dim_type(1), 1) != (label_set<4>::label_t) -1) {
            fail_test(testname, __FILE__, __LINE__,
                    "Unexpected label at (1, 1).");
        }
        if (el.get_label(el.get_dim_type(2), 0) != 0) {
            fail_test(testname, __FILE__, __LINE__,
                    "Unexpected label at (2, 0).");
        }
        if (el.get_label(el.get_dim_type(2), 1) != 2) {
            fail_test(testname, __FILE__, __LINE__,
                    "Unexpected label at (2, 1).");
        }
        if (el.get_label(el.get_dim_type(2), 2) != 1) {
            fail_test(testname, __FILE__, __LINE__,
                    "Unexpected label at (2, 2).");
        }
        if (el.get_label(el.get_dim_type(2), 3) != 3) {
            fail_test(testname, __FILE__, __LINE__,
                    "Unexpected label at (2, 3).");
        }

        // Check intrinsic labels
        el.add_intrinsic(1);
        el.add_intrinsic(0);

        unsigned int i = 0;
        for (label_set<4>::iterator it = el.begin();
                it != el.end(); it++, i++) {

            if (el.get_intrinsic(it) != i) {
                fail_test(testname, __FILE__, __LINE__,
                        "Unexpected intrinsic label.");
            }
        }
        if (i != 2) {
            fail_test(testname, __FILE__, __LINE__,
                    "Wrong number of labels.");
        }

        el.set_mask(m0011);
        if (! m0011.equals(el.get_mask())) {
            fail_test(testname, __FILE__, __LINE__, "Evaluation mask.");
        }

        // Check clears
        el.clear_mask();
        if (! mask<4>().equals(el.get_mask())) {
            fail_test(testname, __FILE__, __LINE__,
                    "Evaluation mask not empty.");
        }

        el.clear_intrinsic();
        if (el.begin() != el.end()) {
            fail_test(testname, __FILE__, __LINE__,
                    "Intrinsic label still set.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Tests assigning labels and matching them
 **/
void label_set_test::test_basic_2() throw(libtest::test_exception) {

    static const char *testname = "label_set_test::test_basic_2()";

    try {

        index<4> i1, i2;
        i2[0] = 3; i2[1] = 3; i2[2] = 3; i2[3] = 3;
        block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
        mask<4> m0011, m1100;
        m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;
        bis.split(m1100, 2);
        bis.split(m0011, 1);
        bis.split(m0011, 2);
        bis.split(m0011, 3);

        mask<4> m0100, m0010, m0001, m0111;
        m0100[1] = true; m0010[2] = true; m0001[3] = true;
        m0111[1] = true; m0111[2] = true; m0111[3] = true;
        label_set<4> el(bis.get_block_index_dims(), table_id);

        // Check assignment
        for (label_set<4>::label_t i = 0; i < 4; i++) el.assign(m0001, i, i);
        el.assign(m0100, 0, 0); // ag
        el.assign(m0100, 1, 2); // au
        for (label_set<4>::label_t i = 0; i < 4; i++) el.assign(m0010, i, i);

        el.match_blk_labels();

        if (el.get_dim_type(1) == el.get_dim_type(2)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Equal types for dims 1 and 2.");
        }
        if (el.get_dim_type(2) != el.get_dim_type(3)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Non-equal types for dims 2 and 3.");
        }
        if (el.get_label(el.get_dim_type(1), 0) != 0) {
            fail_test(testname, __FILE__, __LINE__,
                    "Unexpected label at (1, 0).");
        }
        if (el.get_label(el.get_dim_type(1), 1) != 2) {
            fail_test(testname, __FILE__, __LINE__,
                    "Unexpected label at (1, 1).");
        }
        for (label_set<4>::label_t i = 0; i < 4; i++) {
            if (el.get_label(el.get_dim_type(2), i) != i) {
                fail_test(testname, __FILE__, __LINE__,
                        "Unexpected label in dim 2.");
            }
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/**	\test Two blocks, full label set, all blocks labeled, single
        intrinsic labels (2-dim)
 **/
void label_set_test::test_set_1() throw(libtest::test_exception) {

    static const char *testname = "label_set_test::test_set_1()";

    try {

        index<2> i1, i2;
        i2[0] = 3; i2[1] = 3;
        block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
        mask<2> m11;
        m11[0] = true; m11[1] = true;
        bis.split(m11, 2);

        label_set<2> elem1(bis.get_block_index_dims(), table_id);
        elem1.assign(m11, 0, 0); // ag
        elem1.assign(m11, 1, 2); // au
        elem1.add_intrinsic(0);

        label_set<2> elem2(elem1), elem3(elem1), elem4(elem1);
        elem2.clear_intrinsic();
        elem2.add_intrinsic(1);
        elem3.clear_intrinsic();
        elem3.add_intrinsic(2);
        elem4.clear_intrinsic();
        elem4.add_intrinsic(3);

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
        if(elem2.is_allowed(i01)) {
            fail_test(testname, __FILE__, __LINE__,
                    "elem2.is_allowed(i01)");
        }
        if(elem2.is_allowed(i10)) {
            fail_test(testname, __FILE__, __LINE__,
                    "elem2.is_allowed(i10)");
        }
        if(elem2.is_allowed(i11)) {
            fail_test(testname, __FILE__, __LINE__,
                    "elem2.is_allowed(i11)");
        }

        if(elem3.is_allowed(i00)) {
            fail_test(testname, __FILE__, __LINE__,
                    "elem3.is_allowed(i00)");
        }
        if(! elem3.is_allowed(i01)) {
            fail_test(testname, __FILE__, __LINE__,
                    "! elem3.is_allowed(i01)");
        }
        if(! elem3.is_allowed(i10)) {
            fail_test(testname, __FILE__, __LINE__,
                    "! elem3.is_allowed(i10)");
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


/**	\test Four blocks, all labeled, different index types, multiple
        intrinsic labels (2-dim)
 **/
void label_set_test::test_set_2() throw(libtest::test_exception) {

    static const char *testname = "label_set_test::test_set_2()";

    try {

        index<2> i1, i2;
        i2[0] = 4; i2[1] = 4;
        block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
        mask<2> m01, m10, m11;
        m10[0] = true; m01[1] = true;
        m11[0] = true; m11[1] = true;
        bis.split(m11, 1);
        bis.split(m11, 2);
        bis.split(m11, 3);
        dimensions<2> bidims = bis.get_block_index_dims();

        label_set<2> elem1(bidims, table_id);
        for (size_t i = 0; i < 4; i++) elem1.assign(m10, i, i);
        elem1.assign(m01, 0, 0); // ag
        elem1.assign(m01, 1, 2); // au
        elem1.assign(m01, 2, 1); // eg
        elem1.assign(m01, 3, 3); // eu
        elem1.add_intrinsic(0);

        label_set<2> elem2(elem1), elem3(elem1), elem4(elem1);
        elem2.add_intrinsic(1);
        elem3.add_intrinsic(1);
        elem3.add_intrinsic(2);
        elem3.add_intrinsic(3);
        elem4.clear_intrinsic();

        std::vector<bool> r1(bidims.get_size(), false);
        std::vector<bool> r2(bidims.get_size(), false);
        std::vector<bool> r3(bidims.get_size(), true);
        std::vector<bool> r4(bidims.get_size(), false);
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
            if (elem3.is_allowed(ai.get_index()) != r3[ai.get_abs_index()]) {
                std::ostringstream oss;
                oss << (r3[ai.get_abs_index()] ? "!" : "") <<
                        "elem3.is_allowed(" << ai.get_index() << ")";
                fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
            }
            if (elem4.is_allowed(ai.get_index()) != r4[ai.get_abs_index()]) {
                std::ostringstream oss;
                oss << (r4[ai.get_abs_index()] ? "!" : "") <<
                        "elem4.is_allowed(" << ai.get_index() << ")";
                fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
            }

        } while (ai.inc());

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/**	\test Four blocks, all labeled, different index types,
        partial eval mask, multiple intrinsic labels (2-dim)
 **/
void label_set_test::test_set_3() throw(libtest::test_exception) {

    static const char *testname = "label_set_test::test_set_3()";

    try {

        index<2> i1, i2;
        i2[0] = 4; i2[1] = 4;
        block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
        mask<2> m01, m10, m11;
        m10[0] = true; m01[1] = true;
        m11[0] = true; m11[1] = true;
        bis.split(m11, 1);
        bis.split(m11, 2);
        bis.split(m11, 3);
        dimensions<2> bidims = bis.get_block_index_dims();

        label_set<2> elem1(bidims, table_id);
        elem1.set_mask(m10);
        for (size_t i = 0; i < 4; i++) elem1.assign(m10, i, i);
        elem1.assign(m01, 0, 0); // ag
        elem1.assign(m01, 1, 2); // au
        elem1.assign(m01, 2, 1); // eg
        elem1.add_intrinsic(0);

        label_set<2> elem2(elem1), elem3(elem1), elem4(elem1);
        elem2.add_intrinsic(3);
        elem3.add_intrinsic(1);
        elem3.add_intrinsic(2);
        elem3.add_intrinsic(3);
        elem4.clear_intrinsic();

        std::vector<bool> r1(bidims.get_size(), false);
        std::vector<bool> r2(bidims.get_size(), false);
        std::vector<bool> r3(bidims.get_size(), true);
        std::vector<bool> r4(bidims.get_size(), false);
        r1[0] = r1[1] = r1[2] = r1[3] = true;
        r2[0] = r2[1] = r2[2] = r2[3] =
                r2[12] = r2[13] = r2[14] = r2[15] = true;

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
            if (elem3.is_allowed(ai.get_index()) != r3[ai.get_abs_index()]) {
                std::ostringstream oss;
                oss << (r3[ai.get_abs_index()] ? "!" : "") <<
                        "elem3.is_allowed(" << ai.get_index() << ")";
                fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
            }
            if (elem4.is_allowed(ai.get_index()) != r4[ai.get_abs_index()]) {
                std::ostringstream oss;
                oss << (r4[ai.get_abs_index()] ? "!" : "") <<
                        "elem4.is_allowed(" << ai.get_index() << ")";
                fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
            }

        } while (ai.inc());

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/**	\test Four blocks, unlabeled blocks, different index types (2-dim)
 **/
void label_set_test::test_set_4() throw(libtest::test_exception) {

    static const char *testname = "label_set_test::test_set_4()";

    try {

        index<2> i1, i2;
        i2[0] = 4; i2[1] = 4;
        block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
        mask<2> m01, m10, m11;
        m10[0] = true; m01[1] = true;
        m11[0] = true; m11[1] = true;
        bis.split(m11, 1);
        bis.split(m11, 2);
        bis.split(m11, 3);
        dimensions<2> bidims = bis.get_block_index_dims();

        label_set<2> elem1(bidims, table_id);
        elem1.set_mask(m10);
        for (size_t i = 0; i < 3; i++) elem1.assign(m10, i, i);
        elem1.assign(m01, 0, 0); // ag
        elem1.assign(m01, 1, 2); // au
        elem1.assign(m01, 2, 1); // eg
        elem1.assign(m01, 3, 3); // eg
        elem1.add_intrinsic(0);

        label_set<2> elem2(elem1), elem3(elem1), elem4(elem1);
        elem2.add_intrinsic(2);
        elem3.add_intrinsic(1);
        elem3.add_intrinsic(2);
        elem3.add_intrinsic(3);
        elem4.clear_intrinsic();

        std::vector<bool> r1(bidims.get_size(), false);
        std::vector<bool> r2(bidims.get_size(), false);
        std::vector<bool> r3(bidims.get_size(), true);
        std::vector<bool> r4(bidims.get_size(), false);
        r1[0] = r1[1] = r1[2] = r1[3] =
                r1[12] = r1[13] = r1[14] = r1[15] = true;
        r2[0] = r2[1] = r2[2] = r2[3] =
                r2[8] = r2[9] = r2[10] = r2[11] =
                        r2[12] = r2[13] = r2[14] = r2[15] = true;
        r4[12] = r4[13] = r4[14] = r4[15] = true;

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
            if (elem3.is_allowed(ai.get_index()) != r3[ai.get_abs_index()]) {
                std::ostringstream oss;
                oss << (r3[ai.get_abs_index()] ? "!" : "") <<
                        "elem3.is_allowed(" << ai.get_index() << ")";
                fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
            }
            if (elem4.is_allowed(ai.get_index()) != r4[ai.get_abs_index()]) {
                std::ostringstream oss;
                oss << (r4[ai.get_abs_index()] ? "!" : "") <<
                        "elem4.is_allowed(" << ai.get_index() << ")";
                fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
            }

        } while (ai.inc());



    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/**	\test Full set, partial eval mask, permutation (3-dim)
 **/
void label_set_test::test_permute_1() throw(libtest::test_exception) {

    static const char *testname = "label_set_test::test_permute_1()";

    try {

        index<3> i1, i2;
        i2[0] = 3; i2[1] = 3; i2[2] = 5;
        block_index_space<3> bis(dimensions<3>(index_range<3>(i1, i2)));
        mask<3> m001, m011, m110, m111;
        m110[0] = true; m110[1] = true; m001[2] = true;
        m011[1] = true; m011[2] = true;
        m111[0] = true; m111[1] = true; m111[2] = true;
        bis.split(m110, 1); bis.split(m110, 2); bis.split(m110, 3);
        bis.split(m001, 1); bis.split(m001, 2); bis.split(m001, 3);
        bis.split(m001, 4); bis.split(m001, 5);

        dimensions<3> bidims = bis.get_block_index_dims();
        label_set<3> elem1(bidims, table_id);

        // assign labels
        for (size_t i = 0; i < 4; i++) elem1.assign(m110, i, i);
        elem1.assign(m001, 0, 0); elem1.assign(m001, 1, 1);
        elem1.assign(m001, 2, 1); elem1.assign(m001, 3, 2);
        elem1.assign(m001, 4, 3); elem1.assign(m001, 5, 3);

        elem1.set_mask(m011);

        elem1.add_intrinsic(0);

        label_set<3> elem2(elem1), elem3(elem1), elem4(elem1);
        elem2.clear_intrinsic();
        elem2.add_intrinsic(2);
        elem2.add_intrinsic(3);
        elem3.add_intrinsic(1);
        elem3.add_intrinsic(2);
        elem3.add_intrinsic(3);
        elem4.clear_intrinsic();

        permutation<3> p; p.permute(0, 1).permute(1, 2);
        elem1.permute(p);
        elem2.permute(p);
        elem3.permute(p);
        elem4.permute(p);

        bidims.permute(p);
        std::vector<bool> r1(bidims.get_size(), false);
        std::vector<bool> r2(bidims.get_size(), false);
        std::vector<bool> r3(bidims.get_size(), true);
        std::vector<bool> r4(bidims.get_size(), false);

        for (size_t i = 0; i < 4; i++) {
            r1[i     ] = true; // x00
            r1[i + 28] = r1[i + 32] = true;
            r1[i + 60] = true; // x23
            r1[i + 88] = r1[i + 92] = true;
            r2[i + 12] = r2[i + 16] = r2[i + 20] = true;
            r2[i + 36] = r2[i + 40] = r2[i + 44] = true;
            r2[i + 48] = r2[i + 52] = r2[i + 56] = true;
            r2[i + 72] = r2[i + 76] = r2[i + 80] = true;
        }

        abs_index<3> ai(bidims);
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
            if (elem3.is_allowed(ai.get_index()) != r3[ai.get_abs_index()]) {
                std::ostringstream oss;
                oss << (r3[ai.get_abs_index()] ? "!" : "") <<
                        "elem3.is_allowed(" << ai.get_index() << ")";
                fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
            }
            if (elem4.is_allowed(ai.get_index()) != r4[ai.get_abs_index()]) {
                std::ostringstream oss;
                oss << (r4[ai.get_abs_index()] ? "!" : "") <<
                        "elem4.is_allowed(" << ai.get_index() << ")";
                fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
            }

        } while (ai.inc());

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

} // namespace libtensor
