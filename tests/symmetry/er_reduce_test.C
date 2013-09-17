#include <libtensor/symmetry/inst/er_optimize.h>
#include <libtensor/symmetry/inst/er_reduce.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/print_symmetry.h>
#include "er_reduce_test.h"

namespace libtensor {


void er_reduce_test::perform() throw(libtest::test_exception) {

    std::string s6 = "S6", c2v = "C2v";
    setup_pg_table(s6);

    try {

        test_1(s6);
        test_2(s6);
        test_3(s6);
        test_4(s6);
        test_5(s6);


    } catch (libtest::test_exception &e) {
        clear_pg_table(s6);
        throw;
    }

    clear_pg_table(s6);

    setup_pg_table(c2v);

    try {

        test_6(c2v);
        test_7(c2v);

    } catch (libtest::test_exception &e) {
        clear_pg_table(c2v);
        throw;
    }

    clear_pg_table(c2v);

}


/** \brief Reduce of two dimensions in one step (complete dim)
 **/
void er_reduce_test::test_1(
        const std::string &id) throw(libtest::test_exception) {

    static const char *testname = "er_reduce_test::test_1()";

    typedef product_table_i::label_set_t label_set_t;
    typedef product_table_i::label_group_t label_group_t;

    evaluation_rule<4> r1;
    evaluation_rule<2> r2;

    try {

        sequence<4, size_t> seq1(1), seq2(1);
        seq1[2] = 0; seq1[3] = 0;
        seq2[0] = 0; seq2[1] = 0;
        product_rule<4> &pr1 = r1.new_product();
        pr1.add(seq1, 1);
        pr1.add(seq2, 3);

        sequence<4, size_t> rmap(0);
        rmap[0] = 0; rmap[1] = 2; rmap[2] = 1; rmap[3] = 2;
        sequence<2, label_group_t> rdims;
        rdims[0].push_back(0); rdims[0].push_back(1);
        rdims[0].push_back(2); rdims[0].push_back(3);

        er_reduce<4, 2>(r1, rmap, rdims, id).perform(r2);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

    // Check sequence list
    const eval_sequence_list<2> &sl = r2.get_sequences();
    if (sl.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# seq not optimized.");
    }
    if (sl[0][0] != 1 || sl[0][1] != 1) {
        fail_test(testname, __FILE__, __LINE__, "Wrong sequence.");
    }

    // Check product list
    evaluation_rule<2>::iterator it = r2.begin();
    if (it == r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product list.");
    }
    const product_rule<2> &pr1 = r2.get_product(it);
    it++;
    if (it == r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Only one product.");
    }
    const product_rule<2> &pr2 = r2.get_product(it);
    it++;
    if (it != r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "More than two products.");
    }

    product_rule<2>::iterator ip1 = pr1.begin(), ip2 = pr2.begin();
    if (ip1 == pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product pr1.");
    }
    if (ip2 == pr2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product pr2.");
    }
    if ((pr1.get_intrinsic(ip1) != 2 || pr2.get_intrinsic(ip2) != 3) &&
            (pr1.get_intrinsic(ip1) != 3 || pr2.get_intrinsic(ip2) != 2)) {
        fail_test(testname, __FILE__, __LINE__, "Wrong intrinsic labels.");
    }

    ip1++; ip2++;
    if (ip1 != pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Multiple terms in pr1.");
    }
    if (ip2 != pr2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Multiple terms in pr2.");
    }
}


/** \brief Reduction of four dimensions in two steps (complete dim)
 **/
void er_reduce_test::test_2(
        const std::string &id) throw(libtest::test_exception) {


    static const char *testname = "er_reduce_test::test_2()";

    typedef product_table_i::label_set_t label_set_t;
    typedef product_table_i::label_group_t label_group_t;

    evaluation_rule<6> r1;
    evaluation_rule<2> r2;

    try {

        sequence<6, size_t> seq1(1), seq2(1);
        seq1[3] = 0; seq1[4] = 0; seq1[5] = 0;
        seq2[0] = 0; seq2[1] = 0; seq2[2] = 0;
        product_rule<6> &pr1 = r1.new_product();
        pr1.add(seq1, 1);
        pr1.add(seq2, 3);

        sequence<6, size_t> rmap(0);
        rmap[0] = 0; rmap[1] = 2; rmap[2] = 3;
        rmap[3] = 1; rmap[4] = 3; rmap[5] = 2;
        sequence<4, label_group_t> rdims;
        rdims[0].push_back(0); rdims[0].push_back(1);
        rdims[0].push_back(2); rdims[0].push_back(3);
        rdims[1].push_back(0); rdims[1].push_back(1);
        rdims[1].push_back(2); rdims[1].push_back(3);

        evaluation_rule<2> tmp;
        er_reduce<6, 4>(r1, rmap, rdims, id).perform(tmp);
        er_optimize<2>(tmp, id).perform(r2);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

    // Check sequence list
    const eval_sequence_list<2> &sl = r2.get_sequences();
    if (sl.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# seq not optimized.");
    }
    if (sl[0][0] != 1 || sl[0][1] != 1) {
        fail_test(testname, __FILE__, __LINE__, "Wrong sequence.");
    }

    // Check product list
    evaluation_rule<2>::iterator it = r2.begin();
    if (it == r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product list.");
    }
    const product_rule<2> &pr1 = r2.get_product(it);
    it++;
    if (it == r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Only one product.");
    }
    const product_rule<2> &pr2 = r2.get_product(it);
    it++;
    if (it != r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "More than two products.");
    }

    product_rule<2>::iterator ip1 = pr1.begin(), ip2 = pr2.begin();
    if (ip1 == pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product pr1.");
    }
    if (ip2 == pr2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product pr2.");
    }
    if ((pr1.get_intrinsic(ip1) != 2 || pr2.get_intrinsic(ip2) != 3) &&
            (pr1.get_intrinsic(ip1) != 3 || pr2.get_intrinsic(ip2) != 2)) {
        fail_test(testname, __FILE__, __LINE__, "Wrong intrinsic labels.");
    }

    ip1++; ip2++;
    if (ip1 != pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Multiple terms in pr1.");
    }
    if (ip2 != pr2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Multiple terms in pr2.");
    }
}


/** \brief Reduction of six dimensions in three steps (complete dims)
 **/
void er_reduce_test::test_3(
        const std::string &id) throw(libtest::test_exception) {


    static const char *testname = "er_reduce_test::test_3()";

    typedef product_table_i::label_set_t label_set_t;
    typedef product_table_i::label_group_t label_group_t;

    evaluation_rule<8> r1;
    evaluation_rule<2> r2;

    try {

        sequence<8, size_t> seq1(0), seq2(0), seq3(0), seq4(0);
        seq1[0] = 1; seq1[1] = 1; seq2[2] = 1; seq2[3] = 1;
        seq3[4] = 1; seq3[5] = 1; seq4[6] = 1; seq4[7] = 1;
        product_rule<8> &pr1 = r1.new_product();
        pr1.add(seq1, 0);
        pr1.add(seq2, 1);
        pr1.add(seq3, 2);
        pr1.add(seq4, 0);

        sequence<8, size_t> rmap(0);
        rmap[0] = 0; rmap[1] = 2; rmap[2] = 4; rmap[3] = 2;
        rmap[4] = 4; rmap[5] = 3; rmap[6] = 3; rmap[7] = 1;
        sequence<6, label_group_t> rdims;
        rdims[0].push_back(0); rdims[0].push_back(1);
        rdims[0].push_back(2); rdims[0].push_back(3);
        rdims[1].push_back(0); rdims[1].push_back(1);
        rdims[1].push_back(2); rdims[1].push_back(3);
        rdims[2].push_back(0); rdims[2].push_back(1);
        rdims[2].push_back(2); rdims[2].push_back(3);

        er_reduce<8, 6>(r1, rmap, rdims, id).perform(r2);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

    // Check sequence list
    const eval_sequence_list<2> &sl = r2.get_sequences();
    if (sl.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# seq not optimized.");
    }
    if (sl[0][0] != 1 || sl[0][1] != 1) {
        fail_test(testname, __FILE__, __LINE__, "Wrong sequence.");
    }

    // Check product list
    evaluation_rule<2>::iterator it = r2.begin();
    if (it == r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product list.");
    }
    const product_rule<2> &pr1 = r2.get_product(it);
    it++;
    if (it != r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Multiple products.");
    }
    product_rule<2>::iterator ip1 = pr1.begin();
    if (ip1 == pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product pr1.");
    }
    if (pr1.get_intrinsic(ip1) != 3) {
        fail_test(testname, __FILE__, __LINE__, "Wrong intrinsic label.");
    }
    ip1++;
    if (ip1 != pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Multiple terms in pr1.");
    }


}


/** \brief Reduction of four dimensions in two steps (one complete dims)
 **/
void er_reduce_test::test_4(
        const std::string &id) throw(libtest::test_exception) {


    static const char *testname = "er_reduce_test::test_4()";

    typedef product_table_i::label_t label_t;
    typedef product_table_i::label_set_t label_set_t;
    typedef product_table_i::label_group_t label_group_t;

    evaluation_rule<6> r1;
    evaluation_rule<2> r2;

    try {

        sequence<6, size_t> seq1(0), seq2(0), seq3(0);
        seq1[0] = 1; seq1[1] = 1;
        seq2[2] = 1; seq2[3] = 1;
        seq3[4] = 1; seq3[5] = 1;
        product_rule<6> &pr1 = r1.new_product();
        pr1.add(seq1, 0);
        pr1.add(seq2, 1);
        pr1.add(seq3, 2);

        sequence<6, size_t> rmap(0);
        rmap[0] = 0; rmap[1] = 2; rmap[2] = 3;
        rmap[3] = 2; rmap[4] = 3; rmap[5] = 1;

        sequence<4, label_group_t> rdims;
        rdims[0].push_back(0); rdims[0].push_back(1);
        rdims[0].push_back(2); rdims[0].push_back(3);
        rdims[1].push_back(0); rdims[1].push_back(2);

        evaluation_rule<2> tmp;
        er_reduce<6, 4>(r1, rmap, rdims, id).perform(tmp);
        er_optimize<2>(tmp, id).perform(r2);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

    // Check sequence list
    const eval_sequence_list<2> &sl = r2.get_sequences();
    if (sl.size() != 2) {
        fail_test(testname, __FILE__, __LINE__, "# seq.");
    }
    size_t isa = 0;
    if (sl[0][0] == 1 && sl[0][1] == 0) {
        if (sl[1][0] != 0 || sl[1][1] != 1)
            fail_test(testname, __FILE__, __LINE__, "2nd seq.");
    }
    else if (sl[0][0] == 0 && sl[0][1] == 1) {
        if (sl[1][0] != 1 || sl[1][1] != 0)
            fail_test(testname, __FILE__, __LINE__, "2nd seq.");

        isa = 1;
    }
    else {
        fail_test(testname, __FILE__, __LINE__, "1st seq.");
    }

    // Check product list
    evaluation_rule<2>::iterator it = r2.begin();
    if (it == r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product list.");
    }
    const product_rule<2> &pr1 = r2.get_product(it);
    it++;
    if (it == r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Only one product.");
    }
    const product_rule<2> &pr2 = r2.get_product(it);
    it++;
    if (it != r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "More than two products.");
    }
    product_rule<2>::iterator ip1a = pr1.begin(), ip1b = pr1.begin();
    product_rule<2>::iterator ip2a = pr2.begin(), ip2b = pr2.begin();
    ip1b++; ip2b++;
    if (ip1a == pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product pr1.");
    }
    if (ip1b == pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Only one term in pr1.");
    }
    if (ip2a == pr2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product pr2.");
    }
    if (ip2b == pr2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Only one term in pr2.");
    }

    label_t l1a(1), l1b(2), l2a(3), l2b(0);
    if (ip1a->first != isa) std::swap(l1a, l1b);
    if (ip2a->first != isa) std::swap(l2a, l2b);

    if (pr1.get_intrinsic(ip1a) != l1a)
        fail_test(testname, __FILE__, __LINE__, "Intrinsic l1a.");
    if (pr1.get_intrinsic(ip1b) != l1b)
        fail_test(testname, __FILE__, __LINE__, "Intrinsic l1b.");
    if (pr1.get_intrinsic(ip2a) != l2a)
        fail_test(testname, __FILE__, __LINE__, "Intrinsic l2a.");
    if (pr1.get_intrinsic(ip2b) != l2b)
        fail_test(testname, __FILE__, __LINE__, "Intrinsic l2b.");

    ip1b++;
    if (ip1b != pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Too many terms in pr1.");
    }
    ip2b++;
    if (ip2b != pr2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Too many terms in pr2.");
    }

}


/** \brief Reduction of two dimensions in one step (complete dims)
 **/
void er_reduce_test::test_5(
        const std::string &id) throw(libtest::test_exception) {

    static const char *testname = "er_reduce_test::test_5()";

    typedef product_table_i::label_t label_t;
    typedef product_table_i::label_set_t label_set_t;
    typedef product_table_i::label_group_t label_group_t;

    evaluation_rule<4> r1;
    evaluation_rule<2> r2;

    try {

        sequence<4, size_t> seq1(1);
        product_rule<4> &pr1 = r1.new_product();
        pr1.add(seq1, 2);

        sequence<4, size_t> rmap(0);
        rmap[0] = 0; rmap[1] = 1; rmap[2] = 2; rmap[3] = 2;
        sequence<2, label_group_t> rdims;
        rdims[0].push_back(0); rdims[0].push_back(1);
        rdims[0].push_back(2); rdims[0].push_back(3);

        evaluation_rule<2> tmp;
        er_reduce<4, 2>(r1, rmap, rdims, id).perform(tmp);
        er_optimize<2>(tmp, id).perform(r2);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

    // Check sequence list
    const eval_sequence_list<2> &sl = r2.get_sequences();
    if (sl.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# seq.");
    }
    if (sl[0][0] != 1 || sl[0][1] != 1) {
        fail_test(testname, __FILE__, __LINE__, "seq.");
    }

    // Check product list
    evaluation_rule<2>::iterator it = r2.begin();
    if (it == r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product list.");
    }
    const product_rule<2> &pr1 = r2.get_product(it);
    it++;
    if (it == r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Only one product.");
    }
    const product_rule<2> &pr2 = r2.get_product(it);
    it++;
    if (it != r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "More than two products.");
    }
    product_rule<2>::iterator ip1 = pr1.begin();
    product_rule<2>::iterator ip2 = pr2.begin();
    if (ip1 == pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product pr1.");
    }
    if (ip2 == pr2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product pr2.");
    }

    if ((pr1.get_intrinsic(ip1) != 2 || pr2.get_intrinsic(ip2) != 3) &&
            (pr1.get_intrinsic(ip1) != 3 || pr2.get_intrinsic(ip2) != 2))
        fail_test(testname, __FILE__, __LINE__, "Intrinsic label.");

    ip1++;
    if (ip1 != pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Too many terms in pr1.");
    }
    ip2++;
    if (ip2 != pr2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Too many terms in pr2.");
    }
}


/** \brief Reduction of four dimensions in two steps (non-complete dims)
 **/
void er_reduce_test::test_6(
        const std::string &id) throw(libtest::test_exception) {


    static const char *testname = "er_reduce_test::test_6()";

    typedef product_table_i::label_t label_t;
    typedef product_table_i::label_set_t label_set_t;
    typedef product_table_i::label_group_t label_group_t;

    evaluation_rule<6> r1;
    evaluation_rule<2> r2;

    try {

        sequence<6, size_t> seq1(0), seq2(0), seq3(1), seq4(1), seq5(0);
        seq1[2] = 1; seq2[4] = 1; seq3[2] = seq3[4] = 0;
        seq4[2] = seq4[3] = seq4[4] = 0; seq5[3] = 1;
        product_rule<6> &pr1a = r1.new_product();
        pr1a.add(seq1, 0);
        pr1a.add(seq2, 0);
        pr1a.add(seq3, 0);
        product_rule<6> &pr1b = r1.new_product();
        pr1b.add(seq1, 1);
        pr1b.add(seq2, 1);
        pr1b.add(seq3, 0);
        product_rule<6> &pr1c = r1.new_product();
        pr1c.add(seq1, 2);
        pr1c.add(seq2, 2);
        pr1c.add(seq3, 0);
        product_rule<6> &pr1d = r1.new_product();
        pr1d.add(seq1, 3);
        pr1d.add(seq2, 3);
        pr1d.add(seq3, 0);
        product_rule<6> &pr2a = r1.new_product();
        pr2a.add(seq1, 0);
        pr2a.add(seq2, 0);
        pr2a.add(seq4, 0);
        pr2a.add(seq5, 0);
        product_rule<6> &pr2b = r1.new_product();
        pr2b.add(seq1, 1);
        pr2b.add(seq2, 1);
        pr2b.add(seq4, 0);
        pr2b.add(seq5, 0);
        product_rule<6> &pr2c = r1.new_product();
        pr2c.add(seq1, 2);
        pr2c.add(seq2, 2);
        pr2c.add(seq4, 0);
        pr2c.add(seq5, 0);
        product_rule<6> &pr2d = r1.new_product();
        pr2d.add(seq1, 3);
        pr2d.add(seq2, 3);
        pr2d.add(seq4, 0);
        pr2d.add(seq5, 0);

        sequence<6, size_t> rmap(0);
        rmap[0] = 0; rmap[1] = 1; rmap[2] = 2; rmap[3] = 2; rmap[4] = rmap[5] = 3;
        sequence<4, label_group_t> rdims;
        rdims[0].push_back(0);
        rdims[0].push_back(2);
        rdims[0].push_back(3);
        rdims[1].push_back(0);
        rdims[1].push_back(1);
        rdims[1].push_back(2);
        rdims[1].push_back(3);

        evaluation_rule<2> tmp;
        er_reduce<6, 4>(r1, rmap, rdims, id).perform(tmp);
        er_optimize<2>(tmp, id).perform(r2);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

    // Check sequence list
    const eval_sequence_list<2> &sl = r2.get_sequences();
    if (sl.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# seq.");
    }
    if (sl[0][0] != 1 || sl[0][1] != 1) {
        fail_test(testname, __FILE__, __LINE__, "seq.");
    }

    // Check product list
    evaluation_rule<2>::iterator it = r2.begin();
    if (it == r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product list.");
    }

    const product_rule<2> &pr1 = r2.get_product(it);
    it++;
    if (it != r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "More than one product.");
    }
    product_rule<2>::iterator ip1 = pr1.begin();
    if (ip1 == pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product pr1.");
    }
    if (pr1.get_intrinsic(ip1) != 0) {
        fail_test(testname, __FILE__, __LINE__, "Intrinsic label.");
    }
}


/** \brief Reduction of two dimensions in one steps
 **/
void er_reduce_test::test_7(
        const std::string &id) throw(libtest::test_exception) {


    static const char *testname = "er_reduce_test::test_7()";

    typedef product_table_i::label_t label_t;
    typedef product_table_i::label_set_t label_set_t;
    typedef product_table_i::label_group_t label_group_t;

    evaluation_rule<6> r1, r3;
    evaluation_rule<4> r2, r4;

    //    Rmap: [ 0 1 2 3 4 4]
    //    Table ID: C2v
    //    Block labels:  [0(0): * *] [1(0): * *] [2(0): * *]
    //      [3(1): 0 1 2 3 0 1 2 3] [4(0): * *] [5(0): * *]
    //    Rule:
    //    ([000101], 0) ([111010], 0)

    try {

        sequence<6, size_t> seq1a(0), seq1b(0), seq2(0);
        seq1a[3] = 1; seq1a[5] = 1;
        seq1b[0] = 1; seq1b[1] = 1; seq1b[2] = 1; seq1b[4] = 1;
        seq2[3] = 1;

        product_rule<6> &pr1 = r1.new_product();
        pr1.add(seq1a, product_table_i::k_identity);
        pr1.add(seq1b, product_table_i::k_identity);

        sequence<6, size_t> rmap(0);
        rmap[0] = 0; rmap[1] = 1; rmap[2] = 2; rmap[3] = 3;
        rmap[4] = 4; rmap[5] = 4;
        sequence<2, label_group_t> rdims;
        rdims[0].push_back(0);
        rdims[0].push_back(1);
        rdims[0].push_back(2);
        rdims[0].push_back(3);

        er_reduce<6, 2>(r1, rmap, rdims, id).perform(r2);
        er_reduce<6, 2>(r3, rmap, rdims, id).perform(r4);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

    // Check sequence list
    const eval_sequence_list<4> &sl = r2.get_sequences();
    if (sl.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# seq.");
    }
    if (sl[0][0] != 1 || sl[0][1] != 1 || sl[0][2] != 1 || sl[0][3] != 1) {
        fail_test(testname, __FILE__, __LINE__, "seq.");
    }

    // Check product list
    evaluation_rule<4>::iterator it = r2.begin();
    if (it == r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product list.");
    }

    const product_rule<4> &pr1 = r2.get_product(it);
    it++;
    if (it != r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "More than one product.");
    }
    product_rule<2>::iterator ip1 = pr1.begin();
    if (ip1 == pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product pr1.");
    }
    if (pr1.get_intrinsic(ip1) != 0) {
        fail_test(testname, __FILE__, __LINE__, "Intrinsic label.");
    }
}


} // namespace libtensor
