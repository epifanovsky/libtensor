#include <libtensor/symmetry/er_optimize.h>
#include "er_optimize_test.h"

namespace libtensor {


void er_optimize_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
    test_3();
}


/** \test Optimization: remove all allowed terms from product rules
 **/
void er_optimize_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "er_optimize_test::test_1()";

    typedef product_table_i::label_set_t label_set_t;

    try {

    evaluation_rule<2> from, to;
    {
        product_rule<2> &pr = from.new_product();
        sequence<2, size_t> seq1(1), seq2(1);
        seq2[0] = 0;
        pr.add(seq1, product_table_i::k_identity);
        pr.add(seq2, product_table_i::k_invalid);
    }

    er_optimize<2>(from).perform(to);

    // Check sequence list
    const eval_sequence_list<2> &sl = to.get_sequences();
    if (sl.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# seq not optimized.");
    }
    if (sl[0][0] != sl[0][1] || sl[0][0] != 1) {
        fail_test(testname, __FILE__, __LINE__, "Wrong sequence.");
    }

    // Check product list
    evaluation_rule<2>::const_iterator it = to.begin();
    if (it == to.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product list.");
    }
    const product_rule<2> &px = to.get_product(it);
    it++;
    if (it != to.end()) {
        fail_test(testname, __FILE__, __LINE__, "Multiple products.");
    }

    product_rule<2>::iterator ip = px.begin();
    if (ip == px.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product.");
    }
    if (px.get_intrinsic(ip) != 0) {
        fail_test(testname, __FILE__, __LINE__, "Wrong intrinsic label");
    }
    ip++;
    if (ip != px.end()) {
        fail_test(testname, __FILE__, __LINE__, "Multiple terms.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Optimization: Rule simplification due to a sole all-allowed term
 **/
void er_optimize_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "er_optimize_test::test_2()";

    typedef product_table_i::label_set_t label_set_t;

    try {

    evaluation_rule<2> from, to;
    {
        sequence<2, size_t> seq1(1), seq2(1);
        seq1[0] = 0; seq2[1] = 0;
        product_rule<2> &pr1 = from.new_product();
        pr1.add(seq1, product_table_i::k_identity);
        product_rule<2> &pr2 = from.new_product();
        pr2.add(seq2, product_table_i::k_invalid);
    }

    er_optimize<2>(from).perform(to);

    // Check sequence list
    const eval_sequence_list<2> &sl = to.get_sequences();
    if (sl.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# seq not optimized.");
    }
    if (sl[0][0] != sl[0][1] || sl[0][0] != 1) {
        fail_test(testname, __FILE__, __LINE__, "Wrong sequence.");
    }

    // Check product list
    evaluation_rule<2>::const_iterator it = to.begin();
    if (it == to.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product list.");
    }
    const product_rule<2> &px = to.get_product(it);
    it++;
    if (it != to.end()) {
        fail_test(testname, __FILE__, __LINE__, "Multiple products.");
    }

    product_rule<2>::iterator ip = px.begin();
    if (ip == px.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product.");
    }
    if (px.get_intrinsic(ip) != product_table_i::k_invalid) {
        fail_test(testname, __FILE__, __LINE__, "Wrong intrinsic label");
    }
    ip++;
    if (ip != px.end()) {
        fail_test(testname, __FILE__, __LINE__, "Multiple terms.");
    }


    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Optimization: remove products with all forbidden rules
 **/
void er_optimize_test::test_3() throw(libtest::test_exception) {

    static const char *testname = "er_optimize_test::test_3()";

    typedef product_table_i::label_set_t label_set_t;

    try {

    evaluation_rule<2> from, to;
    {
        sequence<2, size_t> seq1(1), seq2(1), seq3(0);
        seq1[0] = 0; seq2[1] = 0;
        product_rule<2> &pr1 = from.new_product();
        pr1.add(seq1, product_table_i::k_identity);
        pr1.add(seq3, 1);
        product_rule<2> &pr2 = from.new_product();
        pr2.add(seq2, 2);
    }

    er_optimize<2>(from).perform(to);

    // Check sequence list
    const eval_sequence_list<2> &sl = to.get_sequences();
    if (sl.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# seq not optimized.");
    }
    if (sl[0][0] != 1 || sl[0][1] != 0) {
        fail_test(testname, __FILE__, __LINE__, "Wrong sequence.");
    }

    // Check product list
    evaluation_rule<2>::const_iterator it = to.begin();
    if (it == to.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product list.");
    }
    const product_rule<2> &px = to.get_product(it);
    it++;
    if (it != to.end()) {
        fail_test(testname, __FILE__, __LINE__, "Multiple products.");
    }

    product_rule<2>::iterator ip = px.begin();
    if (ip == px.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product.");
    }
    if (px.get_intrinsic(ip) != 2) {
        fail_test(testname, __FILE__, __LINE__, "Wrong intrinsic label");
    }
    ip++;
    if (ip != px.end()) {
        fail_test(testname, __FILE__, __LINE__, "Multiple terms.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
