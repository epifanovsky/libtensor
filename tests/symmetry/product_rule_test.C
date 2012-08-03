#include <libtensor/symmetry/product_rule.h>
#include "product_rule_test.h"

namespace libtensor {


void product_rule_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
}


/** \test Add terms to product rule
 **/
void product_rule_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "product_rule_test::test_1()";

    typedef product_table_i::label_set_t label_set_t;

    try {

    eval_sequence_list<3> sl;

    product_rule<3> pr(&sl);

    if (! pr.empty()) {
        fail_test(testname, __FILE__, __LINE__, "! pr.empty()");
    }

    sequence<3, size_t> seq1(1), seq2(1);
    seq2[1] = 0;

    pr.add(seq1, 0);
    pr.add(seq2, 1);

    if (pr.empty()) {
        fail_test(testname, __FILE__, __LINE__, "pr.empty()");
    }

    if (! sl.has_sequence(seq1)) {
        fail_test(testname, __FILE__, __LINE__, "seq1");
    }
    if (! sl.has_sequence(seq2)) {
        fail_test(testname, __FILE__, __LINE__, "seq1");
    }

    product_rule<3>::iterator it = pr.begin();
    if (it == pr.end()) {
        fail_test(testname, __FILE__, __LINE__, "it (0)");
    }
    if (pr.get_intrinsic(it) != 0) {
        fail_test(testname, __FILE__, __LINE__, "intrinsic (0)");
    }
    it++;
    if (it == pr.end()) {
        fail_test(testname, __FILE__, __LINE__, "it (1)");
    }
    if (pr.get_intrinsic(it) != 0) {
        fail_test(testname, __FILE__, __LINE__, "intrinsic (1)");
    }
    it++;
    if (it != pr.end()) {
        fail_test(testname, __FILE__, __LINE__, "it (2)");
    }


    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Compare to product rules
 **/
void product_rule_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "product_rule_test::test_2()";

    typedef product_table_i::label_set_t label_set_t;

    try {

    eval_sequence_list<3> sl1, sl2;

    product_rule<3> pr1a(&sl1), pr1b(&sl1), pr2(&sl2);

    sequence<3, size_t> seq1(1), seq2(1);
    seq2[1] = 0;

    pr1a.add(seq1, 0);
    pr1a.add(seq2, 2);
    pr1b.add(seq2, 2);
    pr1b.add(seq1, 0);
    pr2.add(seq1, 0);
    pr2.add(seq2, 2);

    if (pr1a != pr1b) {
        fail_test(testname, __FILE__, __LINE__, "pr1a != pr1b");
    }
    if (pr1a == pr2) {
        fail_test(testname, __FILE__, __LINE__, "pr1a == pr2");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
