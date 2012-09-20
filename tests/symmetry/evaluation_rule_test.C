#include <libtensor/symmetry/evaluation_rule.h>
#include <libtensor/symmetry/point_group_table.h>
#include "evaluation_rule_test.h"

namespace libtensor {


void evaluation_rule_test::perform() throw(libtest::test_exception) {

    test_1();
    test_copy_1();
}


/** \test Create product(s), traverse them, clear them
 **/
void evaluation_rule_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "evaluation_rule_test::test_1()";

    typedef product_table_i::label_set_t label_set_t;

    try {

    evaluation_rule<3> rule;
    product_rule<3> &pr1 = rule.new_product();
    product_rule<3> &pr2 = rule.new_product();
    sequence<3, size_t> seq1(1), seq2(1);
    seq2[0] = 0;

    pr1.add(seq1, 0);
    pr2.add(seq1, 1);
    pr1.add(seq2, 1);

    // Check sequences in rule
    const eval_sequence_list<3> &sl = rule.get_sequences();
    if (sl.size() != 2) {
        fail_test(testname, __FILE__, __LINE__, "# seq.");
    }
    for (size_t i = 0; i < 3; i++) {
        if (seq1[i] != sl[0][i]) {
            fail_test(testname, __FILE__, __LINE__, "sl[0]");
        }
        if (seq2[i] != sl[1][i]) {
            fail_test(testname, __FILE__, __LINE__, "sl[1]");
        }
    }

    // Check products
    evaluation_rule<3>::iterator it = rule.begin();
    if (rule.get_product(it) != pr1) {
        fail_test(testname, __FILE__, __LINE__, "1st product rule.");
    }
    it++;
    if (rule.get_product(it) != pr2) {
        fail_test(testname, __FILE__, __LINE__, "2nd product rule.");
    }

    rule.clear();
    if (rule.begin() != rule.end()) {
        fail_test(testname, __FILE__, __LINE__, "Product rules not cleared.");
    }
    if (sl.size() != 0) {
        fail_test(testname, __FILE__, __LINE__,
                "Evaluation sequences not cleared.");
    }


    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Copy constructor, operator=
 **/
void evaluation_rule_test::test_copy_1() throw(libtest::test_exception) {

    static const char *testname = "evaluation_rule_test::test_copy_1()";

    typedef product_table_i::label_set_t label_set_t;

    try {

    evaluation_rule<3> rule1, rule2, rule3;
    product_rule<3> &pr1 = rule1.new_product();
    product_rule<3> &pr2 = rule1.new_product();
    sequence<3, size_t> seq1(1), seq2(1);
    seq2[0] = 0;

    pr1.add(seq1, 0);
    pr2.add(seq1, 1);
    pr1.add(seq2, 1);

    rule3 = rule2 = rule1;

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
