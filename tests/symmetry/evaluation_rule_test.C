#include <libtensor/symmetry/evaluation_rule.h>
#include "evaluation_rule_test.h"

namespace libtensor {


void evaluation_rule_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
    test_opt_1();
    test_opt_2();
}


/** \test Add sequences to the list of sequences
 **/
void evaluation_rule_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "evaluation_rule_test::test_1()";

    typedef product_table_i::label_set_t label_set_t;

    try {

        evaluation_rule<3> rules;

        sequence<3, size_t> s1(1), s2(1), s3(1);
        s3[2] = 2;

        size_t id1, id2, id3;
        id1 = rules.add_sequence(s1);
        id2 = rules.add_sequence(s2);
        id3 = rules.add_sequence(s3);

        if (id1 != id2)
            fail_test(testname, __FILE__, __LINE__,
                    "Two different IDs for identical sequences.");

        if (rules.get_n_sequences() != 2)
            fail_test(testname, __FILE__, __LINE__,
                    "Wrong number of sequences.");

        const sequence<3, size_t> &s1_ref = rules[id1];
        for (size_t i = 0; i < 3; i++) {
            if (s1[i] != s1_ref[i])
                fail_test(testname, __FILE__, __LINE__, "s1 != s1_ref.");
        }

        const sequence<3, size_t> &s3_ref = rules[id3];
        for (size_t i = 0; i < 3; i++) {
            if (s3[i] != s3_ref[i])
                fail_test(testname, __FILE__, __LINE__, "s3 != s3_ref.");
        }
    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Tests add sequences + create list of lists
 **/
void evaluation_rule_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "evaluation_rule_test::test_2()";

    typedef product_table_i::label_set_t label_set_t;

    try {

        evaluation_rule<3> rules;
        sequence<3, size_t> s1;
        s1[0] = s1[1] = s1[2] = 1;

        size_t sno = rules.add_sequence(s1);

        size_t pno1 = rules.add_product(sno, 0, 0);
        rules.add_to_product(pno1, sno, 1, 2);

        size_t pno2 = rules.add_product(sno, 0, 1);
        size_t pno3 = rules.add_product(sno, 1, 0);

        if (rules.get_n_products() != 3)
            fail_test(testname, __FILE__, __LINE__, "Unexpected # products.");

        evaluation_rule<3>::iterator it = rules.begin(pno1);
        if (rules.get_seq_no(it) != sno)
            fail_test(testname, __FILE__, __LINE__, "Unknown sequence.");
        if (rules.get_intrinsic(it) != 0)
            fail_test(testname, __FILE__, __LINE__, "Wrong intrinsic label.");
        if (rules.get_target(it) != 0)
            fail_test(testname, __FILE__, __LINE__, "Wrong target.");
        it++;
        if (it == rules.end(pno1))
            fail_test(testname, __FILE__, __LINE__,
                    "Term missing in product");
        if (rules.get_seq_no(it) != sno)
            fail_test(testname, __FILE__, __LINE__, "Unknown sequence.");
        if (rules.get_intrinsic(it) != 1)
            fail_test(testname, __FILE__, __LINE__, "Wrong intrinsic label.");
        if (rules.get_target(it) != 2)
            fail_test(testname, __FILE__, __LINE__, "Wrong target.");
        it++;
        if (it != rules.end(pno1))
            fail_test(testname, __FILE__, __LINE__,
                    "Two many triples in product.");

        it = rules.begin(pno2);
        if (rules.get_seq_no(it) != sno)
            fail_test(testname, __FILE__, __LINE__, "Unknown sequence.");
        if (rules.get_intrinsic(it) != 0)
            fail_test(testname, __FILE__, __LINE__, "Wrong intrinsic label.");
        if (rules.get_target(it) != 1)
            fail_test(testname, __FILE__, __LINE__, "Wrong target.");
        it++;
        if (it != rules.end(pno2))
            fail_test(testname, __FILE__, __LINE__,
                    "Two many pairs in product.");

        it = rules.begin(pno3);
        if (rules.get_seq_no(it) != sno)
            fail_test(testname, __FILE__, __LINE__, "Unknown sequence.");
        if (rules.get_intrinsic(it) != 1)
            fail_test(testname, __FILE__, __LINE__, "Wrong intrinsic label.");
        if (rules.get_target(it) != 0)
            fail_test(testname, __FILE__, __LINE__, "Wrong target.");
        it++;
        if (it != rules.end(pno3))
            fail_test(testname, __FILE__, __LINE__,
                    "Two many pairs in product.");

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Tests optimization of rules: all forbidden and all allowed rules
 **/
void evaluation_rule_test::test_opt_1() throw(libtest::test_exception) {

    static const char *testname = "evaluation_rule_test::test_opt_1()";

    typedef product_table_i::label_set_t label_set_t;

    try {

        evaluation_rule<3> r1, r2, r3;
        sequence<3, size_t> s1(0), s2(1);

        size_t id1a = r1.add_sequence(s1);
        size_t id1b = r1.add_sequence(s2);
        r1.add_product(id1a, 0, 0);
        r1.add_to_product(0, id1b, 0, 0);
        r1.add_product(id1b, 0, product_table_i::k_invalid);

        r1.optimize();

        size_t id2 = r2.add_sequence(s2);
        r2.add_product(id2, product_table_i::k_invalid, 0);
        r2.add_product(id2, 1, 0);
        r2.add_to_product(1, id2, 2, 0);

        r2.optimize();

        if (r1.get_n_products() != 0 || r1.get_n_sequences() != 0)
            fail_test(testname, __FILE__, __LINE__, "Empty rule expected.");
        if (r2.get_n_sequences() != 1)
            fail_test(testname, __FILE__, __LINE__,
                    "Only one sequence expected.");
        if (r2.get_n_products() != 1)
            fail_test(testname, __FILE__, __LINE__,
                    "One single product expected.");
        evaluation_rule<3>::iterator it = r2.begin(0);
        if (r2.get_intrinsic(it) != product_table_i::k_invalid)
            fail_test(testname, __FILE__, __LINE__,
                    "All-allowed term expected.");
        it++;
        if (it != r2.end(0))
            fail_test(testname, __FILE__, __LINE__,
                    "Only one term expected in product");

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Tests optimization of rules: unused sequences
 **/
void evaluation_rule_test::test_opt_2() throw(libtest::test_exception) {

    static const char *testname = "evaluation_rule_test::test_opt_2()";

    typedef product_table_i::label_set_t label_set_t;

    try {

        evaluation_rule<3> r1;
        sequence<3, size_t> s1(0), s2(0), s3(0);
        s1[0] = 1; s2[1] = s2[2] = 1; s3[2] = 1;

        size_t id1a = r1.add_sequence(s1);
        size_t id1b = r1.add_sequence(s2);
        size_t id1c = r1.add_sequence(s3);

        r1.add_product(id1a, 0, 0);
        r1.add_to_product(0, id1b, 0, 0);
        r1.add_product(id1b, 1, 1);

        r1.optimize();

        if (r1.get_n_sequences() != 2)
            fail_test(testname, __FILE__, __LINE__,
                    "Only two sequences expected.");
        if (r1.get_n_products() != 2)
            fail_test(testname, __FILE__, __LINE__,
                    "Two products expected.");
        evaluation_rule<3>::iterator it = r1.begin(0);
        it++; it++;
        if (it != r1.end(0))
            fail_test(testname, __FILE__, __LINE__,
                    "Two terms expected in product");
        it = r1.begin(1);
        it++;
        if (it != r1.end(1))
            fail_test(testname, __FILE__, __LINE__,
                    "One term expected in product");

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
