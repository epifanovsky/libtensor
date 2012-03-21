#include <libtensor/symmetry/label/evaluation_rule.h>
#include "evaluation_rule_test.h"

namespace libtensor {

void evaluation_rule_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();

}

/** \test Add sequences to the list of sequences
 **/
void evaluation_rule_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "evaluation_rule_test::test_1()";

    typedef product_table_i::label_set_t label_set_t;

    try {

        evaluation_rule<3> rules;

        sequence<3, size_t> s1, s2, s3;
        s1[0] = s1[1] = s1[2] = 1;
        s2[0] = s2[1] = s2[2] = 1;
        s3[0] = s3[1] = 1; s3[2] = 2;

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

        size_t pno1 = rules.add_product(sno, 0);
        rules.add_to_product(pno1, sno, 1);

        size_t pno2 = rules.add_product(sno, 0);
        size_t pno3 = rules.add_product(sno, 1);

        if (rules.get_n_products() != 3)
            fail_test(testname, __FILE__, __LINE__, "Unexpected # products.");

        size_t nterms = 0;
        evaluation_rule<3>::iterator it = rules.begin(pno1);
        for(; it != rules.end(pno1); it++) {

            if (rules.get_seq_no(it) != sno)
                fail_test(testname, __FILE__, __LINE__, "Unknown sequence.");

            const sequence<3, size_t> &s1_ref = rules.get_sequence(it);
            for (size_t i = 0; i < 3; i++)
                if (s1_ref[i] != s1[i])
                fail_test(testname, __FILE__, __LINE__, "Unknown sequence.");

            nterms++;
        }
        if (nterms != 2)
            fail_test(testname, __FILE__, __LINE__, "Pairs missing in product");

        it = rules.begin(pno2);
        if (rules.get_seq_no(it) != sno)
            fail_test(testname, __FILE__, __LINE__, "Unknown sequence.");
        if (rules.get_target(it) != 0)
            fail_test(testname, __FILE__, __LINE__, "Wrong target.");
        it++;
        if (it != rules.end(pno2))
            fail_test(testname, __FILE__, __LINE__,
                    "Two many pairs in product.");

        it = rules.begin(pno3);
        if (rules.get_seq_no(it) != sno)
            fail_test(testname, __FILE__, __LINE__, "Unknown sequence.");
        if (rules.get_target(it) != 1)
            fail_test(testname, __FILE__, __LINE__, "Wrong target.");
        it++;
        if (it != rules.end(pno3))
            fail_test(testname, __FILE__, __LINE__,
                    "Two many pairs in product.");

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

} // namespace libtensor
