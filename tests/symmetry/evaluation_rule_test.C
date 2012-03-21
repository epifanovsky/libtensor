#include <libtensor/symmetry/label/evaluation_rule.h>
#include "evaluation_rule_test.h"

namespace libtensor {

void evaluation_rule_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();

}

/** \test Add rules to the list of rules
 **/
void evaluation_rule_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "evaluation_rule_test::test_1()";

    typedef evaluation_rule<3> eval_rule_t;
    typedef eval_rule_t::label_set_t label_set_t;
    typedef eval_rule_t::basic_rule_t basic_rule_t;

    try {

        eval_rule_t rules;

        basic_rule_t br1, br2, br3;
        br1[0] = br1[1] = br1[2] = 1;
        br1.set_target(0);
        br2[0] = br2[1] = br2[2] = 1;
        br2.set_target(1);
        br3[0] = br3[1] = br3[2] = 1;
        br3.set_target(0);
        br3.set_target(1);

        eval_rule_t::rule_id_t id1, id2, id3;
        id1 = rules.add_rule(br1);
        id2 = rules.add_rule(br2);
        id3 = rules.add_rule(br3);

        bool done1 = false, done2 = false, done3 = false;
        for (eval_rule_t::rule_iterator it = rules.begin();
                it != rules.end(); it++) {

            eval_rule_t::rule_id_t cur_id = rules.get_rule_id(it);
            const basic_rule_t &cur = rules.get_rule(it);

            if (cur_id == id1) {
                if (done1)
                    fail_test(testname, __FILE__, __LINE__,
                            "Non-unique rule ID");

                if (cur != br1)
                    fail_test(testname, __FILE__, __LINE__,
                            "Wrong basic rule.");

                done1 = true;
            }
            else if (cur_id == id2) {
                if (done2)
                    fail_test(testname, __FILE__, __LINE__,
                            "Non-unique rule ID");

                if (cur != br2)
                    fail_test(testname, __FILE__, __LINE__,
                            "Wrong basic rule.");

                done2 = true;
            }
            else if (cur_id == id3) {
                if (done3)
                    fail_test(testname, __FILE__, __LINE__,
                            "Non-unique rule ID");

                if (cur != br3)
                    fail_test(testname, __FILE__, __LINE__,
                            "Wrong basic rule.");

                done3 = true;
            }
            else {
                fail_test(testname, __FILE__, __LINE__,
                        "Unknown ID.");
            }

        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Tests add rules + create setup
 **/
void evaluation_rule_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "evaluation_rule_test::test_2()";

    typedef evaluation_rule<3> eval_rule_t;
    typedef eval_rule_t::label_set_t label_set_t;
    typedef eval_rule_t::basic_rule_t basic_rule_t;

    try {

        eval_rule_t rules;

        basic_rule_t br1, br2, br3;
        br1[0] = br1[1] = br1[2] = 1;
        br1.set_target(0);
        br2[0] = br2[1] = br2[2] = 1;
        br2.set_target(1);
        br3[0] = br3[1] = br3[2] = 1;
        br3.set_target(0);
        br3.set_target(1);

        eval_rule_t::rule_id_t id1, id2, id3;
        id1 = rules.add_rule(br1);
        id2 = rules.add_rule(br2);
        id3 = rules.add_rule(br3);

        size_t pno1 = rules.add_product(id1);
        rules.add_to_product(pno1, id2);

        size_t pno2 = rules.add_product(id3);
        rules.add_to_product(pno2, id2);

        if (rules.get_n_products() != 2)
            fail_test(testname, __FILE__, __LINE__, "Unexpected # products.");

        bool done1 = false, done2 = false;
        for (eval_rule_t::product_iterator it = rules.begin(pno1);
                it != rules.end(pno1); it++) {

            eval_rule_t::rule_id_t id = rules.get_rule_id(it);
            if (id == id1) {
                done1 = true;
            }
            else if (id == id2) {
                done2 = true;
            }
            else {
                fail_test(testname, __FILE__, __LINE__, "Unknown rule.");
            }
        }

        if (! (done1 && done2)) {
            fail_test(testname, __FILE__, __LINE__, "Rules missing in product");
        }

        done1 = false, done2 = false;
        for (eval_rule_t::product_iterator it = rules.begin(pno2);
                it != rules.end(pno2); it++) {

            eval_rule_t::rule_id_t id = rules.get_rule_id(it);
            if (id == id3) {
                done1 = true;
            }
            else if (id == id2) {
                done2 = true;
            }
            else {
                fail_test(testname, __FILE__, __LINE__, "Unknown rule.");
            }
        }

        if (! (done1 && done2)) {
            fail_test(testname, __FILE__, __LINE__, "Rules missing in product");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

} // namespace libtensor
