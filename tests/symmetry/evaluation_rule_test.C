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

    typedef evaluation_rule::label_set label_set;

    try {

        evaluation_rule rules;

        label_set i1, i2, i3;
        i1.insert(0); i2.insert(1);
        i3.insert(0); i3.insert(1);
        std::vector<size_t> o1(4, 0), o2(3, 0), o3(4, 0);
        o1[0] = 0; o1[1] = 1; o1[2] = 2; o1[3] = evaluation_rule::k_intrinsic;
        o2[0] = 1; o2[1] = 0; o2[2] = 2;
        o3[0] = 1; o1[1] = evaluation_rule::k_intrinsic; o1[2] = 0; o1[3] = 2;

        evaluation_rule::rule_id id1, id2, id3;
        id1 = rules.add_rule(i1, o1);
        id2 = rules.add_rule(i2, o2);
        id3 = rules.add_rule(i3, o3);

        bool done1 = false, done2 = false, done3 = false;
        for (evaluation_rule::rule_iterator it = rules.begin();
                it != rules.end(); it++) {

            evaluation_rule::rule_id cur_id = rules.get_rule_id(it);
            const evaluation_rule::basic_rule &cur = rules.get_rule(it);

            if (cur_id == id1) {
                if (done1)
                    fail_test(testname, __FILE__, __LINE__,
                            "Non-unique rule ID");

                if (cur.intr.size() != i1.size())
                    fail_test(testname, __FILE__, __LINE__,
                            "Unexpected intrinsic labels");

                label_set::const_iterator it = cur.intr.begin();
                label_set::const_iterator it1 = i1.begin();
                for (; it != cur.intr.end(); it++, it1++)
                    if ((*it) != (*it1))
                        fail_test(testname, __FILE__, __LINE__,
                                "Unexpected intrinsic labels");

                if (cur.order.size() != o1.size())
                    fail_test(testname, __FILE__, __LINE__,
                            "Unexpected evaluation order");

                for (size_t i = 0; i < cur.order.size(); i++)
                    if (cur.order[i] != o1[i])
                        fail_test(testname, __FILE__, __LINE__,
                                "Unexpected evaluation order");

                done1 = true;
            }
            else if (cur_id == id2) {
                if (done2)
                    fail_test(testname, __FILE__, __LINE__,
                            "Non-unique rule ID");

                if (cur.intr.size() != i2.size())
                    fail_test(testname, __FILE__, __LINE__,
                            "Unexpected intrinsic labels");

                label_set::const_iterator it = cur.intr.begin();
                label_set::const_iterator it2 = i2.begin();
                for (; it != cur.intr.end(); it++, it2++)
                    if ((*it) != (*it2))
                        fail_test(testname, __FILE__, __LINE__,
                                "Unexpected intrinsic labels");

                if (cur.order.size() != o2.size())
                    fail_test(testname, __FILE__, __LINE__,
                            "Unexpected evaluation order");

                for (size_t i = 0; i < cur.order.size(); i++)
                    if (cur.order[i] != o2[i])
                        fail_test(testname, __FILE__, __LINE__,
                                "Unexpected evaluation order");

                done2 = true;
            }
            else if (cur_id == id3) {
                if (done3)
                    fail_test(testname, __FILE__, __LINE__,
                            "Non-unique rule ID");

                if (cur.intr.size() != i3.size())
                    fail_test(testname, __FILE__, __LINE__,
                            "Unexpected intrinsic labels");

                label_set::const_iterator it = cur.intr.begin();
                label_set::const_iterator it3 = i3.begin();
                for (; it != cur.intr.end(); it++, it3++)
                    if ((*it) != (*it3))
                        fail_test(testname, __FILE__, __LINE__,
                                "Unexpected intrinsic labels");

                if (cur.order.size() != o3.size())
                    fail_test(testname, __FILE__, __LINE__,
                            "Unexpected evaluation order");

                for (size_t i = 0; i < cur.order.size(); i++)
                    if (cur.order[i] != o3[i])
                        fail_test(testname, __FILE__, __LINE__,
                                "Unexpected evaluation order");

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

    typedef evaluation_rule::label_set label_set;

    try {

        evaluation_rule rules;

        label_set i1, i2, i3;
        i1.insert(0); i2.insert(1); i3.insert(0); i3.insert(1);

        std::vector<size_t> o1(4, 0), o2(3, 0), o3(4, 0);
        o1[0] = 0; o1[1] = 1; o1[2] = 2; o1[3] = evaluation_rule::k_intrinsic;
        o2[0] = 1; o2[1] = 0; o2[2] = 2;
        o3[0] = 1; o1[1] = evaluation_rule::k_intrinsic; o1[2] = 0; o1[3] = 2;

        evaluation_rule::rule_id id1, id2, id3;
        id1 = rules.add_rule(i1, o1);
        id2 = rules.add_rule(i2, o2);
        id3 = rules.add_rule(i3, o3);

        size_t pno1 = rules.add_product(id1);
        rules.add_to_product(pno1, id2);

        size_t pno2 = rules.add_product(id3);
        rules.add_to_product(pno2, id2);

        if (rules.get_n_products() != 2)
            fail_test(testname, __FILE__, __LINE__, "Unexpected # products.");

        bool done1 = false, done2 = false;
        for (evaluation_rule::product_iterator it = rules.begin(pno1);
                it != rules.end(pno1); it++) {

            evaluation_rule::rule_id id = rules.get_rule_id(it);
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
        for (evaluation_rule::product_iterator it = rules.begin(pno2);
                it != rules.end(pno2); it++) {

            evaluation_rule::rule_id id = rules.get_rule_id(it);
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
