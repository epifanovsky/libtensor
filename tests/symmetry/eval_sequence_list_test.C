#include <libtensor/symmetry/eval_sequence_list.h>
#include <libtensor/symmetry/product_table_i.h>
#include "eval_sequence_list_test.h"

namespace libtensor {


void eval_sequence_list_test::perform() throw(libtest::test_exception) {

    test_1();
}


/** \test Add sequences to the list of sequences + try accessing them.
 **/
void eval_sequence_list_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "eval_sequence_list_test::test_1()";

    typedef product_table_i::label_set_t label_set_t;

    try {

        eval_sequence_list<3> sl;

        sequence<3, size_t> s1(1), s2(1), s3(1);
        s3[2] = 2;

        size_t id1, id2, id3;
        id1 = sl.add(s1);
        id2 = sl.add(s2);
        id3 = sl.add(s3);

        if (id1 != id2) {
            fail_test(testname, __FILE__, __LINE__,
                    "Two different IDs for identical sequences.");
        }
        if (sl.size() != 2) {
            fail_test(testname, __FILE__, __LINE__,
                    "Wrong number of sequences.");
        }
        if (! sl.has_sequence(s1)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Sequences s1 not found.");
        }
        if (! sl.has_sequence(s3)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Sequence s2 not found.");
        }
        if (sl.get_position(s1) != id1) {
            fail_test(testname, __FILE__, __LINE__,
                    "Wrong position of s1.");
        }
        if (sl.get_position(s3) != id3) {
            fail_test(testname, __FILE__, __LINE__,
                    "Wrong position of s2.");
        }
        const sequence<3, size_t> &s1_ref = sl[id1];
        for (size_t i = 0; i < 3; i++) {
            if (s1[i] != s1_ref[i])
                fail_test(testname, __FILE__, __LINE__, "s1 != s1_ref.");
        }
        const sequence<3, size_t> &s3_ref = sl[id3];
        for (size_t i = 0; i < 3; i++) {
            if (s3[i] != s3_ref[i])
                fail_test(testname, __FILE__, __LINE__, "s3 != s3_ref.");
        }
    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
