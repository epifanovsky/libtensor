#include <libtensor/core/sequence_generator.h>
#include "sequence_generator_test.h"
#include <list>
#include <iostream>


namespace libtensor {


void sequence_generator_test::perform() throw(libtest::test_exception) {

    test_1();
}


/** \test Add sequences to the list of sequences
 **/
void sequence_generator_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "sequence_generator_test::test_1()";

    try {

    sequence_generator gen(3, 7);
    if (gen.get_seq().size() != 3) {
        fail_test(testname, __FILE__, __LINE__, "seq.size() != 3");
    }

    std::list< std::vector<size_t> > lst;
    do {
        const std::vector<size_t> &seq = gen.get_seq();
        for (size_t i = 1; i < seq.size(); i++) {
            if (seq[i] <= seq[i - 1]) {
                fail_test(testname, __FILE__, __LINE__,
                        "seq[i] <= seq[i - 1]");
            }
        }

        for (std::list< std::vector<size_t> >::const_iterator it =
                lst.begin(); it != lst.end(); it++) {

            const std::vector<size_t> &seq2 = *it;
            size_t i = 0;
            for (; i != seq2.size(); i++) {
                if (seq[i] != seq2[i]) break;
            }
            if (i == seq2.size()) {
                fail_test(testname, __FILE__, __LINE__, "seq == seq2");
            }
        }
        lst.push_back(seq);

    } while (gen.next());

    if (lst.size() != 35) {
        fail_test(testname, __FILE__, __LINE__, "# sequences.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
