#include <list>
#include <libtensor/core/sequence_generator.h>
#include "../test_utils.h"

using namespace libtensor;


/** \test Add sequences to the list of sequences
 **/
int test_1() {

    static const char testname[] = "sequence_generator_test::test_1()";

    try {

    sequence_generator gen(3, 7);
    if (gen.get_seq().size() != 3) {
        return fail_test(testname, __FILE__, __LINE__, "seq.size() != 3");
    }

    std::list< std::vector<size_t> > lst;
    do {
        const std::vector<size_t> &seq = gen.get_seq();
        for (size_t i = 1; i < seq.size(); i++) {
            if (seq[i] <= seq[i - 1]) {
                return fail_test(testname, __FILE__, __LINE__,
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
                return fail_test(testname, __FILE__, __LINE__, "seq == seq2");
            }
        }
        lst.push_back(seq);

    } while (gen.next());

    if (lst.size() != 35) {
        return fail_test(testname, __FILE__, __LINE__, "# sequences.");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int main() {

    return test_1();
}

