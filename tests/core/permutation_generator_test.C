#include <libtensor/core/permutation_generator.h>
#include "permutation_generator_test.h"

namespace libtensor {


void permutation_generator_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();

}


/**	\test Tests the generation of permutations of a whole sequence
 **/
void permutation_generator_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "permutation_generator_test::test_1()";

    sequence<4, double> seq;
    mask<4> msk;
    for (size_t i = 0; i < 4; i++) { seq[i] = (double) i; msk[i] = true; }

    permutation_generator<4, double> pg(seq, msk);
    std::vector< sequence<4, double> > res;

    try {

        do {
            res.push_back(pg.get_sequence());
        } while (pg.next());

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

    if (res.size() != 24)
        fail_test(testname, __FILE__, __LINE__,
                "Wrong number of permutations.");

    for (size_t i = 0; i < res.size(); i++) {
        for (size_t j = i + 1; j < res.size(); j++) {
            size_t k = 0;
            for (; k < 4; k++) {
                if (res[i][k] != res[j][k]) break;
            }
            if (k == 4) {
                fail_test(testname, __FILE__, __LINE__,
                        "Identical permutations.");
            }
        }
    }
}

/** \test Tests the generation of permutations of a part of a sequence
 **/
void permutation_generator_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "permutation_generator_test::test_2()";

    sequence<6, double> seq;
    mask<6> msk;
    for (size_t i = 0; i < 6; i++) { seq[i] = 1. / (double) i; }
    msk[1] = msk[3] = msk[4] = true;

    permutation_generator<6, double> pg(seq, msk);
    std::vector< sequence<6, double> > res;

    try {

        do {
            res.push_back(pg.get_sequence());
        } while (pg.next());

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

    if (res.size() != 6)
        fail_test(testname, __FILE__, __LINE__,
                "Wrong number of permutations.");

    for (size_t i = 0; i < res.size(); i++) {
        for (size_t j = i + 1; j < res.size(); j++) {
            size_t k = 0;
            for (; k < 6; k++) {
                if (res[i][k] != res[j][k]) break;
            }
            if (k == 6) {
                fail_test(testname, __FILE__, __LINE__,
                        "Identical permutations.");
            }
        }
    }
}


} // namespace libtensor
