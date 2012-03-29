#include <libtensor/core/permutation_generator.h>
#include "permutation_generator_test.h"

namespace libtensor {


void permutation_generator_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();

}


/**	\test Tests the generation of permutations
 **/
void permutation_generator_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "permutation_generator_test::test_1()";

    sequence<4, double> seq, pseq;
    for (size_t i = 0; i < 4; i++) { seq[i] = (double) i; }

    permutation_generator pg(4);
    std::vector< sequence<4, double> > res;

    try {

        do {
            for (size_t i = 0; i < 4; i++) { pseq[i] = seq[pg[i]]; }
            res.push_back(pseq);
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

    sequence<6, double> seq, pseq;
    for (size_t i = 0; i < 6; i++) { pseq[i] = seq[i] = 1. / (double) i; }
    std::vector<size_t> map(3);
    map[0] = 1; map[1] = 3; map[2] = 4;

    permutation_generator pg(3);
    std::vector< sequence<6, double> > res;

    try {

        do {
            for (size_t i = 0; i < 3; i++) { pseq[map[i]] = seq[map[pg[i]]]; }
            res.push_back(pseq);
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
