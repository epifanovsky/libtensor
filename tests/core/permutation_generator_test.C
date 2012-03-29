#include <libtensor/core/permutation_generator.h>
#include "permutation_generator_test.h"

namespace libtensor {


void permutation_generator_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
    test_3();

}


/**	\test Tests the generation of permutations
 **/
void permutation_generator_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "permutation_generator_test::test_1()";

    permutation_generator<4> pg;
    std::vector< permutation<4> > res;

    try {
        do {
            res.push_back(pg.get_perm());
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

    mask<6> msk;
    msk[1] = msk[3] = msk[4] = true;

    permutation_generator<6> pg(msk);
    std::vector< permutation<6> > res;

    try {

        do {
            res.push_back(pg.get_perm());
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

/** \test Tests the generation of permutations
 **/
void permutation_generator_test::test_3() throw(libtest::test_exception) {

    static const char *testname = "permutation_generator_test::test_3()";

    permutation_generator<4> pg;

    try {

        permutation<4> p0;
        while (pg.next()) {
            const permutation<4> &p1 = pg.get_perm();
            size_t i = 0;
            for (; i < 4 && p1[i] == p0[i]; i++) ;
            if (i == 4) fail_test(testname, __FILE__, __LINE__, "p0 == p1");
            i++;
            for (; i < 4 && p1[i] == p0[i]; i++) ;
            if (i == 4) fail_test(testname, __FILE__, __LINE__, "p1 invalid");
            i++;
            for (; i < 4 && p1[i] == p0[i]; i++) ;
            if (i != 4) fail_test(testname, __FILE__, __LINE__,
                    "p0^-1 p1 not pair permutation");

            p0.reset();
            p0.permute(p1);
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

} // namespace libtensor
