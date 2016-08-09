#include <libtensor/core/mask.h>
#include "../test_utils.h"

using namespace libtensor;


/** \test Tests the unary operator OR
 **/
int test_op_1() {

    static const char testname[] = "mask_test::test_op_1()";

    try {

    mask<4> m0, m1, m2;

    m1 |= m2;
    if(!m1.equals(m0)) {
        return fail_test(testname, __FILE__, __LINE__,
            "!m1.equals(m0) (0000)");
    }

    m2[0] = true; m0[0] = true;
    m1 |= m2;
    if(!m1.equals(m0)) {
        return fail_test(testname, __FILE__, __LINE__,
            "!m1.equals(m0) (1000)");
    }

    m2[2] = true; m0[2] = true;
    m2[3] = true; m0[3] = true;
    m1 |= m2;
    if(!m1.equals(m0)) {
        return fail_test(testname, __FILE__, __LINE__,
            "!m1.equals(m0) (1011)");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


/** \test Tests the binary operator OR
 **/
int test_op_2() {

    static const char testname[] = "mask_test::test_op_2()";

    try {

    mask<4> m0, m1, m2, m3;

    m1 = m2 | m3;
    if(!m1.equals(m0)) {
        return fail_test(testname, __FILE__, __LINE__,
            "!m1.equals(m0) (0000)");
    }

    m2[0] = true; m0[0] = true;
    m1 = m2 | m3;
    if(!m1.equals(m0)) {
        return fail_test(testname, __FILE__, __LINE__,
            "!m1.equals(m0) (1000)");
    }

    m2[0] = false; m2[1] = true; m2[3] = true; m3[3] = true;
    m0[0] = false; m0[1] = true; m0[3] = true;
    m1 = m2 | m3;
    if(!m1.equals(m0)) {
        return fail_test(testname, __FILE__, __LINE__,
            "!m1.equals(m0) (0101)");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


/** \test Tests the unary operator AND
 **/
int test_op_3() {

    static const char testname[] = "mask_test::test_op_3()";

    try {

    mask<4> m0, m1, m2;

    m1 &= m2;
    if(!m1.equals(m0)) {
        return fail_test(testname, __FILE__, __LINE__,
            "!m1.equals(m0) (0000)");
    }

    m1[0] = true; m2[0] = true; m2[1] = true; m0[0] = true;
    m1 &= m2;
    if(!m1.equals(m0)) {
        return fail_test(testname, __FILE__, __LINE__,
            "!m1.equals(m0) (1000)");
    }

    m1[2] = true;
    m2[2] = true; m2[3] = true;
    m0[2] = true;
    m1 &= m2;
    if(!m1.equals(m0)) {
        return fail_test(testname, __FILE__, __LINE__,
            "!m1.equals(m0) (1010)");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


/** \test Tests the binary operator AND
 **/
int test_op_4() {

    static const char testname[] = "mask_test::test_op_4()";

    try {

    mask<4> m0, m1, m2, m3;

    m1 = m2 & m3;
    if(!m1.equals(m0)) {
        return fail_test(testname, __FILE__, __LINE__,
            "!m1.equals(m0) (0000)");
    }

    m2[0] = true; m2[1] = true;
    m3[0] = true; m3[3] = true;
    m0[0] = true;
    m1 = m2 & m3;
    if(!m1.equals(m0)) {
        return fail_test(testname, __FILE__, __LINE__,
            "!m1.equals(m0) (1000)");
    }

    m2[0] = false; m2[1] = true; m2[3] = true;
    m3[0] = false; m3[1] = true; m3[3] = true;
    m0[0] = false; m0[1] = true; m0[3] = true;
    m1 = m2 & m3;
    if(!m1.equals(m0)) {
        return fail_test(testname, __FILE__, __LINE__,
            "!m1.equals(m0) (0101)");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int main() {

    mask<2> msk1;
    mask<2> msk2(msk1);

    return

    test_op_1() |
    test_op_2() |
    test_op_3() |
    test_op_4() |

    0;
}

