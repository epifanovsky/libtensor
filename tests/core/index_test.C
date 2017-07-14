#include <sstream>
#include <string>
#include <libtensor/core/index.h>
#include "../test_utils.h"

using namespace libtensor;


int test_ctor() {

    index<2> i1;
    return 0;
}


int test_less() {

    static const char testname[] = "index_test::test_less()";

    index<2> i1, i2;

    i1[0] = 1; i1[1] = 1;
    i2[0] = 2; i2[1] = 2;
    if(!i1.less(i2)) {
        return fail_test(testname, __FILE__, __LINE__,
            "less doesn't return (1,1)<(2,2)");
    }
    if(i2.less(i1)) {
        return fail_test(testname, __FILE__, __LINE__,
            "less returns (2,2)<(1,1)");
    }
    i1[0] = 2;
    if(!i1.less(i2)) {
        return fail_test(testname, __FILE__, __LINE__,
            "less doesn't return (2,1)<(2,2)");
    }
    if(i2.less(i1)) {
        return fail_test(testname, __FILE__, __LINE__,
            "less returns (2,2)<(2,1)");
    }
    i1[1] = 2;
    if(i1.less(i2)) {
        return fail_test(testname, __FILE__, __LINE__,
            "less returns (2,2)<(2,2)");
    }

    i1[0] = 0; i1[1] = 10;
    i2[0] = 10; i2[1] = 12;
    if(!i1.less(i2)) {
        return fail_test(testname, __FILE__, __LINE__,
            "less returns (10,12)<(0,10)");
    }
    i1[1] = 12;
    if(!i1.less(i2)) {
        return fail_test(testname, __FILE__, __LINE__,
            "less returns (10,12)<(0,12)");
    }

    return 0;
}


int test_print() {

    static const char testname[] = "index_test::test_print()";

    std::ostringstream ss1;
    index<1> i1;
    ss1 << i1;
    if(ss1.str().compare("[0]")!=0) {
        std::ostringstream err;
        err << "output error: expected \'[0]\', received \'";
        err << ss1.str() << "\'";
        return fail_test(testname, __FILE__, __LINE__, err.str().c_str());
    }

    std::ostringstream ss2;
    i1[0]=25;
    ss2 << i1;
    if(ss2.str().compare("[25]")!=0) {
        std::ostringstream err;
        err << "output error: expected \'[25]\', received \'";
        err << ss2.str() << "\'";
        return fail_test(testname, __FILE__, __LINE__, err.str().c_str());
    }

    std::ostringstream ss3;
    index<1> i1a; i1a[0]=3;
    ss3 << i1a << i1;
    if(ss3.str().compare("[3][25]")!=0) {
        std::ostringstream err;
        err << "output error: expected \'[3][25]\', received \'";
        err << ss3.str() << "\'";
        return fail_test(testname, __FILE__, __LINE__, err.str().c_str());
    }

    std::ostringstream ss4;
    index<2> i2;
    ss4 << i2;
    if(ss4.str().compare("[0,0]")!=0) {
        std::ostringstream err;
        err << "output error: expected \'[0,0]\', received \'";
        err << ss4.str() << "\'";
        return fail_test(testname, __FILE__, __LINE__, err.str().c_str());
    }

    std::ostringstream ss5;
    i2[0]=3; i2[1]=4;
    ss5 << i2;
    if(ss5.str().compare("[3,4]")!=0) {
        std::ostringstream err;
        err << "output error: expected \'[3,4]\', received \'";
        err << ss5.str() << "\'";
        return fail_test(testname, __FILE__, __LINE__, err.str().c_str());
    }

    return 0;
}

int test_op() {

    static const char testname[] = "index_test::test_op()";

    index<2> i1, i2, i3, i4;
    i1[0] = 3; i1[1] = 5;
    i2[0] = 3; i2[1] = 5;
    i3[0] = 0; i3[1] = 0;
    i4[0] = 3; i4[1] = 6;

    if (! (i1 == i2))
        return fail_test(testname, __FILE__, __LINE__, "operator==(i1, i2)");

    if (i1 != i2)
        return fail_test(testname, __FILE__, __LINE__, "operator!=(i1, i2)");

    if (i1 == i3)
        return fail_test(testname, __FILE__, __LINE__, "operator==(i1, i3)");

    if (! (i1 != i3))
        return fail_test(testname, __FILE__, __LINE__, "operator!=(i1, i3)");

    if (i1 < i2)
        return fail_test(testname, __FILE__, __LINE__, "operator<(i1, i2)");

    if (i1 < i3)
        return fail_test(testname, __FILE__, __LINE__, "operator<(i1, i3)");

    if (! (i1 < i4))
        return fail_test(testname, __FILE__, __LINE__, "operator<(i1, i4)");

    if (! (i1 <= i2))
        return fail_test(testname, __FILE__, __LINE__, "operator<=(i1, i2)");

    if (i1 <= i3)
        return fail_test(testname, __FILE__, __LINE__, "operator<=(i1, i3)");

    if (! (i1 <= i4))
        return fail_test(testname, __FILE__, __LINE__, "operator<=(i1, i4)");

    if (i1 > i2)
        return fail_test(testname, __FILE__, __LINE__, "operator>(i1, i2)");

    if (! (i1 > i3))
        return fail_test(testname, __FILE__, __LINE__, "operator>(i1, i3)");

    if (i1 > i4)
        return fail_test(testname, __FILE__, __LINE__, "operator>(i1, i4)");

    if (! (i1 >= i2))
        return fail_test(testname, __FILE__, __LINE__, "operator>=(i1, i2)");

    if (! (i1 >= i3))
        return fail_test(testname, __FILE__, __LINE__, "operator>=(i1, i3)");

    if (i1 > i4)
        return fail_test(testname, __FILE__, __LINE__, "operator>=(i1, i4)");

    return 0;
}


int main() {

    return

    test_ctor() |
    test_less() |
    test_print() |
    test_op() |

    0;
}

