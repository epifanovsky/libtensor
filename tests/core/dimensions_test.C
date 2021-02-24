#include <libtensor/core/dimensions.h>
#include "../test_utils.h"

using namespace libtensor;


int test_ctor() {

    static const char testname[] = "dimensions_test::test_ctor()";

    try {

    libtensor::index<2> i1a, i1b;
    i1b[0] = 1; i1b[1] = 2;
    index_range<2> ir1(i1a, i1b); // Indexes run from (0,0) to (1,2)
    dimensions<2> d1(ir1);

    if(d1[0] != 2) {
        return fail_test(testname, __FILE__, __LINE__,
            "Incorrect number of elements along d1[0]");
    }
    if(d1[1] != 3) {
        return fail_test(testname, __FILE__, __LINE__,
            "Incorrect number of elements along d1[1]");
    }
    if(d1.get_size() != 6) {
        return fail_test(testname, __FILE__, __LINE__,
            "Incorrect total number of elements in d1");
    }

    dimensions<2> d2(d1);

    if(d2[0] != 2) {
        return fail_test(testname, __FILE__, __LINE__,
            "Incorrect number of elements along d2[0]");
    }
    if(d2[1] != 3) {
        return fail_test(testname, __FILE__, __LINE__,
            "Incorrect number of elements along d2[1]");
    }
    if(d2.get_size() != 6) {
        return fail_test(testname, __FILE__, __LINE__,
            "Incorrect total number of elements in d2");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_contains() {

    static const char testname[] = "dimensions_test::test_contains()";

    try {

    libtensor::index<2> i1, i2;
    i2[0] = 10; i2[1] = 12;
    dimensions<2> d1(index_range<2> (i1, i2));

    i1[0] = 0; i1[1] = 0;
    if(!d1.contains(i1)) {
        return fail_test(testname, __FILE__, __LINE__,
            "(11,13).contains(0,0) returns false");
    }

    i1[1] = 1;
    if(!d1.contains(i1)) {
        return fail_test(testname, __FILE__, __LINE__,
            "(11,13).contains(0,1) returns false");
    }

    i1[1] = 12;
    if(!d1.contains(i1)) {
        return fail_test(testname, __FILE__, __LINE__,
            "(11,13).contains(0,12) returns false");
    }

    i1[1] = 13;
    if(d1.contains(i1)) {
        return fail_test(testname, __FILE__, __LINE__,
            "(11,13).contains(0,13) returns true");
    }

    i1[0] = 1; i1[1] = 0;
    if(!d1.contains(i1)) {
        return fail_test(testname, __FILE__, __LINE__,
            "(11,13).contains(1,0) returns false");
    }

    i1[0] = 10;
    if(!d1.contains(i1)) {
        return fail_test(testname, __FILE__, __LINE__,
            "(11,13).contains(10,0) returns false");
    }

    i1[0] = 11;
    if(d1.contains(i1)) {
        return fail_test(testname, __FILE__, __LINE__,
            "(11,13).contains(11,0) returns true");
    }

    i1[1] = 100;
    if(d1.contains(i1)) {
        return fail_test(testname, __FILE__, __LINE__,
            "(11,13).contains(11,100) returns true");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_comp() {

    static const char testname[] = "dimensions_test::test_comp()";

    try {

    libtensor::index<2> i1a, i1b, i2b;
    i1b[0] = 1; i1b[1] = 2; i1b[0] = 2; i1b[1] = 3;
    index_range<2> ir1(i1a, i1b), ir2(i1a, i2b);
    dimensions<2> d1(ir1), d2(d1), d3(ir2);

    if(!(d1 == d2)) {
        return fail_test(testname, __FILE__, __LINE__,
            "Operator == on identical dimensions returned false.");
    }
    if(d1 == d3) {
        return fail_test(testname, __FILE__, __LINE__,
            "Operator == on different dimensions returned true.");
    }
    if(d1 != d2) {
        return fail_test(testname, __FILE__, __LINE__,
            "Operator != on identical dimensions returned true.");
    }

    if(!(d1 != d3)) {
        return fail_test(testname, __FILE__, __LINE__,
            "Operator != on different dimensions returned false.");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int main() {

    return

    test_ctor() |
    test_contains() |
    test_comp() |

    0;
}

