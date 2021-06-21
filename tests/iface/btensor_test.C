#include <libtensor/expr/btensor/btensor.h>
#include "btensor_test.h"

namespace libtensor {


void btensor_test::perform() {

    test_1();
    test_2();
}

using expr::label;

/** \test Checks the dimensions of a new btensor
 **/
void btensor_test::test_1() {

    static const char *testname = "btensor_test::test_1()";

    try {

    bispace<1> i_sp(10), a_sp(20);
    i_sp.split(5); a_sp.split(5).split(10).split(15);
    bispace<2> ia(i_sp|a_sp);
    btensor<2> bt2(ia);

    dimensions<2> bt2_dims(bt2.get_bis().get_dims());
    if(bt2_dims[0] != 10) {
        fail_test("btensor_test::perform()", __FILE__, __LINE__,
            "Block tensor bt2 has the wrong dimension: i");
    }

    if(bt2_dims[1] != 20) {
        fail_test("btensor_test::perform()", __FILE__, __LINE__,
            "Block tensor bt2 has the wrong dimension: a");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Checks operator() with various letter labels
 **/
void btensor_test::test_2() {

    static const char *testname = "btensor_test::test_2()";

    try {

    bispace<1> s(10);
    bispace<2> ss(s&s);

    letter i, j;

    btensor<1> bt1(s);
    bt1(i);
    label<1> le_i(i);
    bt1(le_i);

    btensor<2> bt2(ss);
    bt2(i|j);
    label<2> le_ij(i|j);
    bt2(le_ij);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

