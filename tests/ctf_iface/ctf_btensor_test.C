#include <libtensor/expr/ctf_btensor/ctf_btensor.h>
#include "ctf_btensor_test.h"

namespace libtensor {


void ctf_btensor_test::perform() throw(libtest::test_exception) {

    test_1();
}


void ctf_btensor_test::test_1() {

    static const char testname[] = "ctf_btensor_test::test_1()";

    try {

    bispace<1> o(10), v(20);;
    o.split(5);
    v.split(10);
    bispace<2> ov(o|v);

    ctf_btensor<2, double> bt(ov);

    dimensions<2> btdims(bt.get_bis().get_dims());
    if(btdims[0] != 10) {
        fail_test(testname, __FILE__, __LINE__, "btdims[0] != 10");
    }
    if(btdims[1] != 20) {
        fail_test(testname, __FILE__, __LINE__, "btdims[1] != 20");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

