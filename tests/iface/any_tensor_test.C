#include <libtensor/exception.h>
#include <libtensor/iface/any_tensor_impl.h>
#include "any_tensor_test.h"

namespace libtensor {


void any_tensor_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
}


namespace {

class tensor_i {
public:
    bool equals(const tensor_i &other) const {
        return this == &other;
    }
};

class tensor : public tensor_i {

};

} // unnamed namespace


/** \test Tests a one-dimensional any_tensor
 **/
void any_tensor_test::test_1() {

    static const char testname[] = "any_tensor_test::test_1()";

    try {

    tensor t;
    tensor_i &ti = t;
    any_tensor<1, int> any(ti);

    if(!t.equals(any.get_tensor<tensor_i>())) {
        fail_test(testname, __FILE__, __LINE__, "Equality test failed.");
    }

    letter i;
    any(i);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    } catch(std::exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Tests a multi-dimensional any_tensor
 **/
void any_tensor_test::test_2() {

    static const char testname[] = "any_tensor_test::test_2()";

    try {

    tensor t;
    tensor_i &ti = t;
    any_tensor<4, int> any(ti);

    if(!t.equals(any.get_tensor<tensor_i>())) {
        fail_test(testname, __FILE__, __LINE__, "Equality test failed.");
    }

    letter i, j, k, l;
    any(i|j|k|l);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    } catch(std::exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

