#include <libtensor/exception.h>
#include <libtensor/expr/iface/any_tensor_impl.h>
#include "../test_utils.h"

using namespace libtensor;

namespace {

class tensor_i {
public:
    bool equals(const tensor_i &other) const {
        return this == &other;
    }
};

class tensor_impl : public tensor_i {

};

template<size_t N, typename T>
class tensor : public any_tensor<N, T> {
public:
    tensor_impl t;

public:
    tensor() : any_tensor<N, T>((tensor_i&)t) { }
};

} // unnamed namespace


/** \test Tests a one-dimensional any_tensor
 **/
int test_1() {

    static const char testname[] = "test_1()";

    try {

    tensor<1, int> t;
    any_tensor<1, int> &any = t;

    if(!t.t.equals(any.get_tensor<tensor_i>())) {
        return fail_test(testname, __FILE__, __LINE__, "Equality test failed.");
    }

    letter i;
    any(i);

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    } catch(std::exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


/** \test Tests a multi-dimensional any_tensor
 **/
int test_2() {

    static const char testname[] = "test_2()";

    try {

    tensor<4, int> t;
    any_tensor<4, int> &any = t;

    if(!t.t.equals(any.get_tensor<tensor_i>())) {
        return fail_test(testname, __FILE__, __LINE__, "Equality test failed.");
    }

    letter i, j, k, l;
    any(i|j|k|l);

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    } catch(std::exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int main() {

    return

    test_1() |
    test_2() |

    0;
}
