#include <vector>
#include <libtensor/expr/anytensor.h>
#include <libtensor/exception.h>
#include "anytensor_test.h"

namespace libtensor {


void anytensor_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
}


namespace anytensor_test_ns {

template<size_t N, typename T>
class anytensor2 : public anytensor<N, T> {
public:
    static const char *k_tensor_type;

    anytensor2() { }

    template<typename TensorType>
    explicit anytensor2(TensorType &t) : anytensor<N, T>(t) { }

    virtual const char *get_tensor_type() const {
        return k_tensor_type;
    }

};

template<size_t N, typename T>
const char *anytensor2<N, T>::k_tensor_type = "anytensor2";

} // namespace anytensor_test_ns
using namespace anytensor_test_ns;


/** \test Creates an empty anytensor
 **/
void anytensor_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "anytensor_test::test_1()";

    try {

    anytensor2<2, double> a;
    anytensor2<1, double> b, x;

    } catch(exception &e) {
            fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Fills anytensor with an std::vector
 **/
void anytensor_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "anytensor_test::test_2()";

    try {

    std::vector<int> v(5, 0);
    anytensor2<1, int> b(v);

    } catch(exception &e) {
            fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

