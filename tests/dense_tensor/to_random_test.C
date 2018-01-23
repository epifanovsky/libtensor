#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/to_random.h>
#include "../compare_ref.h"
#include "to_random_test.h"

namespace libtensor {

void to_random_test::perform() throw(libtest::test_exception) {
    std::cout << "Testing to_random_test_x<double>   ";
    to_random_test_x<double> t_double;
    t_double.perform();
    std::cout << "Testing to_random_test_x<float>   ";
    to_random_test_x<float> t_float;
    t_float.perform();

}

template<typename T>
void to_random_test_x<T>::perform() throw(libtest::test_exception) {

    typedef allocator<T> allocator;
    typedef dense_tensor<3, T, allocator> tensor3;
    typedef dense_tensor_ctrl<3,T> tensor3_ctrl;

    index<3> i3a, i3b; i3b[0]=10; i3b[1]=12; i3b[2]=11;
    index_range<3> ir3(i3a, i3b); dimensions<3> dims3(ir3);
    tensor3 ta3(dims3), tb3(dims3);

    to_random<3, T> randr;
    bool test_ok=false;
    try {
        randr.perform(ta3);
        randr.perform(tb3);

        compare_ref_x<3, T>::compare("to_random_test",ta3,tb3,0.0);
    } catch ( libtest::test_exception& e ) {
        test_ok=true;
    } catch ( exception& e ) {
        fail_test("to_random_test", __FILE__, __LINE__, e.what());
    }
    if ( ! test_ok )
        fail_test("to_random_test", __FILE__, __LINE__,
            "Two identical random number sequences.");


    to_random<3, T>(2.0).perform(false, ta3);
    dense_tensor_ctrl<3,T> ctrla(ta3);
    const T *cptra=ctrla.req_const_dataptr();
    for (size_t i=0; i<ta3.get_dims().get_size(); i++ ) {
        if ( (cptra[i]<0.0) || (cptra[i]>=3.0) )
            fail_test("to_random_test<N>",__FILE__,__LINE__,
                "Random numbers outside specified interval");
    }
}

template class to_random_test_x<double>;
template class to_random_test_x<float>;

} // namespace libtensor

