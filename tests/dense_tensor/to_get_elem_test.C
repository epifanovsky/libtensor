#include <cmath>
#include <ctime>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/to_get_elem.h>
#include "../compare_ref.h"
#include "to_get_elem_test.h"

namespace libtensor {

void to_get_elem_test::perform() throw(libtest::test_exception) {
    std::cout << "Testing to_get_elem_test_x<double>    ";
    to_get_elem_test_x<double> t_double;
    t_double.perform();
    std::cout << "Testing to_get_elem_test_x<float>    ";
    to_get_elem_test_x<float> t_float;
    t_float.perform();
}

template<typename T>
void to_get_elem_test_x<T>::perform() throw(libtest::test_exception) {

    srand48(time(0));

    test_1();
}


template<typename T>
void to_get_elem_test_x<T>::test_1() throw(libtest::test_exception) {

    static const char *testname = "to_get_elem_test<T>::test_1()";

    typedef allocator<T> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 3; i2[1] = 4;
    dimensions<2> dims(index_range<2>(i1, i2));
    dense_tensor<2, T, allocator_t> t(dims), t_ref(dims);

    {
    dense_tensor_ctrl<2, T> tc(t), tc_ref(t_ref);

    // Fill in random data
    //
    {
    T *d = tc.req_dataptr();
    T *d_ref = tc_ref.req_dataptr();
    size_t sz = dims.get_size();
    for(size_t i = 0; i < sz; i++) d_ref[i] = d[i] = drand48();
    tc.ret_dataptr(d); d = 0;
    tc_ref.ret_dataptr(d_ref); d_ref = 0;
    }

    // Test [0,0]
    //
    {
    index<2> i00;
    abs_index<2> ai00(i00, dims);
    const T* cd_ref = tc_ref.req_const_dataptr();
    T q = cd_ref[ai00.get_abs_index()];
    tc_ref.ret_const_dataptr(cd_ref); //d_ref = 0;
    T elem;
    to_get_elem<2, T>().perform(t, i00, elem);
    compare_ref_x<2, T>::compare(testname, q, elem);
    }
    //std::cout << "MY compare: " << q << "  " << elem << std::endl;

    // Test [3, 2]
    //
    { 
    index<2> i32; i32[0] = 3; i32[1] = 2;
    abs_index<2> ai32(i32, dims);
    const T* cd_ref = tc_ref.req_const_dataptr();
    T q = cd_ref[ai32.get_abs_index()];
    tc_ref.ret_const_dataptr(cd_ref); //d_ref = 0;
    T elem;
    to_get_elem<2, T>().perform(t, i32, elem);
    compare_ref_x<2, T>::compare(testname, q, elem);
    }
    
    }


    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
