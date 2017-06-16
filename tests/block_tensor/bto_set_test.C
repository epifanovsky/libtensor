#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/bto_set.h>
#include <libtensor/block_tensor/bto_random.h>
#include <libtensor/dense_tensor/to_btconv.h>
#include <libtensor/dense_tensor/to_set.h>
#include "bto_set_test.h"
#include "../compare_ref.h"

namespace libtensor {


void bto_set_test::perform() throw(libtest::test_exception) {
    std::cout << "Testing bto_set_test_x<double>  ";
    bto_set_test_x<double> d_test;
    d_test.perform();
    std::cout << "Testing bto_set_test_x<float>  ";
    bto_set_test_x<float> f_test;
    f_test.perform();
}

template<>
const double bto_set_test_x<double>::k_thresh = 1e-15;

template<>
const float bto_set_test_x<float>::k_thresh = 1e-7;

template<typename T>
void bto_set_test_x<T>::perform() throw(libtest::test_exception) {

    allocator<T>::init(4, 16, 65536, 65536);
    try {

    test_1();
    test_2();

    } catch (...) {
        allocator<T>::shutdown();
        throw;
    }
    allocator<T>::shutdown();
}


/** \test Sets all elements of an empty block %tensor to 1.0.
 **/
template<typename T>
void bto_set_test_x<T>::test_1() throw(libtest::test_exception) {

    static const char *testname = "bto_set_test_x::test_1()";

    typedef allocator<T> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    dimensions<2> bidims(bis.get_block_index_dims());

    block_tensor<2, T, allocator_t> bta(bis);
    dense_tensor<2, T, allocator_t> ta(dims), ta_ref(dims);

    //  Prepare the reference

    to_set<2, T>(1.0).perform(true, ta_ref);

    //  Invoke the operation

    bto_set<2, T>(1.0).perform(bta);
    to_btconv<2, T>(bta).perform(ta);

    //  Compare against the reference

    compare_ref_x<2, T>::compare(testname, ta, ta_ref, k_thresh);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Sets all elements of a non-empty block %tensor to 1.0.
 **/
template<typename T>
void bto_set_test_x<T>::test_2() throw(libtest::test_exception) {

    static const char *testname = "bto_set_test_x::test_2()";

    typedef allocator<T> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    dimensions<2> bidims(bis.get_block_index_dims());

    block_tensor<2, T, allocator_t> bta(bis);
    dense_tensor<2, T, allocator_t> ta(dims), ta_ref(dims);

    //  Fill in random data

    bto_random<2, T>().perform(bta);

    //  Prepare the reference

    to_set<2, T>(1.0).perform(true, ta_ref);

    //  Invoke the operation

    bto_set<2, T>(1.0).perform(bta);
    to_btconv<2, T>(bta).perform(ta);

    //  Compare against the reference

    compare_ref_x<2, T>::compare(testname, ta, ta_ref, k_thresh);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

template class bto_set_test_x<double>;
template class bto_set_test_x<float>;

} // namespace libtensor
