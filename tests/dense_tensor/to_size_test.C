#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/to_size.h>
#include "to_size_test.h"

namespace libtensor {

void to_size_test::perform() throw(libtest::test_exception) {
    std::cout << "Testing to_size_test_x<double>    ";
    to_size_test_x<double> t_double;
    t_double.perform();
    std::cout << "Testing to_size_test_x<float>    ";
    to_size_test_x<float> t_float;
    t_float.perform();
}

template<typename T>
void to_size_test_x<T>::perform() throw(libtest::test_exception) {

    allocator<T>::init(16, 16, 16777216, 16777216);

    try {

        test_1();

    } catch(...) {
        allocator<T>::shutdown();
        throw;
    }

    allocator<T>::shutdown();
}


template<typename T>
void to_size_test_x<T>::test_1() {

    static const char testname[] = "to_size_test::test_1()";

    typedef allocator<T> allocator_t;

    try {

        index<1> i1, i2;
        i2[0] = 10;
        dimensions<1> dims(index_range<1>(i1, i2));

        dense_tensor<1, T, allocator_t> t1(dims);

        size_t sz = to_size<1, T>().get_size(t1);
#if !defined(WITHOUT_LIBVMM)
        size_t sz_ref = 16 * sizeof(T);
#else
        size_t sz_ref = 11 * sizeof(T);
#endif
        if(sz != sz_ref) {
            fail_test(testname, __FILE__, __LINE__, "Bad size.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

