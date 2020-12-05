#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod_size.h>
#include "../test_utils.h"

using namespace libtensor;


int test_1() {

    static const char testname[] = "tod_size_test::test_1()";

    typedef allocator<double> allocator_t;

    try {

        libtensor::index<1> i1, i2;
        i2[0] = 10;
        dimensions<1> dims(index_range<1>(i1, i2));

        dense_tensor<1, double, allocator_t> t1(dims);

        size_t sz = tod_size<1>().get_size(t1);
/*
#if !defined(WITHOUT_LIBVMM)
        size_t sz_ref = 16 * sizeof(double);
#else
        size_t sz_ref = 11 * sizeof(double);
#endif
        if(sz != sz_ref) {
            return fail_test(testname, __FILE__, __LINE__, "Bad size.");
        }
 */

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int main() {

    int rc = 0;

    allocator<double>::init(16, 16, 16777216, 16777216);

    try {

        rc =

        test_1() |

        0;

    } catch(...) {
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();

    return rc;
}


