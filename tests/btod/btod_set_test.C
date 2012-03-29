#include <libtensor/core/allocator.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/btod/scalar_transf_double.h>
#include <libtensor/btod/btod_set.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/tod/tod_btconv.h>
#include <libtensor/tod/tod_set.h>
#include "btod_set_test.h"
#include "../compare_ref.h"

namespace libtensor {


void btod_set_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
}


/** \test Sets all elements of an empty block %tensor to 1.0.
 **/
void btod_set_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "btod_set_test::test_1()";

    typedef std_allocator<double> allocator_t;

    cpu_pool cpus(1);

    try {

    index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    dimensions<2> bidims(bis.get_block_index_dims());

    block_tensor<2, double, allocator_t> bta(bis);
    dense_tensor<2, double, allocator_t> ta(dims), ta_ref(dims);

    //  Prepare the reference

    tod_set<2>(1.0).perform(cpus, ta_ref);

    //  Invoke the operation

    btod_set<2>(1.0).perform(bta);
    tod_btconv<2>(bta).perform(ta);

    //  Compare against the reference

    compare_ref<2>::compare(testname, ta, ta_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Sets all elements of a non-empty block %tensor to 1.0.
 **/
void btod_set_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "btod_set_test::test_2()";

    typedef std_allocator<double> allocator_t;

    cpu_pool cpus(1);

    try {

    index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    dimensions<2> bidims(bis.get_block_index_dims());

    block_tensor<2, double, allocator_t> bta(bis);
    dense_tensor<2, double, allocator_t> ta(dims), ta_ref(dims);

    //  Fill in random data

    btod_random<2>().perform(bta);

    //  Prepare the reference

    tod_set<2>(1.0).perform(cpus, ta_ref);

    //  Invoke the operation

    btod_set<2>(1.0).perform(bta);
    tod_btconv<2>(bta).perform(ta);

    //  Compare against the reference

    compare_ref<2>::compare(testname, ta, ta_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
