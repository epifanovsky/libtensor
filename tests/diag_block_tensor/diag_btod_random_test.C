#include <libtensor/core/allocator.h>
#include <libtensor/diag_block_tensor/diag_block_tensor.h>
#include <libtensor/diag_block_tensor/diag_btod_random.h>
#include "diag_btod_random_test.h"
#include "../compare_ref.h"

namespace libtensor {


void diag_btod_random_test::perform() throw(libtest::test_exception) {

    test_1();
}


void diag_btod_random_test::test_1() {

    static const char *testname = "diag_btod_random_test::test_1()";

    typedef allocator<double> allocator_t;

    try {

    index<2> i0, i1, i2;
    i2[0] = 15; i2[1] = 15;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);

    diag_block_tensor<2, double, allocator_t> bt(bis);

    diag_btod_random<2>().perform(bt);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


} // namespace libtensor
