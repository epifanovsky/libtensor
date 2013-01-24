#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/cuda_block_tensor/cuda_block_tensor.h>
#include <libtensor/cuda_block_tensor/cuda_btod_copy_d2h.h>
#include <libtensor/cuda_block_tensor/cuda_btod_copy_h2d.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/core/allocator.h>
#include <libvmm/cuda_allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
//#include <libtensor/dense_tensor/tod_btconv.h>
//#include <libtensor/symmetry/se_perm.h>
#include "../compare_ref.h"
#include "cuda_btod_copy_hd_test.h"

namespace libtensor {


void cuda_btod_copy_hd_test::perform() throw(libtest::test_exception) {

    allocator<double>::init(16, 16, 65536, 65536);

    try {

    test1();


    }
    catch (...) {
        allocator<double>::shutdown();
        throw;
    }
    allocator<double>::shutdown();
}


void cuda_btod_copy_hd_test::test1() throw(libtest::test_exception) {

    static const char *testname = "cuda_btod_copy_hd_test::test1()";

    typedef std_allocator<double> allocator_t;
    typedef libvmm::cuda_allocator<double> cuda_allocator_t;

        try {

        index<2> i1, i2;
        i2[0] = 10; i2[1] = 10;
        dimensions<2> dims(index_range<2>(i1, i2));
        block_index_space<2> bis(dims);
        dense_tensor<2, double, allocator_t> ta(dims), ta_copy(dims);
//        dense_tensor<2, double, cuda_allocator_t> tb(dims);
        block_tensor<2, double, allocator_t> bta(bis), bta_copy(bis);
        cuda_block_tensor<2, double, cuda_allocator_t> btb(bis);

        //  Fill the input with random data

        btod_random<2>().perform(bta);
        bta.set_immutable();

        //  Copy from device to host memory

        cuda_btod_copy_h2d<2>(bta).perform(btb);

        //  Copy back from host to device memory

        cuda_btod_copy_d2h<2>(btb).perform(bta_copy);

        //  Compare against the reference

        tod_btconv<2>(bta).perform(ta);
        tod_btconv<2>(bta_copy).perform(ta_copy);

        compare_ref<2>::compare(testname, ta_copy, ta, 0.0);

        } catch(exception &exc) {
            fail_test(testname, __FILE__, __LINE__, exc.what());
        }
}

} // namespace libtensor
