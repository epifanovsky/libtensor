#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include <libtensor/dense_tensor/tod_btconv.h>
#include <libtensor/symmetry/se_perm.h>
#include "../compare_ref.h"
#include "gen_bto_unfold_symmetry_test.h"

namespace libtensor {


void gen_bto_unfold_symmetry_test::perform() throw(libtest::test_exception) {

    allocator<double>::vmm().init(16, 16, 65536, 65536);

    try {

    test_1();

    } catch (...) {
        allocator<double>::vmm().shutdown();
        throw;
    }
    allocator<double>::vmm().shutdown();
}


void gen_bto_unfold_symmetry_test::test_1() {

    static const char *testname = "gen_bto_unfold_symmetry_test::test_1()";

    typedef std_allocator<double> allocator_type;

    try {

    index<2> i1, i2;
    i2[0] = 10; i2[1] = 10;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    dimensions<2> bidims(bis.get_block_index_dims());
    dense_tensor<2, double, allocator_type> ta(dims), tb(dims);
    block_tensor<2, double, allocator_type> bta(bis), btb(bis);

    //  Fill the output with random data

//    btod_random<2>().perform(btb);
//    bta.set_immutable();

    //  Make a copy

//    btod_copy<2>(bta).perform(btb);

    //  The set of non-zero blocks in the output must be empty now

//    block_tensor_ctrl<2, double> btb_ctrl(btb);
//    orbit_list<2, double> orblst(btb_ctrl.req_symmetry());
//    orbit_list<2, double>::iterator iorbit = orblst.begin();
//    for(; iorbit != orblst.end(); iorbit++) {
//        orbit<2, double> orb(btb_ctrl.req_symmetry(),
//            orblst.get_index(iorbit));
//        abs_index<2> blkidx(orb.get_abs_canonical_index(), bidims);
//        if(!btb_ctrl.req_is_zero_block(blkidx.get_index())) {
//            fail_test(testname, __FILE__, __LINE__,
//                "All blocks are expected to be empty.");
//        }
//    }

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


} // namespace libtensor
