#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_unfold_symmetry.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/dense_tensor/tod_btconv.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_copy.h>
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

    symmetry<2, double> syma(bis);
    se_perm<2, double> se1(permutation<2>().permute(0, 1),
        scalar_transf<double>(1.0));
    syma.insert(se1);

    block_tensor<2, double, allocator_type> bta(bis), btb(bis);

    {
        block_tensor_wr_ctrl<2, double> ca(bta);
        so_copy<2, double>(syma).perform(ca.req_symmetry());
    }

    btod_random<2>().perform(bta);
    bta.set_immutable();
    btod_copy<2>(bta).perform(btb);

    gen_bto_unfold_symmetry<2, btod_traits>().perform(btb);

    {
        block_tensor_rd_ctrl<2, double> cb(btb);
        const symmetry<2, double> &symb = cb.req_const_symmetry();
        if(symb.begin() != symb.end()) {
            fail_test(testname, __FILE__, __LINE__,
                "Symmetry of B is not empty");
        }
    }

    dense_tensor<2, double, allocator_type> ta(dims), tb(dims);
    tod_btconv<2>(bta).perform(ta);
    tod_btconv<2>(btb).perform(tb);

    compare_ref<2>::compare(testname, ta, tb, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


} // namespace libtensor
