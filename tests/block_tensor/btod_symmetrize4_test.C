#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include <libtensor/block_tensor/btod_add.h>
#include <libtensor/block_tensor/btod_contract2.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/block_tensor/btod_symmetrize4.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/se_label.h>
#include <libtensor/symmetry/se_part.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/dense_tensor/tod_btconv.h>
#include "btod_symmetrize4_test.h"
#include "../compare_ref.h"

namespace libtensor {


void btod_symmetrize4_test::perform() {

    allocator<double>::init();

    try {

        test_1();

    } catch(...) {
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();
}


/** \test Symmetrization of a non-symmetric 4-index block tensor
        over four indexes
 **/
void btod_symmetrize4_test::test_1() {

    static const char testname[] = "btod_symmetrize4_test::test_1()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<4> i1, i2;
    i2[0] = 10; i2[1] = 10; i2[2] = 10; i2[3] = 10;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> m;
    m[0] = true; m[1] = true; m[2] = true; m[2] = true;
    bis.split(m, 2);
    bis.split(m, 5);

    block_tensor<4, double, allocator_t> bta(bis), btb(bis), btb_ref(bis);

    //  Fill in random input

    btod_random<4>().perform(bta);
    bta.set_immutable();

    //  Prepare reference data

    dense_tensor<4, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
    tod_btconv<4>(bta).perform(ta);
    permutation<4> perm;
    tod_add<4> refop(ta);
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 3; j++) {
            if(i > 0 || j > 0) refop.add_op(ta, perm, 1.0);
            perm.permute(2, 3);
            refop.add_op(ta, perm, 1.0);
            perm.permute(2, 3);
            perm.permute(1, 2).permute(2, 3);
        }
        perm.permute(0, 1).permute(1, 2).permute(2, 3);
    }
    refop.perform(true, tb_ref);

    symmetry<4, double> symb(bis), symb_ref(bis);
    scalar_transf<double> tr0, tr1(-1.0);
    symb_ref.insert(se_perm<4, double>(
        permutation<4>().permute(0, 1), tr0));
    symb_ref.insert(se_perm<4, double>(
        permutation<4>().permute(0, 2), tr0));
    symb_ref.insert(se_perm<4, double>(
        permutation<4>().permute(0, 3), tr0));

    //  Run the symmetrization operation

    btod_copy<4> op_copy(bta);
    btod_symmetrize4<4> op_sym(op_copy, 0, 1, 2, 3, true);

    compare_ref<4>::compare(testname, op_sym.get_symmetry(), symb_ref);

    op_sym.perform(btb);
    tod_btconv<4>(btb).perform(tb);

    //  Compare against the reference: symmetry and data

    {
        block_tensor_ctrl<4, double> ctrlb(btb);
        so_copy<4, double>(ctrlb.req_const_symmetry()).perform(symb);
    }

    compare_ref<4>::compare(testname, symb, symb_ref);
    compare_ref<4>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


} // namespace libtensor
