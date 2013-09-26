#include <mpi.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/gen_block_tensor/gen_block_tensor_ctrl.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_distribute.h>
#include <libtensor/ctf_block_tensor/ctf_block_tensor.h>
#include <libtensor/ctf_block_tensor/ctf_btod_collect.h>
#include "../compare_ref.h"
#include "ctf_btod_collect_test.h"

namespace libtensor {


void ctf_btod_collect_test::perform() throw(libtest::test_exception) {

    int nproc, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ctf::get().init(MPI_COMM_WORLD, rank, nproc);

    try {

        test_1();
        test_2();

    } catch(...) {
        ctf::get().exit();
        throw;
    }

    ctf::get().exit();
}


void ctf_btod_collect_test::test_1() {

    static const char testname[] = "ctf_btod_collect_test::test_1()";

    typedef std_allocator<double> allocator_t;
    typedef ctf_block_tensor_i_traits<double> ctf_bti_traits;

    try {

    mask<2> m11;
    m11[0] = true; m11[1] = true;

    index<2> i1, i2;
    i2[0] = 199; i2[1] = 199;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    bis.split(m11, 90);
    dimensions<2> bidims = bis.get_block_index_dims();

    block_tensor<2, double, allocator_t> bta(bis), btb(bis), btb_ref(bis);
    ctf_block_tensor<2, double> btd(bis);

    btod_random<2>().perform(bta);

    //  Prepare the distributed block tensor

    {
        block_tensor_rd_ctrl<2, double> ca(bta);
        gen_block_tensor_wr_ctrl<2, ctf_bti_traits> cd(btd);

        abs_index<2> aidx(bidims);
        do {
            index<2> idx(aidx.get_index());
            dense_tensor_rd_i<2, double> &blka = ca.req_const_block(idx);
            ctf_dense_tensor_i<2, double> &blkd = cd.req_block(idx);
            ctf_tod_distribute<2>(blka).perform(blkd);
            cd.ret_block(idx);
            ca.ret_const_block(idx);
        } while(aidx.inc());
    }

    //  Collect the block tensor locally

    ctf_btod_collect<2>(btd).perform(btb);

    //  Prepare the reference

    btod_copy<2>(bta).perform(btb_ref);

    compare_ref<2>::compare(testname, btb, btb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_btod_collect_test::test_2() {

    static const char testname[] = "ctf_btod_collect_test::test_2()";

    typedef std_allocator<double> allocator_t;
    typedef ctf_block_tensor_i_traits<double> ctf_bti_traits;

    try {

    mask<2> m11;
    m11[0] = true; m11[1] = true;

    index<2> i1, i2;
    i2[0] = 199; i2[1] = 199;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    bis.split(m11, 90);
    dimensions<2> bidims = bis.get_block_index_dims();

    se_perm<2, double> se(permutation<2>().permute(0, 1),
        scalar_transf<double>());

    block_tensor<2, double, allocator_t> bta(bis), btb(bis), btb_ref(bis);
    ctf_block_tensor<2, double> btd(bis);

    {
        block_tensor_ctrl<2, double> ca(bta);
        ca.req_symmetry().insert(se);
    }
    btod_random<2>().perform(bta);

    //  Prepare the distributed block tensor

    {
        block_tensor_rd_ctrl<2, double> ca(bta);
        gen_block_tensor_wr_ctrl<2, ctf_bti_traits> cd(btd);

        cd.req_symmetry().insert(se);

        abs_index<2> aidx(bidims);
        do {
            index<2> idx(aidx.get_index());
            if(idx[0] > idx[1]) continue;
            dense_tensor_rd_i<2, double> &blka = ca.req_const_block(idx);
            ctf_dense_tensor_i<2, double> &blkd = cd.req_block(idx);
            ctf_tod_distribute<2>(blka).perform(blkd);
            cd.ret_block(idx);
            ca.ret_const_block(idx);
        } while(aidx.inc());
    }

    //  Collect the block tensor locally

    ctf_btod_collect<2>(btd).perform(btb);

    //  Prepare the reference

    btod_copy<2>(bta).perform(btb_ref);

    compare_ref<2>::compare(testname, btb, btb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

