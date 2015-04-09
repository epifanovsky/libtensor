#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/block_tensor/btod_symmetrize3.h>
#include <libtensor/ctf_block_tensor/ctf_block_tensor.h>
#include <libtensor/ctf_block_tensor/ctf_btod_collect.h>
#include <libtensor/ctf_block_tensor/ctf_btod_copy.h>
#include <libtensor/ctf_block_tensor/ctf_btod_distribute.h>
#include <libtensor/ctf_block_tensor/ctf_btod_symmetrize3.h>
#include "../compare_ref.h"
#include "ctf_btod_symmetrize3_test.h"

namespace libtensor {


void ctf_btod_symmetrize3_test::perform() throw(libtest::test_exception) {

    ctf::init();

    try {

        test_1(true);
        test_1(false);

    } catch(...) {
        ctf::exit();
        throw;
    }

    ctf::exit();
}


void ctf_btod_symmetrize3_test::test_1(bool symm) {

    std::ostringstream tnss;
    tnss << "ctf_btod_symmetrize3_test::test_1(" << symm << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

    mask<3> m111;
    m111[0] = true; m111[1] = true; m111[2] = true;

    index<3> i1, i2;
    i2[0] = 19; i2[1] = 19; i2[2] = 19;
    dimensions<3> dimsa(index_range<3>(i1, i2)), dimsb(dimsa);
    block_index_space<3> bisa(dimsa);
    bisa.split(m111, 6);
    block_index_space<3> bisb(bisa);

    block_tensor<3, double, allocator_t> bta(bisa), btb(bisb), btb_ref(bisb);
    ctf_block_tensor<3, double> dbta(bisa), dbtb(bisb);

    btod_random<3>().perform(bta);
    btod_random<3>().perform(btb);

    btod_copy<3> cp(bta);
    btod_symmetrize3<3>(cp, 0, 1, 2, symm).perform(btb_ref);

    ctf_btod_distribute<3>(bta).perform(dbta);
    ctf_btod_distribute<3>(btb).perform(dbtb);

    ctf_btod_copy<3> dcp(dbta);
    ctf_btod_symmetrize3<3>(dcp, 0, 1, 2, symm).perform(dbtb);
    ctf_btod_collect<3>(dbtb).perform(btb);

    compare_ref<3>::compare(tn.c_str(), btb, btb_ref, 1e-15);

    // check block symmetry

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

