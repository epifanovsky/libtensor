#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/block_tensor/btod_dotprod.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/ctf_block_tensor/ctf_block_tensor.h>
#include <libtensor/ctf_block_tensor/ctf_btod_distribute.h>
#include <libtensor/ctf_block_tensor/ctf_btod_dotprod.h>
#include "../compare_ref.h"
#include "ctf_btod_dotprod_test.h"

namespace libtensor {


void ctf_btod_dotprod_test::perform() throw(libtest::test_exception) {

    ctf::init();

    try {

        test_1();

    } catch(...) {
        ctf::exit();
        throw;
    }

    ctf::exit();
}


void ctf_btod_dotprod_test::test_1() {

    static const char testname[] = "ctf_btod_dotprod_test::test_1()";

    typedef allocator<double> allocator_t;

    try {

    mask<2> m10, m01;
    m10[0] = true; m01[1] = true;

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 89;
    block_index_space<2> bisa(dimensions<2>(index_range<2>(i1, i2)));
    bisa.split(m10, 30);
    bisa.split(m01, 40);
    block_index_space<2> bisb(bisa);
    bisb.permute(permutation<2>().permute(0, 1));

    block_tensor<2, double, allocator_t> bta(bisa), btb(bisb);
    ctf_block_tensor<2, double> dbta(bisa), dbtb(bisb);

    btod_random<2>().perform(bta);
    btod_random<2>().perform(btb);

    ctf_btod_distribute<2>(bta).perform(dbta);
    ctf_btod_distribute<2>(btb).perform(dbtb);

    permutation<2> perma;
    permutation<2> permb;
    permb.permute(0, 1);
    double d = ctf_btod_dotprod<2>(dbta, perma, dbtb, permb).calculate();
    double d_ref = btod_dotprod<2>(bta, perma, btb, permb).calculate();

    if(fabs(d - d_ref) > 1e-11) {
        std::ostringstream ss;
        ss << "Result does not match reference: " << d << " vs. "
            << d_ref << " (ref), " << (d - d_ref) << " (diff).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

