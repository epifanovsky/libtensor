#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/block_tensor/btod_trace.h>
#include <libtensor/ctf_block_tensor/ctf_block_tensor.h>
#include <libtensor/ctf_block_tensor/ctf_btod_distribute.h>
#include <libtensor/ctf_block_tensor/ctf_btod_trace.h>
#include "../compare_ref.h"
#include "ctf_btod_trace_test.h"

namespace libtensor {


void ctf_btod_trace_test::perform() throw(libtest::test_exception) {

    ctf::init();

    try {

        test_1();

    } catch(...) {
        ctf::exit();
        throw;
    }

    ctf::exit();
}


void ctf_btod_trace_test::test_1() {

    static const char testname[] = "ctf_btod_trace_test::test_1()";

    typedef std_allocator<double> allocator_t;

    try {

    mask<4> m1010, m0101;
    m1010[0] = true; m0101[1] = true; m1010[2] = true; m0101[3] = true;

    index<4> i1, i2;
    i2[0] = 29; i2[1] = 19; i2[2] = 29; i2[3] = 19;
    block_index_space<4> bisa(dimensions<4>(index_range<4>(i1, i2)));
    bisa.split(m1010, 10);
    bisa.split(m0101, 15);

    block_tensor<4, double, allocator_t> bta(bisa);
    ctf_block_tensor<4, double> dbta(bisa);

    btod_random<4>().perform(bta);

    ctf_btod_distribute<4>(bta).perform(dbta);

    double d = ctf_btod_trace<2>(dbta).calculate();
    double d_ref = btod_trace<2>(bta).calculate();

    if(fabs(d - d_ref) > 1e-12) {
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

