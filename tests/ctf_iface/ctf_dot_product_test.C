#include <libtensor/libtensor.h>
#include <libtensor/core/allocator.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/ctf_block_tensor/ctf_btod_collect.h>
#include <libtensor/ctf_block_tensor/ctf_btod_distribute.h>
#include <libtensor/expr/ctf_btensor/ctf_btensor.h>
#include "../compare_ref.h"
#include "ctf_dot_product_test.h"

namespace libtensor {


void ctf_dot_product_test::perform() throw(libtest::test_exception) {

    allocator<double>::init(16, 16, 16777216, 16777216);
    ctf::init();

    try {

        test_1();

    } catch(...) {
        ctf::exit();
        allocator<double>::shutdown();
        throw;
    }

    ctf::exit();
    allocator<double>::shutdown();
}


void ctf_dot_product_test::test_1() {

    static const char testname[] = "ctf_dot_product_test::test_1()";

    try {

    bispace<1> so(13); so.split(3).split(7).split(10);
    bispace<1> sv(7); sv.split(2).split(3).split(5);

    bispace<2> sov(so|sv);

    btensor<2, double> t1(sov), t2(sov);
    ctf_btensor<2, double> dt1(sov), dt2(sov);

    btod_random<2>().perform(t1);
    btod_random<2>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    ctf_btod_distribute<2>(t1).perform(dt1);
    ctf_btod_distribute<2>(t2).perform(dt2);

    letter i, a;

    double d_ref = dot_product(t1(i|a), t2(i|a));
    double d = dot_product(dt1(i|a), dt2(i|a));

    compare_ref<1>::compare(testname, d, d_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
