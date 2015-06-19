#include <libtensor/libtensor.h>
#include <libtensor/core/allocator.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/ctf_block_tensor/ctf_btod_collect.h>
#include <libtensor/ctf_block_tensor/ctf_btod_distribute.h>
#include <libtensor/expr/ctf_btensor/ctf_btensor.h>
#include "../compare_ref.h"
#include "ctf_mult_test.h"

namespace libtensor {


void ctf_mult_test::perform() throw(libtest::test_exception) {

    allocator<double>::init(16, 16, 16777216, 16777216);
    ctf::init();

    try {

        test_tt_1a();
        test_tt_1b();

    } catch(...) {
        ctf::exit();
        allocator<double>::shutdown();
        throw;
    }

    ctf::exit();
    allocator<double>::shutdown();
}


void ctf_mult_test::test_tt_1a() {

    static const char testname[] = "mult_test::test_tt_1a()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);

    btensor<2, double> t1(sp_ia), t2(sp_ia), t3(sp_ia), t3_ref(sp_ia);
    ctf_btensor<2, double> dt1(sp_ia), dt2(sp_ia), dt3(sp_ia);

    btod_random<2>().perform(t1);
    btod_random<2>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    ctf_btod_distribute<2>(t1).perform(dt1);
    ctf_btod_distribute<2>(t2).perform(dt2);

    letter i, a;
    t3_ref(i|a) = mult(t1(i|a), t2(i|a));
    dt3(i|a) = mult(dt1(i|a), dt2(i|a));

    ctf_btod_collect<2>(dt3).perform(t3);

    compare_ref<2>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_mult_test::test_tt_1b() {

    static const char testname[] = "mult_test::test_tt_1b()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);

    btensor<2, double> t1(sp_ia), t2(sp_ia), t3(sp_ia), t3_ref(sp_ia);
    ctf_btensor<2, double> dt1(sp_ia), dt2(sp_ia), dt3(sp_ia);

    btod_random<2>().perform(t1);
    btod_random<2>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    ctf_btod_distribute<2>(t1).perform(dt1);
    ctf_btod_distribute<2>(t2).perform(dt2);

    letter i, a;
    t3_ref(i|a) = div(t1(i|a), t2(i|a));
    dt3(i|a) = div(dt1(i|a), dt2(i|a));

    ctf_btod_collect<2>(dt3).perform(t3);

    compare_ref<2>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
