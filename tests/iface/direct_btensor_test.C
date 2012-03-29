#include <libtensor/btod/scalar_transf_double.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/iface/iface.h>
#include "direct_btensor_test.h"
#include "../compare_ref.h"

namespace libtensor {


void direct_btensor_test::perform() throw(libtest::test_exception) {

    allocator<double>::vmm().init(16, 16, 16777216, 16777216);

    try {

        test_1();
        test_2();
        test_3();
        test_4();

    } catch(...) {
        allocator<double>::vmm().shutdown();
        throw;
    }

    allocator<double>::vmm().shutdown();
}


void direct_btensor_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "direct_btensor_test::test_1()";

    try {

    bispace<1> si(5), sa(6);
    si.split(2); sa.split(3);
    bispace<2> sia(si|sa);

    btensor<2> bt1(sia);
    btod_random<2>().perform(bt1);

    letter i, a;
    direct_btensor<2> dbt(i|a, 1.5 * bt1(i|a));

    btensor<2> bt(sia), bt_ref(sia);
    btod_copy<2>(bt1, 1.5).perform(bt_ref);

    bt(i|a) = dbt(i|a);

    compare_ref<2>::compare(testname, bt, bt_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void direct_btensor_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "direct_btensor_test::test_2()";

    try {

    bispace<1> si(5), sa(6);
    si.split(2); sa.split(3);
    bispace<2> sia(si|sa), sai(sa|si);

    btensor<2> bt1(sia);
    btod_random<2>().perform(bt1);

    letter i, a;
    direct_btensor<2> dbt(a|i, -0.5 * bt1(i|a));

    btensor<2> bt(sai), bt_ref(sai);

    permutation<2> p10;
    p10.permute(0, 1);
    btod_copy<2>(bt1, p10, -0.5).perform(bt_ref);

    bt(a|i) = dbt(a|i);

    compare_ref<2>::compare(testname, bt, bt_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void direct_btensor_test::test_3() throw(libtest::test_exception) {

    static const char *testname = "direct_btensor_test::test_3()";

    try {

    bispace<1> si(5), sa(6);
    si.split(2); sa.split(3);
    bispace<2> sia(si|sa), sai(sa|si);

    btensor<2> bt1(sia), bt2(sia);
    btod_random<2>().perform(bt1);
    btod_random<2>().perform(bt2);

    letter i, a;
    direct_btensor<2> dbt(i|a, bt1(i|a) - bt2(i|a));

    btensor<2> bt(sia), bt_ref(sia);

    btod_copy<2>(bt1).perform(bt_ref);
    btod_copy<2>(bt2).perform(bt_ref, -1.0);

    bt(i|a) = dbt(i|a);

    compare_ref<2>::compare(testname, bt, bt_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void direct_btensor_test::test_4() throw(libtest::test_exception) {

    static const char *testname = "direct_btensor_test::test_4()";

    try {

    bispace<1> si(5), sa(6);
    si.split(2); sa.split(3);
    bispace<2> sia(si|sa), sai(sa|si);
    bispace<4> sijab(si&si|sa&sa);

    btensor<2> bt1(sia), bt2(sia);
    btod_random<2>().perform(bt1);
    btod_random<2>().perform(bt2);

    letter i, j, a, b;
    direct_btensor<4> dbt(i|j|a|b, dirsum(bt1(i|a), bt2(j|b)));

    btensor<4> bt(sijab), bt_ref(sijab);

    btod_dirsum<2, 2>(bt1, 1.0, bt2, 1.0, permutation<4>().permute(1, 2)).
        perform(bt_ref);

    bt(i|j|a|b) = dbt(i|j|a|b);

    compare_ref<4>::compare(testname, bt, bt_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
