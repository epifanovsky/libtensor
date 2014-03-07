#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod_random.h>
#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/ctf_dense_tensor/ctf_dense_tensor.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_collect.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_copy.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_distribute.h>
#include "../compare_ref.h"
#include "ctf_tod_copy_test.h"

namespace libtensor {


void ctf_tod_copy_test::perform() throw(libtest::test_exception) {

    ctf::init();

    try {

        test_1a();
        test_1b();
        test_2a();
        test_2b();
        test_3a();
        test_3b();
        test_4a();
        test_4b();

    } catch(...) {
        ctf::exit();
        throw;
    }

    ctf::exit();
}


void ctf_tod_copy_test::test_1a() {

    static const char testname[] = "ctf_tod_copy_test::test_1a()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa);
    dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb), tb_ref(dimsb);
    ctf_dense_tensor<2, double> dta(dimsa), dtb(dimsb);

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);
    tod_copy<2>(ta).perform(true, tb_ref);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);

    ctf_tod_copy<2>(dta).perform(true, dtb);
    ctf_tod_collect<2>(dtb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_copy_test::test_1b() {

    static const char testname[] = "ctf_tod_copy_test::test_1b()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa);
    dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb), tb_ref(dimsb);
    ctf_dense_tensor<2, double> dta(dimsa), dtb(dimsb);

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);
    tod_copy<2>(tb).perform(true, tb_ref);
    tod_copy<2>(ta, -0.5).perform(false, tb_ref);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);

    ctf_tod_copy<2>(dta, -0.5).perform(false, dtb);
    ctf_tod_collect<2>(dtb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_copy_test::test_2a() {

    static const char testname[] = "ctf_tod_copy_test::test_2a()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa);
    permutation<2> perma;
    perma.permute(0, 1);
    dimsb.permute(perma);
    dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb), tb_ref(dimsb);
    ctf_dense_tensor<2, double> dta(dimsa), dtb(dimsb);

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);
    tod_copy<2>(ta, perma).perform(true, tb_ref);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);

    ctf_tod_copy<2>(dta, perma).perform(true, dtb);
    ctf_tod_collect<2>(dtb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_copy_test::test_2b() {

    static const char testname[] = "ctf_tod_copy_test::test_2b()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa);
    permutation<2> perma;
    perma.permute(0, 1);
    dimsb.permute(perma);
    dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb), tb_ref(dimsb);
    ctf_dense_tensor<2, double> dta(dimsa), dtb(dimsb);

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);
    tod_copy<2>(tb).perform(true, tb_ref);
    tod_copy<2>(ta, perma, -0.5).perform(false, tb_ref);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);

    ctf_tod_copy<2>(dta, perma, -0.5).perform(false, dtb);
    ctf_tod_collect<2>(dtb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_copy_test::test_3a() {

    static const char testname[] = "ctf_tod_copy_test::test_3a()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa);
    permutation<2> perma;
    perma.permute(0, 1);
    dimsb.permute(perma);
    dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb), tb_ref(dimsb);
    ctf_dense_tensor<2, double> dta(dimsa), dtb(dimsb);

    tensor_transf<2, double> tra(perma, scalar_transf<double>(2.0));

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);
    tod_copy<2>(ta, tra).perform(true, tb_ref);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);

    ctf_tod_copy<2>(dta, tra).perform(true, dtb);
    ctf_tod_collect<2>(dtb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_copy_test::test_3b() {

    static const char testname[] = "ctf_tod_copy_test::test_3b()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa);
    permutation<2> perma;
    perma.permute(0, 1);
    dimsb.permute(perma);
    dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb), tb_ref(dimsb);
    ctf_dense_tensor<2, double> dta(dimsa), dtb(dimsb);

    tensor_transf<2, double> tra(perma, scalar_transf<double>(2.0));

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);
    tod_copy<2>(tb).perform(true, tb_ref);
    tod_copy<2>(ta, tra).perform(false, tb_ref);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);

    ctf_tod_copy<2>(dta, tra).perform(false, dtb);
    ctf_tod_collect<2>(dtb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_copy_test::test_4a() {

    static const char testname[] = "ctf_tod_copy_test::test_4a()";

    typedef std_allocator<double> allocator_t;

    try {

    index<3> i1, i2;
    i2[0] = 19; i2[1] = 9; i2[2] = 7;
    dimensions<3> dimsa(index_range<3>(i1, i2)), dimsb(dimsa);
    permutation<3> perma;
    perma.permute(0, 1).permute(1, 2);
    dimsb.permute(perma);
    dense_tensor<3, double, allocator_t> ta(dimsa), tb(dimsb), tb_ref(dimsb);
    ctf_dense_tensor<3, double> dta(dimsa), dtb(dimsb);

    tod_random<3>().perform(ta);
    tod_random<3>().perform(tb);
    tod_copy<3>(ta, perma, 0.5).perform(true, tb_ref);

    ctf_tod_distribute<3>(ta).perform(dta);
    ctf_tod_distribute<3>(tb).perform(dtb);

    ctf_tod_copy<3>(dta, perma, 0.5).perform(true, dtb);
    ctf_tod_collect<3>(dtb).perform(tb);

    compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_copy_test::test_4b() {

    static const char testname[] = "ctf_tod_copy_test::test_4b()";

    typedef std_allocator<double> allocator_t;

    try {

    index<3> i1, i2;
    i2[0] = 19; i2[1] = 9; i2[2] = 7;
    dimensions<3> dimsa(index_range<3>(i1, i2)), dimsb(dimsa);
    permutation<3> perma;
    perma.permute(0, 2).permute(1, 2);
    dimsb.permute(perma);
    dense_tensor<3, double, allocator_t> ta(dimsa), tb(dimsb), tb_ref(dimsb);
    ctf_dense_tensor<3, double> dta(dimsa), dtb(dimsb);

    tod_random<3>().perform(ta);
    tod_random<3>().perform(tb);
    tod_copy<3>(tb).perform(true, tb_ref);
    tod_copy<3>(ta, perma, -1.0).perform(false, tb_ref);

    ctf_tod_distribute<3>(ta).perform(dta);
    ctf_tod_distribute<3>(tb).perform(dtb);

    ctf_tod_copy<3>(dta, perma, -1.0).perform(false, dtb);
    ctf_tod_collect<3>(dtb).perform(tb);

    compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

