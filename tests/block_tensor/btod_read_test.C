#include <cmath>
#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/btod/btod_read.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/dense_tensor/tod_btconv.h>
#include "btod_read_test.h"
#include "../compare_ref.h"

namespace libtensor {


void btod_read_test::perform() {

    allocator<double>::init();

    try {

    test_1();
    test_2();
    test_3();
    test_4();
    test_5();
    test_6();
    test_7();
    test_8();
    test_9();
    test_10();

    } catch (...) {
        allocator<double>::shutdown();
        throw;
    }
    allocator<double>::shutdown();
}


void btod_read_test::test_1() {

    //
    //  Block tensor (2-dim) with one block
    //

    static const char *testname = "btod_read_test::test_1()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<2> i1, i2;
    i2[0] = 4; i2[1] = 5;
    dimensions<2> dims(index_range<2>(i1, i2));
    dense_tensor<2, double, allocator_t> t(dims), t_ref(dims);
    tod_random<2>().perform(t_ref);

    std::stringstream ss;
    ss << "2 " << dims[0] << " " << dims[1] << std::endl;

    {
        dense_tensor_ctrl<2, double> ctrl(t_ref);
        const double *p = ctrl.req_const_dataptr();
        for(size_t i = 0; i < dims[0]; i++) {
            libtensor::index<2> idx;
            idx[0] = i;
            for(size_t j = 0; j < dims[1]; j++) {
                idx[1] = j;
                abs_index<2> aidx(idx, dims);
                ss.precision(15);
                ss.setf(std::ios::fixed, std::ios::floatfield);
                ss << p[aidx.get_abs_index()] << " ";
            }
            ss << std::endl;
        }
        ctrl.ret_const_dataptr(p); p = 0;
    }

    block_index_space<2> bis(dims);
    block_tensor<2, double, allocator_t> bt(bis);
    btod_read<2>(ss).perform(bt);
    tod_btconv<2>(bt).perform(t);

    compare_ref<2>::compare(testname, t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_read_test::test_2() {

    //
    //  Block tensor (2-dim), two blocks along each dimension
    //

    static const char *testname = "btod_read_test::test_2()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<2> i1, i2;
    i2[0] = 4; i2[1] = 5;
    dimensions<2> dims(index_range<2>(i1, i2));
    dense_tensor<2, double, allocator_t> t(dims), t_ref(dims);
    tod_random<2>().perform(t_ref);

    std::stringstream ss;
    ss << "2 " << dims[0] << " " << dims[1] << std::endl;

    {
        dense_tensor_ctrl<2, double> ctrl(t_ref);
        const double *p = ctrl.req_const_dataptr();
        for(size_t i = 0; i < dims[0]; i++) {
            libtensor::index<2> idx;
            idx[0] = i;
            for(size_t j = 0; j < dims[1]; j++) {
                idx[1] = j;
                abs_index<2> aidx(idx, dims);
                ss.precision(15);
                ss.setf(std::ios::fixed, std::ios::floatfield);
                ss << p[aidx.get_abs_index()] << " ";
            }
            ss << std::endl;
        }
        ctrl.ret_const_dataptr(p); p = 0;
    }

    block_index_space<2> bis(dims);
    mask<2> msk1, msk2; msk1[0] = true; msk2[1] = true;
    bis.split(msk1, 2); bis.split(msk2, 3);
    block_tensor<2, double, allocator_t> bt(bis);
    btod_read<2>(ss).perform(bt);
    tod_btconv<2>(bt).perform(t);

    compare_ref<2>::compare(testname, t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_read_test::test_3() {

    //
    //  Block tensor (2-dim) with one zero block
    //

    static const char *testname = "btod_read_test::test_3()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<2> i1, i2;
    i2[0] = 4; i2[1] = 5;
    dimensions<2> dims(index_range<2>(i1, i2));
    dense_tensor<2, double, allocator_t> t(dims), t_ref(dims);
    tod_set<2>().perform(true, t_ref);

    std::stringstream ss;
    ss << "2 " << dims[0] << " " << dims[1] << std::endl;

    {
        dense_tensor_ctrl<2, double> ctrl(t_ref);
        const double *p = ctrl.req_const_dataptr();
        for(size_t i = 0; i < dims[0]; i++) {
            libtensor::index<2> idx;
            idx[0] = i;
            for(size_t j = 0; j < dims[1]; j++) {
                idx[1] = j;
                abs_index<2> aidx(idx, dims);
                ss.precision(15);
                ss.setf(std::ios::fixed, std::ios::floatfield);
                ss << p[aidx.get_abs_index()] << " ";
            }
            ss << std::endl;
        }
        ctrl.ret_const_dataptr(p); p = 0;
    }

    block_index_space<2> bis(dims);
    block_tensor<2, double, allocator_t> bt(bis), bt_ref(bis);
    btod_read<2>(ss).perform(bt);

    compare_ref<2>::compare(testname, bt, bt_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_read_test::test_4() {

    //
    //  Block tensor (2-dim), two blocks along each dimension,
    //  zero off-diagonal blocks
    //

    static const char *testname = "btod_read_test::test_4()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<2> i1, i2;
    i2[0] = 4; i2[1] = 5;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> msk1, msk2; msk1[0] = true; msk2[1] = true;
    bis.split(msk1, 2); bis.split(msk2, 3);

    block_tensor<2, double, allocator_t> bt(bis), bt_ref(bis);
    dense_tensor<2, double, allocator_t> t_ref(dims);
    libtensor::index<2> ii;
    btod_random<2> rand;
    rand.perform(bt_ref, ii);
    ii[0] = 1; ii[1] = 1;
    rand.perform(bt_ref, ii);
    tod_btconv<2>(bt_ref).perform(t_ref);

    std::stringstream ss;
    ss << "2 " << dims[0] << " " << dims[1] << std::endl;

    {
        dense_tensor_ctrl<2, double> ctrl(t_ref);
        const double *p = ctrl.req_const_dataptr();
        for(size_t i = 0; i < dims[0]; i++) {
            libtensor::index<2> idx;
            idx[0] = i;
            for(size_t j = 0; j < dims[1]; j++) {
                idx[1] = j;
                abs_index<2> aidx(idx, dims);
                ss.precision(15);
                ss.setf(std::ios::fixed, std::ios::floatfield);
                ss << p[aidx.get_abs_index()] << " ";
            }
            ss << std::endl;
        }
        ctrl.ret_const_dataptr(p); p = 0;
    }

    btod_read<2>(ss).perform(bt);

    compare_ref<2>::compare(testname, bt, bt_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_read_test::test_5() {

    //
    //  Block tensor (4-dim) with one block
    //

    static const char *testname = "btod_read_test::test_5()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<4> i1, i2;
    i2[0] = 4; i2[1] = 5; i2[2] = 4; i2[3] = 5;
    dimensions<4> dims(index_range<4>(i1, i2));
    dense_tensor<4, double, allocator_t> t(dims), t_ref(dims);
    tod_random<4>().perform(t_ref);

    std::stringstream ss;
    ss << "4 " << dims[0] << " " << dims[1] << " " << dims[2] << " "
        << dims[3] << std::endl;

    {
        dense_tensor_ctrl<4, double> ctrl(t_ref);
        const double *p = ctrl.req_const_dataptr();
        abs_index<4> aidx(dims);
        do {
            ss.precision(15);
            ss.setf(std::ios::fixed, std::ios::floatfield);
            ss << p[aidx.get_abs_index()] << " ";
        } while(aidx.inc());
        ctrl.ret_const_dataptr(p); p = 0;
    }

    block_index_space<4> bis(dims);
    block_tensor<4, double, allocator_t> bt(bis);
    btod_read<4>(ss).perform(bt);
    tod_btconv<4>(bt).perform(t);

    compare_ref<4>::compare(testname, t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_read_test::test_6() {

    //
    //  Block tensor (4-dim), two blocks along each dimension
    //

    static const char *testname = "btod_read_test::test_6()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<4> i1, i2;
    i2[0] = 4; i2[1] = 5; i2[2] = 4; i2[3] = 5;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> m1, m2;
    m1[0] = true; m1[2] = true;
    m2[1] = true; m2[3] = true;
    bis.split(m1, 2); bis.split(m2, 3);

    dense_tensor<4, double, allocator_t> t_ref(dims);
    block_tensor<4, double, allocator_t> bt(bis), bt_ref(bis);
    btod_random<4>().perform(bt_ref);
    tod_btconv<4>(bt_ref).perform(t_ref);

    std::stringstream ss;
    ss << "4 " << dims[0] << " " << dims[1] << " " << dims[2] << " "
        << dims[3] << std::endl;

    {
        dense_tensor_ctrl<4, double> ctrl(t_ref);
        const double *p = ctrl.req_const_dataptr();
        abs_index<4> aidx(dims);
        do {
            ss.precision(15);
            ss.setf(std::ios::fixed, std::ios::floatfield);
            ss << p[aidx.get_abs_index()] << " ";
        } while(aidx.inc());
        ctrl.ret_const_dataptr(p); p = 0;
    }

    btod_read<4>(ss).perform(bt);

    compare_ref<4>::compare(testname, bt, bt_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_read_test::test_7() {

    //
    //  Block tensor (2-dim), two blocks along each dimension,
    //  the size of each block is 1x1
    //

    static const char *testname = "btod_read_test::test_7()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<2> i1, i2;
    i2[0] = 1; i2[1] = 1;
    dimensions<2> dims(index_range<2>(i1, i2));
    dense_tensor<2, double, allocator_t> t(dims), t_ref(dims);
    tod_random<2>().perform(t_ref);

    std::stringstream ss;
    ss << "2 " << dims[0] << " " << dims[1] << std::endl;

    {
        dense_tensor_ctrl<2, double> ctrl(t_ref);
        const double *p = ctrl.req_const_dataptr();
        for(size_t i = 0; i < dims[0]; i++) {
            libtensor::index<2> idx;
            idx[0] = i;
            for(size_t j = 0; j < dims[1]; j++) {
                idx[1] = j;
                abs_index<2> aidx(idx, dims);
                ss.precision(15);
                ss.setf(std::ios::fixed, std::ios::floatfield);
                ss << p[aidx.get_abs_index()] << " ";
            }
            ss << std::endl;
        }
        ctrl.ret_const_dataptr(p); p = 0;
    }

    block_index_space<2> bis(dims);
    mask<2> msk1, msk2; msk1[0] = true; msk2[1] = true;
    bis.split(msk1, 1); bis.split(msk2, 1);
    block_tensor<2, double, allocator_t> bt(bis);
    btod_read<2>(ss).perform(bt);
    tod_btconv<2>(bt).perform(t);

    compare_ref<2>::compare(testname, t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_read_test::test_8() {

    //
    //  Block tensor (2-dim), two blocks along each dimension,
    //  the sizes of blocks are 1 and 2
    //

    static const char *testname = "btod_read_test::test_8()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<2> i1, i2;
    i2[0] = 2; i2[1] = 2;
    dimensions<2> dims(index_range<2>(i1, i2));
    dense_tensor<2, double, allocator_t> t(dims), t_ref(dims);
    tod_random<2>().perform(t_ref);

    std::stringstream ss;
    ss << "2 " << dims[0] << " " << dims[1] << std::endl;

    {
        dense_tensor_ctrl<2, double> ctrl(t_ref);
        const double *p = ctrl.req_const_dataptr();
        for(size_t i = 0; i < dims[0]; i++) {
            libtensor::index<2> idx;
            idx[0] = i;
            for(size_t j = 0; j < dims[1]; j++) {
                idx[1] = j;
                abs_index<2> aidx(idx, dims);
                ss.precision(15);
                ss.setf(std::ios::fixed, std::ios::floatfield);
                ss << p[aidx.get_abs_index()] << " ";
            }
            ss << std::endl;
        }
        ctrl.ret_const_dataptr(p); p = 0;
    }

    block_index_space<2> bis(dims);
    mask<2> msk1, msk2; msk1[0] = true; msk2[1] = true;
    bis.split(msk1, 1); bis.split(msk2, 1);
    block_tensor<2, double, allocator_t> bt(bis);
    btod_read<2>(ss).perform(bt);
    tod_btconv<2>(bt).perform(t);

    compare_ref<2>::compare(testname, t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

void btod_read_test::test_9() {

    //
    //  Block tensor (2-dim), two blocks along each dimension,
    //  the sizes of blocks are 1 and 2, permutational symmetry
    //

    static const char *testname = "btod_read_test::test_9()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<2> i1, i2;
    i2[0] = 2; i2[1] = 2;
    dimensions<2> dims(index_range<2>(i1, i2));
    dense_tensor<2, double, allocator_t> t(dims), t_ref(dims);
    tod_random<2>().perform(t_ref);

    std::stringstream ss;
    ss << "2 " << dims[0] << " " << dims[1] << std::endl;

    {
        // modify t_ref to have permutational symmetry
        dense_tensor_ctrl<2, double> ctrl(t_ref);

        double *ptr = ctrl.req_dataptr();
        for(size_t i = 0; i < dims[0]; i++) {
            libtensor::index<2> idx1, idx2;
            idx1[0] = idx2[1] = i;
            for(size_t j = i+1; j < dims[1]; j++) {
                idx1[1] = idx2[0] = j;
                abs_index<2> aidx1(idx1, dims), aidx2(idx2, dims);
                ptr[aidx2.get_abs_index()] = ptr[aidx1.get_abs_index()];
            }
        }
        ctrl.ret_dataptr(ptr);

        // write t_ref to file
        const double *p = ctrl.req_const_dataptr();
        for(size_t i = 0; i < dims[0]; i++) {
            libtensor::index<2> idx;
            idx[0] = i;
            for(size_t j = 0; j < dims[1]; j++) {
                idx[1] = j;
                abs_index<2> aidx(idx, dims);
                ss.precision(15);
                ss.setf(std::ios::fixed, std::ios::floatfield);
                ss << p[aidx.get_abs_index()] << " ";
            }
            ss << std::endl;
        }
        ctrl.ret_const_dataptr(p); p = 0;
    }

    block_index_space<2> bis(dims);
    mask<2> msk1, msk2; msk1[0] = true; msk2[1] = true;
    bis.split(msk1, 1); bis.split(msk2, 1);
    block_tensor<2, double, allocator_t> bt(bis);
    {
        block_tensor_ctrl<2, double> ctrl(bt);
        permutation<2> perm;
        perm.permute(0,1);
        scalar_transf<double> tr0, tr1(-1.);
        se_perm<2, double> sp(perm, tr0);
        ctrl.req_symmetry().insert(sp);
    }

    btod_read<2>(ss).perform(bt);
    tod_btconv<2>(bt).perform(t);

    compare_ref<2>::compare(testname, t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

void btod_read_test::test_10() {

    //
    //  Block tensor (2-dim), two blocks along each dimension,
    //  the sizes of blocks are 1 and 2, permutational symmetry (but not the reference!)
    //

    static const char *testname = "btod_read_test::test_10()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<2> i1, i2;
    i2[0] = 2; i2[1] = 2;
    dimensions<2> dims(index_range<2>(i1, i2));
    dense_tensor<2, double, allocator_t> t(dims), t_ref(dims);
    tod_random<2>().perform(t_ref);

    std::stringstream ss;
    ss << "2 " << dims[0] << " " << dims[1] << std::endl;

    {
        dense_tensor_ctrl<2, double> ctrl(t_ref);

        const double *p = ctrl.req_const_dataptr();
        for(size_t i = 0; i < dims[0]; i++) {
            libtensor::index<2> idx;
            idx[0] = i;
            for(size_t j = 0; j < dims[1]; j++) {
                idx[1] = j;
                abs_index<2> aidx(idx, dims);
                ss.precision(15);
                ss.setf(std::ios::fixed, std::ios::floatfield);
                ss << p[aidx.get_abs_index()] << " ";
            }
            ss << std::endl;
        }
        ctrl.ret_const_dataptr(p); p = 0;
    }

    block_index_space<2> bis(dims);
    mask<2> msk1, msk2; msk1[0] = true; msk2[1] = true;
    bis.split(msk1, 1); bis.split(msk2, 1);
    block_tensor<2, double, allocator_t> bt(bis);
    {
        block_tensor_ctrl<2, double> ctrl(bt);
        permutation<2> perm;
        perm.permute(0,1);
        scalar_transf<double> tr0, tr1(-1.);
        se_perm<2, double> sp(perm, tr0);
        ctrl.req_symmetry().insert(sp);
    }

    bool not_failed = true;
    try {
    btod_read<2>(ss).perform(bt);
    }
    catch (exception &e) {
        not_failed = false;
    }

    if (not_failed) {
        fail_test(testname, __FILE__, __LINE__,
                "Missing symmetry not detected.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

} // namespace libtensor
