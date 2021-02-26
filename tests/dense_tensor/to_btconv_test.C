#include <cmath>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/to_add.h>
#include <libtensor/dense_tensor/to_copy.h>
#include <libtensor/dense_tensor/to_btconv.h>
#include <libtensor/dense_tensor/to_random.h>
#include <libtensor/symmetry/se_perm.h>
#include "../compare_ref.h"
#include "to_btconv_test.h"

namespace libtensor {

void to_btconv_test::perform() throw(libtest::test_exception) {
    std::cout << "Testing to_btconv_test_x<double>   ";
    to_btconv_test_x<double> t_double;
    t_double.perform();
    std::cout << "Testing to_btconv_test_x<float>   ";
    to_btconv_test_x<float> t_float;
    t_float.perform();
}

template<typename T>
void to_btconv_test_x<T>::perform() throw(libtest::test_exception) {

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
    test_11();
    test_12();

}


template<typename T>
void to_btconv_test_x<T>::test_1() throw(libtest::test_exception) {

    //
    // All zero blocks, no symmetry
    //

    static const char *testname = "to_btconv_test_x<T>::test_1()";

    typedef allocator<T> allocator_t;
    typedef dense_tensor<2, T, allocator_t> tensor_t;
    typedef dense_tensor_ctrl<2, T> tensor_ctrl_t;
    typedef block_tensor<2, T, allocator_t> block_tensor_t;
    typedef block_tensor_ctrl<2, T> block_tensor_ctrl_t;

    try {

        index<2> i1, i2;
        i2[0] = 10; i2[1] = 10;
        dimensions<2> dims(index_range<2>(i1, i2));
        block_index_space<2> bis(dims);
        mask<2> splmsk;
        splmsk[0] = true; splmsk[1] = true;
        bis.split(splmsk, 5);

        block_tensor_t bt(bis);
        block_tensor_ctrl_t btctrl(bt);

        tensor_t t(dims), t_ref(dims);

        {
            tensor_ctrl_t tctrl(t), tctrl_ref(t_ref);

            T *pt = tctrl.req_dataptr();
            T *pt_ref = tctrl_ref.req_dataptr();

            // Fill in random input, generate reference

            size_t sz = dims.get_size();
            for(size_t i = 0; i < sz; i++) {
                pt[i] = drand48();
                pt_ref[i] = 0.0;
            }

            tctrl.ret_dataptr(pt); pt = NULL;
            tctrl_ref.ret_dataptr(pt_ref); pt_ref = NULL;
        }

        t_ref.set_immutable();
        bt.set_immutable();

        // Invoke the operation

        to_btconv<2, T> op(bt);
        op.perform(t);

        // Compare the result against the reference

        compare_ref_x<2, T>::compare(testname, t, t_ref, 0.0);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


template<typename T>
void to_btconv_test_x<T>::test_2() throw(libtest::test_exception) {

    //
    // Block [0,0] is non-zero, no symmetry
    //

    static const char *testname = "to_btconv_test_x<T>::test_2()";

    typedef allocator<T> allocator_t;
    typedef dense_tensor<2, T, allocator_t> tensor_t;
    typedef dense_tensor_ctrl<2, T> tensor_ctrl_t;
    typedef block_tensor<2, T, allocator_t> block_tensor_t;
    typedef block_tensor_ctrl<2, T> block_tensor_ctrl_t;

    try {

        index<2> i1, i2;
        i2[0] = 10; i2[1] = 10;
        dimensions<2> dims(index_range<2>(i1, i2));
        block_index_space<2> bis(dims);
        mask<2> splmsk;
        splmsk[0] = true; splmsk[1] = true;
        bis.split(splmsk, 5);

        block_tensor_t bt(bis);
        block_tensor_ctrl_t btctrl(bt);

        tensor_t t(dims), t_ref(dims);

        {
            tensor_ctrl_t tctrl(t), tctrl_ref(t_ref);
            T *pt = tctrl.req_dataptr();
            T *pt_ref = tctrl_ref.req_dataptr();

            // Fill in random input, generate reference

            size_t sz = dims.get_size();
            for(size_t i = 0; i < sz; i++) {
                pt[i] = drand48();
                pt_ref[i] = 0.0;
            }

            index<2> i_00;
            index<2> istart = bis.get_block_start(i_00);
            dimensions<2> dims_00 = bis.get_block_dims(i_00);
            dense_tensor_wr_i<2, T> &blk_00 = btctrl.req_block(i_00);
            {
                dense_tensor_wr_ctrl<2, T> tctrl_00(blk_00);
                T *p_00 = tctrl_00.req_dataptr();
                abs_index<2> aii(dims_00);
                do {
                    index<2> ii(aii.get_index()), iii(istart);
                    for(size_t j = 0; j < 2; j++) iii[j] += ii[j];
                    abs_index<2> aiii(iii, dims);
                    pt_ref[aiii.get_abs_index()] =
                            p_00[aii.get_abs_index()] = drand48();
                } while(aii.inc());
                tctrl_00.ret_dataptr(p_00);
            }
            btctrl.ret_block(i_00);

            tctrl.ret_dataptr(pt); pt = NULL;
            tctrl_ref.ret_dataptr(pt_ref); pt_ref = NULL;
        }

        t_ref.set_immutable();
        bt.set_immutable();

        // Invoke the operation

        to_btconv<2, T> op(bt);
        op.perform(t);

        // Compare the result against the reference

        compare_ref_x<2, T>::compare(testname, t, t_ref, 0.0);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


template<typename T>
void to_btconv_test_x<T>::test_3() throw(libtest::test_exception) {

    //
    // Block [1,1] is non-zero, no symmetry
    //

    static const char *testname = "to_btconv_test_x<T>::test_3()";

    typedef allocator<T> allocator_t;
    typedef dense_tensor<2, T, allocator_t> tensor_t;
    typedef dense_tensor_ctrl<2, T> tensor_ctrl_t;
    typedef block_tensor<2, T, allocator_t> block_tensor_t;
    typedef block_tensor_ctrl<2, T> block_tensor_ctrl_t;

    try {

        index<2> i1, i2;
        i2[0] = 10; i2[1] = 10;
        dimensions<2> dims(index_range<2>(i1, i2));
        block_index_space<2> bis(dims);
        mask<2> splmsk;
        splmsk[0] = true; splmsk[1] = true;
        bis.split(splmsk, 5);

        block_tensor_t bt(bis);
        block_tensor_ctrl_t btctrl(bt);

        tensor_t t(dims), t_ref(dims);

        {
            tensor_ctrl_t tctrl(t), tctrl_ref(t_ref);

            T *pt = tctrl.req_dataptr();
            T *pt_ref = tctrl_ref.req_dataptr();

            // Fill in random input, generate reference

            size_t sz = dims.get_size();
            for(size_t i = 0; i < sz; i++) {
                pt[i] = drand48();
                pt_ref[i] = 0.0;
            }

            index<2> i_11;
            i_11[0] = 1; i_11[1] = 1;
            index<2> istart = bis.get_block_start(i_11);
            dimensions<2> dims_11 = bis.get_block_dims(i_11);
            dense_tensor_wr_i<2, T> &blk_11 = btctrl.req_block(i_11);
            {
                dense_tensor_wr_ctrl<2, T> tctrl_11(blk_11);
                T *p_11 = tctrl_11.req_dataptr();
                abs_index<2> aii(dims_11);
                do {
                    index<2> ii(aii.get_index()), iii(istart);
                    for(size_t j = 0; j < 2; j++) iii[j] += ii[j];
                    abs_index<2> aiii(iii, dims);
                    pt_ref[aiii.get_abs_index()] =
                            p_11[aii.get_abs_index()] = drand48();
                } while(aii.inc());
                tctrl_11.ret_dataptr(p_11);
            }
            btctrl.ret_block(i_11);

            tctrl.ret_dataptr(pt); pt = NULL;
            tctrl_ref.ret_dataptr(pt_ref); pt_ref = NULL;
        }

        t_ref.set_immutable();
        bt.set_immutable();

        // Invoke the operation

        to_btconv<2, T> op(bt);
        op.perform(t);

        // Compare the result against the reference

        compare_ref_x<2, T>::compare(testname, t, t_ref, 0.0);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


template<typename T>
void to_btconv_test_x<T>::test_4() throw(libtest::test_exception) {

    //
    // Diagonal blocks are non-zero, no symmetry
    //

    static const char *testname = "to_btconv_test_x<T>::test_4()";

    typedef allocator<T> allocator_t;
    typedef dense_tensor<2, T, allocator_t> tensor_t;
    typedef dense_tensor_ctrl<2, T> tensor_ctrl_t;
    typedef block_tensor<2, T, allocator_t> block_tensor_t;
    typedef block_tensor_ctrl<2, T> block_tensor_ctrl_t;

    try {

        index<2> i1, i2;
        i2[0] = 10; i2[1] = 10;
        dimensions<2> dims(index_range<2>(i1, i2));
        block_index_space<2> bis(dims);
        mask<2> splmsk;
        splmsk[0] = true; splmsk[1] = true;
        bis.split(splmsk, 3);

        block_tensor_t bt(bis);
        block_tensor_ctrl_t btctrl(bt);

        tensor_t t(dims), t_ref(dims);

        {
            tensor_ctrl_t tctrl(t), tctrl_ref(t_ref);

            T *pt = tctrl.req_dataptr();
            T *pt_ref = tctrl_ref.req_dataptr();

            // Fill in random input, generate reference

            size_t sz = dims.get_size();
            for(size_t i = 0; i < sz; i++) {
                pt[i] = drand48();
                pt_ref[i] = 0.0;
            }

            index<2> i_00, i_11;
            i_11[0] = 1; i_11[1] = 1;
            index<2> istart_00 = bis.get_block_start(i_00);
            index<2> istart_11 = bis.get_block_start(i_11);
            dimensions<2> dims_00 = bis.get_block_dims(i_00);
            dimensions<2> dims_11 = bis.get_block_dims(i_11);
            T *p = NULL;

            dense_tensor_wr_i<2, T> &blk_00 = btctrl.req_block(i_00);
            {
                dense_tensor_wr_ctrl<2, T> tctrl_00(blk_00);
                p = tctrl_00.req_dataptr();
                abs_index<2> aii(dims_00);
                do {
                    index<2> ii(aii.get_index()), iii(istart_00);
                    for(size_t j = 0; j < 2; j++) iii[j] += ii[j];
                    abs_index<2> aiii(iii, dims);
                    pt_ref[aiii.get_abs_index()] = p[aii.get_abs_index()] =
                            drand48();
                } while(aii.inc());
                tctrl_00.ret_dataptr(p);
            }
            btctrl.ret_block(i_00);

            dense_tensor_wr_i<2, T> &blk_11 = btctrl.req_block(i_11);
            {
                dense_tensor_wr_ctrl<2, T> tctrl_11(blk_11);
                p = tctrl_11.req_dataptr();
                abs_index<2> aii(dims_11);
                do {
                    index<2> ii(aii.get_index()), iii(istart_11);
                    for(size_t j = 0; j < 2; j++) iii[j] += ii[j];
                    abs_index<2> aiii(iii, dims);
                    pt_ref[aiii.get_abs_index()] = p[aii.get_abs_index()] =
                            drand48();
                } while(aii.inc());
                tctrl_11.ret_dataptr(p);
            }
            btctrl.ret_block(i_11);

            tctrl.ret_dataptr(pt); pt = NULL;
            tctrl_ref.ret_dataptr(pt_ref); pt_ref = NULL;
        }

        t_ref.set_immutable();
        bt.set_immutable();

        // Invoke the operation

        to_btconv<2, T> op(bt);
        op.perform(t);

        // Compare the result against the reference

        compare_ref_x<2, T>::compare(testname, t, t_ref, 0.0);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


template<typename T>
void to_btconv_test_x<T>::test_5() throw(libtest::test_exception) {

    //
    // Diagonal blocks are non-zero, permutational symmetry
    //

    static const char *testname = "to_btconv_test_x<T>::test_5()";

    typedef allocator<T> allocator_t;
    typedef dense_tensor<2, T, allocator_t> tensor_t;
    typedef dense_tensor_ctrl<2, T> tensor_ctrl_t;
    typedef block_tensor<2, T, allocator_t> block_tensor_t;
    typedef block_tensor_ctrl<2, T> block_tensor_ctrl_t;

    try {

        index<2> i1, i2;
        i2[0] = 10; i2[1] = 10;
        dimensions<2> dims(index_range<2>(i1, i2));
        block_index_space<2> bis(dims);
        mask<2> splmsk;
        splmsk[0] = true; splmsk[1] = true;
        bis.split(splmsk, 3);

        block_tensor_t bt(bis);
        block_tensor_ctrl_t btctrl(bt);

        permutation<2> perm1; perm1.permute(0, 1);
        scalar_transf<T> tr0;
        se_perm<2, T> cycle(perm1, tr0);
        btctrl.req_symmetry().insert(cycle);

        tensor_t t(dims), t_ref(dims);

        {
            tensor_ctrl_t tctrl(t), tctrl_ref(t_ref);

            T *pt = tctrl.req_dataptr();
            T *pt_ref = tctrl_ref.req_dataptr();

            // Fill in random input, generate reference

            size_t sz = dims.get_size();
            for(size_t i = 0; i < sz; i++) {
                pt[i] = drand48();
                pt_ref[i] = 0.0;
            }

            index<2> i_00, i_11;
            index<2> istart_00 = bis.get_block_start(i_00);
            index<2> istart_11 = bis.get_block_start(i_11);
            dimensions<2> dims_00 = bis.get_block_dims(i_00);
            dimensions<2> dims_11 = bis.get_block_dims(i_11);
            T *p = NULL;
            permutation<2> perm; perm.permute(0, 1);

            dense_tensor_wr_i<2, T> &blk_00 = btctrl.req_block(i_00);
            {
                dense_tensor_wr_ctrl<2, T> tctrl_00(blk_00);
                p = tctrl_00.req_dataptr();
                abs_index<2> aii(dims_00);
                do {
                    index<2> ii(aii.get_index());
                    if(ii[0] > ii[1]) continue;
                    index<2> ii1(ii), ii2(ii); ii2.permute(perm);
                    index<2> iii1(istart_00), iii2(istart_00);
                    for(size_t j = 0; j < 2; j++) {
                        iii1[j] += ii1[j];
                        iii2[j] += ii2[j];
                    }
                    T d = drand48();
                    abs_index<2> aii1(ii1, dims_00), aii2(ii2, dims_00);
                    abs_index<2> aiii1(iii1, dims), aiii2(iii2, dims);
                    pt_ref[aiii1.get_abs_index()] = pt_ref[aiii2.get_abs_index()] = d;
                    p[aii1.get_abs_index()] = p[aii2.get_abs_index()] = d;
                } while(aii.inc());
                tctrl_00.ret_dataptr(p);
            }
            btctrl.ret_block(i_00);

            dense_tensor_wr_i<2, T> &blk_11 = btctrl.req_block(i_11);
            {
                dense_tensor_wr_ctrl<2, T> tctrl_11(blk_11);
                p = tctrl_11.req_dataptr();
                abs_index<2> aii(dims_11);
                do {
                    index<2> ii(aii.get_index());
                    if(ii[0] > ii[1]) continue;
                    index<2> ii1(ii), ii2(ii); ii2.permute(perm);
                    index<2> iii1(istart_11), iii2(istart_11);
                    for(size_t j = 0; j < 2; j++) {
                        iii1[j] += ii1[j];
                        iii2[j] += ii2[j];
                    }
                    abs_index<2> aii1(ii1, dims_11), aii2(ii2, dims_11);
                    abs_index<2> aiii1(iii1, dims), aiii2(iii2, dims);
                    pt_ref[aiii1.get_abs_index()] =
                            p[aii1.get_abs_index()] =
                                    pt_ref[aiii2.get_abs_index()] =
                                            p[aii2.get_abs_index()] = drand48();
                } while(aii.inc());
                tctrl_11.ret_dataptr(p);
            }
            btctrl.ret_block(i_11);

            tctrl.ret_dataptr(pt); pt = NULL;
            tctrl_ref.ret_dataptr(pt_ref); pt_ref = NULL;
        }

        t_ref.set_immutable();
        bt.set_immutable();

        // Invoke the operation

        to_btconv<2, T> op(bt);
        op.perform(t);

        // Compare the result against the reference

        compare_ref_x<2, T>::compare(testname, t, t_ref, 0.0);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


template<typename T>
void to_btconv_test_x<T>::test_6() throw(libtest::test_exception) {

    //
    // Off-diagonal blocks are non-zero, no symmetry
    //

    static const char *testname = "to_btconv_test_x<T>::test_6()";

    typedef allocator<T> allocator_t;
    typedef dense_tensor<2, T, allocator_t> tensor_t;
    typedef dense_tensor_ctrl<2, T> tensor_ctrl_t;
    typedef block_tensor<2, T, allocator_t> block_tensor_t;
    typedef block_tensor_ctrl<2, T> block_tensor_ctrl_t;

    try {

        index<2> i1, i2;
        i2[0] = 10; i2[1] = 10;
        dimensions<2> dims(index_range<2>(i1, i2));
        block_index_space<2> bis(dims);
        mask<2> splmsk;
        splmsk[0] = true; splmsk[1] = true;
        bis.split(splmsk, 3);

        block_tensor_t bt(bis);
        block_tensor_ctrl_t btctrl(bt);

        tensor_t t(dims), t_ref(dims);

        {
            tensor_ctrl_t tctrl(t), tctrl_ref(t_ref);

            T *pt = tctrl.req_dataptr();
            T *pt_ref = tctrl_ref.req_dataptr();

            // Fill in random input, generate reference

            size_t sz = dims.get_size();
            for(size_t i = 0; i < sz; i++) {
                pt[i] = drand48();
                pt_ref[i] = 0.0;
            }

            index<2> i_01, i_10;
            i_01[0] = 0; i_01[1] = 1;
            i_10[0] = 1; i_10[1] = 1;
            index<2> istart_01 = bis.get_block_start(i_01);
            index<2> istart_10 = bis.get_block_start(i_10);
            dimensions<2> dims_01 = bis.get_block_dims(i_01);
            dimensions<2> dims_10 = bis.get_block_dims(i_10);
            T *p = NULL;
            permutation<2> perm; perm.permute(0, 1);

            dense_tensor_wr_i<2, T> &blk_01 = btctrl.req_block(i_01);
            {
                dense_tensor_wr_ctrl<2, T> tctrl_01(blk_01);
                p = tctrl_01.req_dataptr();
                abs_index<2> aii(dims_01);
                do {
                    index<2> ii(aii.get_index()), iii(istart_01);
                    for(size_t j = 0; j < 2; j++) iii[j] += ii[j];
                    abs_index<2> aiii(iii, dims);
                    pt_ref[aiii.get_abs_index()] = p[aii.get_abs_index()] = drand48();
                } while(aii.inc());
                tctrl_01.ret_dataptr(p);
            }
            btctrl.ret_block(i_01);

            dense_tensor_wr_i<2, T> &blk_10 = btctrl.req_block(i_10);
            {
                dense_tensor_wr_ctrl<2, T> tctrl_10(blk_10);
                p = tctrl_10.req_dataptr();
                abs_index<2> aii(dims_10);
                do {
                    index<2> ii(aii.get_index()), iii(istart_10);
                    for(size_t j = 0; j < 2; j++) iii[j] += ii[j];
                    abs_index<2> aiii(iii, dims);
                    pt_ref[aiii.get_abs_index()] = p[aii.get_abs_index()] = drand48();
                } while(aii.inc());
                tctrl_10.ret_dataptr(p);
            }
            btctrl.ret_block(i_10);

            tctrl.ret_dataptr(pt); pt = NULL;
            tctrl_ref.ret_dataptr(pt_ref); pt_ref = NULL;
        }

        t_ref.set_immutable();
        bt.set_immutable();

        // Invoke the operation

        to_btconv<2, T> op(bt);
        op.perform(t);

        // Compare the result against the reference

        compare_ref_x<2, T>::compare(testname, t, t_ref, 0.0);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


template<typename T>
void to_btconv_test_x<T>::test_7() throw(libtest::test_exception) {

    //
    // Off-diagonal blocks are non-zero, permutational symmetry
    //

    static const char *testname = "to_btconv_test_x<T>::test_7()";

    typedef allocator<T> allocator_t;
    typedef dense_tensor<2, T, allocator_t> tensor_t;
    typedef dense_tensor_ctrl<2, T> tensor_ctrl_t;
    typedef block_tensor<2, T, allocator_t> block_tensor_t;
    typedef block_tensor_ctrl<2, T> block_tensor_ctrl_t;

    try {

        index<2> i1, i2;
        i2[0] = 10; i2[1] = 10;
        dimensions<2> dims(index_range<2>(i1, i2));
        block_index_space<2> bis(dims);
        mask<2> splmsk;
        splmsk[0] = true; splmsk[1] = true;
        bis.split(splmsk, 3);

        block_tensor_t bt(bis);
        block_tensor_ctrl_t btctrl(bt);

        permutation<2> perm1; perm1.permute(0, 1);
        scalar_transf<T> tr0;
        se_perm<2, T> cycle(perm1, tr0);
        btctrl.req_symmetry().insert(cycle);

        tensor_t t(dims), t_ref(dims);

        {
            tensor_ctrl_t tctrl(t), tctrl_ref(t_ref);

            T *pt = tctrl.req_dataptr();
            T *pt_ref = tctrl_ref.req_dataptr();

            // Fill in random input, generate reference

            size_t sz = dims.get_size();
            for(size_t i = 0; i < sz; i++) {
                pt[i] = drand48();
                pt_ref[i] = 0.0;
            }

            index<2> i_01, i_10;
            i_01[0] = 0; i_01[1] = 1;
            i_10[0] = 1; i_10[1] = 0;
            index<2> istart_01 = bis.get_block_start(i_01);
            index<2> istart_10 = bis.get_block_start(i_10);
            dimensions<2> dims_01 = bis.get_block_dims(i_01);
            T *p = NULL;
            permutation<2> perm; perm.permute(0, 1);

            dense_tensor_wr_i<2, T> &blk_01 = btctrl.req_block(i_01);
            {
                dense_tensor_wr_ctrl<2, T> tctrl_01(blk_01);
                p = tctrl_01.req_dataptr();
                abs_index<2> aii(dims_01);
                do {
                    index<2> ii(aii.get_index());
                    index<2> iii1(istart_01), iii2(istart_10);
                    index<2> ii2(ii); ii2.permute(perm);
                    for(size_t j = 0; j < 2; j++) {
                        iii1[j] += ii[j];
                        iii2[j] += ii2[j];
                    }
                    abs_index<2> aiii1(iii1, dims), aiii2(iii2, dims);
                    pt_ref[aiii1.get_abs_index()] =
                            pt_ref[aiii2.get_abs_index()] =
                                    p[aii.get_abs_index()] = drand48();
                } while(aii.inc());
                tctrl_01.ret_dataptr(p);
            }
            btctrl.ret_block(i_01);

            tctrl.ret_dataptr(pt); pt = NULL;
            tctrl_ref.ret_dataptr(pt_ref); pt_ref = NULL;
        }

        t_ref.set_immutable();
        bt.set_immutable();

        // Invoke the operation

        to_btconv<2, T> op(bt);
        op.perform(t);

        // Compare the result against the reference

        compare_ref_x<2, T>::compare(testname, t, t_ref, 0.0);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


template<typename T>
void to_btconv_test_x<T>::test_8() throw(libtest::test_exception) {

    //
    // All blocks are non-zero, permutational symmetry
    //

    static const char *testname = "to_btconv_test_x<T>::test_8()";

    typedef allocator<T> allocator_t;
    typedef dense_tensor<2, T, allocator_t> tensor_t;
    typedef dense_tensor_ctrl<2, T> tensor_ctrl_t;
    typedef block_tensor<2, T, allocator_t> block_tensor_t;
    typedef block_tensor_ctrl<2, T> block_tensor_ctrl_t;

    try {

        index<2> i1, i2;
        i2[0] = 10; i2[1] = 10;
        dimensions<2> dims(index_range<2>(i1, i2));
        block_index_space<2> bis(dims);
        mask<2> splmsk;
        splmsk[0] = true; splmsk[1] = true;
        bis.split(splmsk, 3);

        block_tensor_t bt(bis);
        block_tensor_ctrl_t btctrl(bt);

        permutation<2> perm1; perm1.permute(0, 1);
        scalar_transf<T> tr0;
        se_perm<2, T> cycle(perm1, tr0);
        btctrl.req_symmetry().insert(cycle);

        tensor_t t(dims), t_ref(dims);

        {
            tensor_ctrl_t tctrl(t), tctrl_ref(t_ref);

            T *pt = tctrl.req_dataptr();
            T *pt_ref = tctrl_ref.req_dataptr();

            // Fill in random input, generate reference

            size_t sz = dims.get_size();
            for(size_t i = 0; i < sz; i++) {
                pt[i] = drand48();
                pt_ref[i] = 0.0;
            }

            index<2> i_00, i_01, i_10, i_11;
            i_01[0] = 0; i_01[1] = 1;
            i_10[0] = 1; i_10[1] = 0;
            i_11[0] = 1; i_11[1] = 1;
            index<2> istart_00 = bis.get_block_start(i_00);
            index<2> istart_01 = bis.get_block_start(i_01);
            index<2> istart_10 = bis.get_block_start(i_10);
            index<2> istart_11 = bis.get_block_start(i_11);
            dimensions<2> dims_00 = bis.get_block_dims(i_00);
            dimensions<2> dims_01 = bis.get_block_dims(i_01);
            dimensions<2> dims_11 = bis.get_block_dims(i_11);
            T *p = NULL;
            permutation<2> perm; perm.permute(0, 1);

            dense_tensor_wr_i<2, T> &blk_00 = btctrl.req_block(i_00);
            {
                dense_tensor_wr_ctrl<2, T> tctrl_00(blk_00);
                p = tctrl_00.req_dataptr();
                abs_index<2> aii(dims_00);
                do {
                    index<2> ii(aii.get_index());
                    if(ii[0] > ii[1]) continue;
                    index<2> ii1(ii), ii2(ii); ii2.permute(perm);
                    index<2> iii1(istart_00), iii2(istart_00);
                    for(size_t j = 0; j < 2; j++) {
                        iii1[j] += ii1[j];
                        iii2[j] += ii2[j];
                    }
                    abs_index<2> aii1(ii1, dims_00), aii2(ii2, dims_00);
                    abs_index<2> aiii1(iii1, dims), aiii2(iii2, dims);
                    pt_ref[aiii1.get_abs_index()] =
                            p[aii1.get_abs_index()] =
                                    pt_ref[aiii2.get_abs_index()] =
                                            p[aii2.get_abs_index()] = drand48();
                } while(aii.inc());
                tctrl_00.ret_dataptr(p);
            }
            btctrl.ret_block(i_00);

            dense_tensor_wr_i<2, T> &blk_01 = btctrl.req_block(i_01);
            {
                dense_tensor_wr_ctrl<2, T> tctrl_01(blk_01);
                p = tctrl_01.req_dataptr();
                abs_index<2> aii(dims_01);
                do {
                    index<2> ii(aii.get_index());
                    index<2> iii1(istart_01), iii2(istart_10);
                    index<2> ii2(ii); ii2.permute(perm);
                    for(size_t j = 0; j < 2; j++) {
                        iii1[j] += ii[j];
                        iii2[j] += ii2[j];
                    }
                    abs_index<2> aiii1(iii1, dims), aiii2(iii2, dims);
                    pt_ref[aiii1.get_abs_index()] =
                            pt_ref[aiii2.get_abs_index()] =
                                    p[aii.get_abs_index()] = drand48();
                } while(aii.inc());
                tctrl_01.ret_dataptr(p);
            }
            btctrl.ret_block(i_01);

            dense_tensor_wr_i<2, T> &blk_11 = btctrl.req_block(i_11);
            {
                dense_tensor_wr_ctrl<2, T> tctrl_11(blk_11);
                p = tctrl_11.req_dataptr();
                abs_index<2> aii(dims_11);
                do {
                    index<2> ii(aii.get_index());
                    if(ii[0] > ii[1]) continue;
                    index<2> ii1(ii), ii2(ii); ii2.permute(perm);
                    index<2> iii1(istart_11), iii2(istart_11);
                    for(size_t j = 0; j < 2; j++) {
                        iii1[j] += ii1[j];
                        iii2[j] += ii2[j];
                    }
                    abs_index<2> aii1(ii1, dims_11), aii2(ii2, dims_11);
                    abs_index<2> aiii1(iii1, dims), aiii2(iii2, dims);
                    pt_ref[aiii1.get_abs_index()] =
                            p[aii1.get_abs_index()] =
                                    pt_ref[aiii2.get_abs_index()] =
                                            p[aii2.get_abs_index()] = drand48();
                } while(aii.inc());
                tctrl_11.ret_dataptr(p);
            }
            btctrl.ret_block(i_11);

            tctrl.ret_dataptr(pt); pt = NULL;
            tctrl_ref.ret_dataptr(pt_ref); pt_ref = NULL;
        }

        t_ref.set_immutable();
        bt.set_immutable();

        // Invoke the operation

        to_btconv<2, T> op(bt);
        op.perform(t);

        // Compare the result against the reference

        compare_ref_x<2, T>::compare(testname, t, t_ref, 0.0);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


template<typename T>
void to_btconv_test_x<T>::test_9() throw(libtest::test_exception) {

    //
    // Fully symmetric four-index tensor, one non-zero block
    //

    static const char *testname = "to_btconv_test_x<T>::test_9()";

    typedef allocator<T> allocator_t;
    typedef dense_tensor<4, T, allocator_t> tensor_t;
    typedef dense_tensor_ctrl<4, T> tensor_ctrl_t;
    typedef block_tensor<4, T, allocator_t> block_tensor_t;
    typedef block_tensor_ctrl<4, T> block_tensor_ctrl_t;

    try {

        index<4> i1, i2;
        i2[0] = 10; i2[1] = 10; i2[2] = 10; i2[3] = 10;
        dimensions<4> dims(index_range<4>(i1, i2));
        block_index_space<4> bis(dims);
        mask<4> splmsk;
        splmsk[0] = true; splmsk[1] = true; splmsk[2] = true; splmsk[3] = true;
        bis.split(splmsk, 3);

        block_tensor_t bt(bis);
        block_tensor_ctrl_t btctrl(bt);

        permutation<4> cperm1, cperm2;
        cperm1.permute(0, 1).permute(1, 2).permute(2, 3);
        cperm2.permute(0, 1);
        scalar_transf<T> tr0;
        se_perm<4, T> cycle1(cperm1, tr0);
        se_perm<4, T> cycle2(cperm2, tr0);
        btctrl.req_symmetry().insert(cycle1);
        btctrl.req_symmetry().insert(cycle2);

        tensor_t t(dims), t_ref(dims);

        {
            tensor_ctrl_t tctrl(t), tctrl_ref(t_ref);

            T *pt = tctrl.req_dataptr();
            T *pt_ref = tctrl_ref.req_dataptr();

            // Fill in random input, generate reference

            size_t sz = dims.get_size();
            for(size_t i = 0; i < sz; i++) {
                pt[i] = drand48();
                pt_ref[i] = 0.0;
            }

            index<4> i_0001, i_0010, i_0100, i_1000;
            i_0001[0] = 0; i_0001[1] = 0; i_0001[2] = 0; i_0001[3] = 1;
            i_0010[0] = 0; i_0010[1] = 0; i_0010[2] = 1; i_0010[3] = 0;
            i_0100[0] = 0; i_0100[1] = 1; i_0100[2] = 0; i_0100[3] = 0;
            i_1000[0] = 1; i_1000[1] = 0; i_1000[2] = 0; i_1000[3] = 0;
            index<4> istart_0001 = bis.get_block_start(i_0001);
            dimensions<4> dims_0001 = bis.get_block_dims(i_0001);
            T *p = NULL;
            permutation<4> perm; perm.permute(0, 1).permute(1, 2).permute(2, 3);
            permutation<4> perm1, perm2, perm3, perm4, perm5;
            perm1.permute(0, 1);
            perm2.permute(0, 2);
            perm3.permute(1, 2);
            perm4.permute(0, 1).permute(1, 2);
            perm5.permute(1, 2).permute(0 ,1);

            dense_tensor_wr_i<4, T> &blk_0001 = btctrl.req_block(i_0001);
            {
                dense_tensor_wr_ctrl<4, T> tctrl_0001(blk_0001);
                p = tctrl_0001.req_dataptr();

                abs_index<4> aii(dims_0001);
                do {
                    index<4> ii(aii.get_index());
                    if(ii[0] > ii[1] || ii[1] > ii[2]) continue;
                    index<4> ii1(ii), ii2(ii), ii3(ii), ii4(ii), ii5(ii);
                    ii1.permute(perm1);
                    ii2.permute(perm2);
                    ii3.permute(perm3);
                    ii4.permute(perm4);
                    ii5.permute(perm5);
                    T d = drand48();
                    index<4> iii(istart_0001);
                    index<4> iii0, iii1, iii2, iii3, iii4, iii5;
                    for(size_t k = 0; k < 4; k++) {
                        iii0[k] = iii[k] + ii[k];
                        iii1[k] = iii[k] + ii1[k];
                        iii2[k] = iii[k] + ii2[k];
                        iii3[k] = iii[k] + ii3[k];
                        iii4[k] = iii[k] + ii4[k];
                        iii5[k] = iii[k] + ii5[k];
                    }
                    for(size_t j = 0; j < 4; j++) {
                        pt_ref[abs_index<4>::get_abs_index(iii0, dims)] = d;
                        pt_ref[abs_index<4>::get_abs_index(iii1, dims)] = d;
                        pt_ref[abs_index<4>::get_abs_index(iii2, dims)] = d;
                        pt_ref[abs_index<4>::get_abs_index(iii3, dims)] = d;
                        pt_ref[abs_index<4>::get_abs_index(iii4, dims)] = d;
                        pt_ref[abs_index<4>::get_abs_index(iii5, dims)] = d;
                        iii0.permute(perm);
                        iii1.permute(perm);
                        iii2.permute(perm);
                        iii3.permute(perm);
                        iii4.permute(perm);
                        iii5.permute(perm);
                    }
                    p[aii.get_abs_index()] =
                            p[abs_index<4>::get_abs_index(ii1, dims_0001)] =
                                    p[abs_index<4>::get_abs_index(ii2, dims_0001)] =
                                            p[abs_index<4>::get_abs_index(ii3, dims_0001)] =
                                                    p[abs_index<4>::get_abs_index(ii4, dims_0001)] =
                                                            p[abs_index<4>::get_abs_index(ii5, dims_0001)] = d;
                } while(aii.inc());
                tctrl_0001.ret_dataptr(p);
            }
            btctrl.ret_block(i_0001);

            tctrl.ret_dataptr(pt); pt = NULL;
            tctrl_ref.ret_dataptr(pt_ref); pt_ref = NULL;
        }

        t_ref.set_immutable();
        bt.set_immutable();

        // Invoke the operation

        to_btconv<4, T> op(bt);
        op.perform(t);

        // Compare the result against the reference

        compare_ref_x<4, T>::compare(testname, t, t_ref, 0.0);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


template<typename T>
void to_btconv_test_x<T>::test_10() throw(libtest::test_exception) {

    //
    // Non-symmetric four-index tensor, one non-zero block
    //

    static const char *testname = "to_btconv_test_x<T>::test_10()";

    typedef allocator<T> allocator_t;
    typedef dense_tensor<4, T, allocator_t> tensor_t;
    typedef dense_tensor_ctrl<4, T> tensor_ctrl_t;
    typedef block_tensor<4, T, allocator_t> block_tensor_t;
    typedef block_tensor_ctrl<4, T> block_tensor_ctrl_t;

    try {

        index<4> i1, i2;
        i2[0] = 10; i2[1] = 10; i2[2] = 10; i2[3] = 10;
        dimensions<4> dims(index_range<4>(i1, i2));
        block_index_space<4> bis(dims);
        mask<4> splmsk;
        splmsk[0] = true; splmsk[1] = true; splmsk[2] = true; splmsk[3] = true;
        bis.split(splmsk, 3);

        block_tensor_t bt(bis);
        block_tensor_ctrl_t btctrl(bt);

        tensor_t t(dims), t_ref(dims);

        {
            tensor_ctrl_t tctrl(t), tctrl_ref(t_ref);

            T *pt = tctrl.req_dataptr();
            T *pt_ref = tctrl_ref.req_dataptr();

            // Fill in random input, generate reference

            size_t sz = dims.get_size();
            for(size_t i = 0; i < sz; i++) {
                pt[i] = drand48();
                pt_ref[i] = 0.0;
            }

            index<4> i_0001, i_0010, i_0100, i_1000;
            i_0001[0] = 0; i_0001[1] = 0; i_0001[2] = 0; i_0001[3] = 1;
            i_0010[0] = 0; i_0010[1] = 0; i_0010[2] = 1; i_0010[3] = 0;
            i_0100[0] = 0; i_0100[1] = 1; i_0100[2] = 0; i_0100[3] = 0;
            i_1000[0] = 1; i_1000[1] = 0; i_1000[2] = 0; i_1000[3] = 0;
            index<4> istart_0010 = bis.get_block_start(i_0010);
            dimensions<4> dims_0010 = bis.get_block_dims(i_0010);
            T *p = NULL;
            permutation<4> perm; perm.permute(0, 2);

            dense_tensor_wr_i<4, T> &blk_0010 = btctrl.req_block(i_0010);
            {
                dense_tensor_wr_ctrl<4, T> tctrl_0010(blk_0010);
                p = tctrl_0010.req_dataptr();

                abs_index<4> aii(dims_0010);
                do {
                    index<4> ii(aii.get_index()), iii(istart_0010);
                    index<4> iii0;
                    for(size_t k = 0; k < 4; k++) iii0[k] = iii[k] + ii[k];
                    T d = drand48();
                    abs_index<4> aiii0(iii0, dims);
                    p[aii.get_abs_index()] = pt_ref[aiii0.get_abs_index()] = d;
                } while(aii.inc());
                tctrl_0010.ret_dataptr(p);
            }
            btctrl.ret_block(i_0010);

            tctrl.ret_dataptr(pt); pt = NULL;
            tctrl_ref.ret_dataptr(pt_ref); pt_ref = NULL;
        }

        t_ref.set_immutable();
        bt.set_immutable();

        // Invoke the operation

        to_btconv<4, T> op(bt);
        op.perform(t);

        // Compare the result against the reference

        compare_ref_x<4, T>::compare(testname, t, t_ref, 0.0);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


template<typename T>
void to_btconv_test_x<T>::test_11() throw(libtest::test_exception) {

    //
    // Pairwise-symmetric four-index tensor, one non-zero block
    //

    static const char *testname = "to_btconv_test_x<T>::test_11()";

    typedef allocator<T> allocator_t;
    typedef dense_tensor<4, T, allocator_t> tensor_t;
    typedef dense_tensor_ctrl<4, T> tensor_ctrl_t;
    typedef block_tensor<4, T, allocator_t> block_tensor_t;
    typedef block_tensor_ctrl<4, T> block_tensor_ctrl_t;

    try {

        index<4> i1, i2;
        i2[0] = 10; i2[1] = 9; i2[2] = 10; i2[3] = 9;
        dimensions<4> dims(index_range<4>(i1, i2));
        block_index_space<4> bis(dims);
        mask<4> spl1, spl2;
        spl1[0] = true; spl1[1] = false; spl1[2] = true; spl1[3] = false;
        spl2[0] = false; spl2[1] = true; spl2[2] = false; spl2[3] = true;
        bis.split(spl1, 3);
        bis.split(spl2, 4);

        block_tensor_t bt(bis);
        block_tensor_ctrl_t btctrl(bt);

        permutation<4> cperm1, cperm2;
        cperm1.permute(0, 2);
        cperm2.permute(1, 3);
        scalar_transf<T> tr0;
        se_perm<4, T> cycle1(cperm1, tr0);
        se_perm<4, T> cycle2(cperm2, tr0);
        btctrl.req_symmetry().insert(cycle1);
        btctrl.req_symmetry().insert(cycle2);

        tensor_t t(dims), t_ref(dims);

        {
            tensor_ctrl_t tctrl(t), tctrl_ref(t_ref);

            T *pt = tctrl.req_dataptr();
            T *pt_ref = tctrl_ref.req_dataptr();

            // Fill in random input, generate reference

            size_t sz = dims.get_size();
            for(size_t i = 0; i < sz; i++) {
                pt[i] = drand48();
                pt_ref[i] = 0.0;
            }

            index<4> i_0001, i_0010, i_0100, i_1000;
            i_0001[0] = 0; i_0001[1] = 0; i_0001[2] = 0; i_0001[3] = 1;
            i_0010[0] = 0; i_0010[1] = 0; i_0010[2] = 1; i_0010[3] = 0;
            i_0100[0] = 0; i_0100[1] = 1; i_0100[2] = 0; i_0100[3] = 0;
            i_1000[0] = 1; i_1000[1] = 0; i_1000[2] = 0; i_1000[3] = 0;
            index<4> istart_0010 = bis.get_block_start(i_0010);
            dimensions<4> dims_0010 = bis.get_block_dims(i_0010);
            T *p = NULL;
            permutation<4> perm; perm.permute(0, 2);
            permutation<4> perm1; perm1.permute(1, 3);

            dense_tensor_wr_i<4, T> &blk_0010 = btctrl.req_block(i_0010);
            {
                dense_tensor_wr_ctrl<4, T> tctrl_0010(blk_0010);
                p = tctrl_0010.req_dataptr();

                abs_index<4> aii(dims_0010);
                do {
                    index<4> ii(aii.get_index());
                    if(ii[1] > ii[3]) continue;
                    index<4> ii1(ii);
                    ii1.permute(perm1);
                    T d = drand48();
                    index<4> iii(istart_0010);
                    index<4> iii0, iii1;
                    for(size_t k = 0; k < 4; k++) {
                        iii0[k] = iii[k] + ii[k];
                        iii1[k] = iii[k] + ii1[k];
                    }
                    for(size_t k = 0; k < 2; k++) {
                        pt_ref[abs_index<4>::get_abs_index(iii0, dims)] = d;
                        pt_ref[abs_index<4>::get_abs_index(iii1, dims)] = d;
                        iii0.permute(perm);
                        iii1.permute(perm);
                    }
                    abs_index<4> aii1(ii1, dims_0010);
                    p[aii.get_abs_index()] = p[aii1.get_abs_index()] = d;
                } while(aii.inc());
                tctrl_0010.ret_dataptr(p);
            }
            btctrl.ret_block(i_0010);

            tctrl.ret_dataptr(pt); pt = NULL;
            tctrl_ref.ret_dataptr(pt_ref); pt_ref = NULL;
        }

        t_ref.set_immutable();
        bt.set_immutable();

        // Invoke the operation

        to_btconv<4, T> op(bt);
        op.perform(t);

        // Compare the result against the reference

        compare_ref_x<4, T>::compare(testname, t, t_ref, 0.0);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


template<typename T>
void to_btconv_test_x<T>::test_12() throw(libtest::test_exception) {

    //
    // Anti-symmetric three-index tensor. Three non-zero blocks:
    // [0,1,2], [0,2,2], [1,1,1]
    //

    static const char *testname = "to_btconv_test_x<T>::test_12()";

    typedef allocator<T> allocator_t;

    try {

        index<3> i1, i2;
        i2[0] = 10; i2[1] = 10; i2[2] = 10;
        dimensions<3> dims(index_range<3>(i1, i2));
        block_index_space<3> bis(dims);
        mask<3> m;
        m[0] = true; m[1] = true; m[2] = true;
        bis.split(m, 3);
        bis.split(m, 8);

        block_tensor<3, T, allocator_t> bta(bis), btb(bis);
        block_tensor_ctrl<3, T> ctrla(bta), ctrlb(btb);

        index<3> i012, i021, i022, i102, i111, i120, i201, i202, i210, i220;
        i012[0] = 0; i012[1] = 1; i012[2] = 2;
        i021[0] = 0; i021[1] = 2; i021[2] = 1;
        i022[0] = 0; i022[1] = 2; i022[2] = 2;
        i102[0] = 1; i102[1] = 0; i102[2] = 2;
        i111[0] = 1; i111[1] = 1; i111[2] = 1;
        i120[0] = 1; i120[1] = 2; i120[2] = 0;
        i201[0] = 2; i201[1] = 0; i201[2] = 1;
        i202[0] = 2; i202[1] = 0; i202[2] = 2;
        i210[0] = 2; i210[1] = 1; i210[2] = 0;
        i220[0] = 2; i220[1] = 2; i220[2] = 0;

        //  Install symmetry in bta
        //
        scalar_transf<T> tr1(-1.);
        ctrla.req_symmetry().insert(se_perm<3, T>(
                permutation<3>().permute(0, 1), tr1));
        ctrla.req_symmetry().insert(se_perm<3, T>(
                permutation<3>().permute(1, 2), tr1));

        // Prepare symmetrized blocks
        //
        dimensions<3> d012 = bis.get_block_dims(i012),
                d111 = bis.get_block_dims(i111),
                d022 = bis.get_block_dims(i022);
        dense_tensor<3, T, allocator_t> t012(d012), t111(d111), t111a(d111),
                t022(d022), t022a(d022);
        to_random<3, T>().perform(t012);
        to_random<3, T>().perform(t111a);
        to_random<3, T>().perform(t022a);
        to_add<3, T> sym111(t111a);
        sym111.add_op(t111a, permutation<3>().permute(0, 1), -1.0);
        sym111.add_op(t111a, permutation<3>().permute(0, 2), -1.0);
        sym111.perform(true, t111);
        to_add<3, T> sym022(t022a);
        sym022.add_op(t022a, permutation<3>().permute(1, 2), -1.0);
        sym022.perform(true, t022);

        // Copy [0,1,2]
        //
        to_copy<3, T>(t012).perform(true, ctrla.req_block(i012));
        ctrla.ret_block(i012);
        to_copy<3, T>(t012).perform(true, ctrlb.req_block(i012));
        ctrlb.ret_block(i012);
        to_copy<3, T>(t012, permutation<3>().permute(1, 2), -1.0).
                perform(true, ctrlb.req_block(i021));
        ctrlb.ret_block(i021);
        to_copy<3, T>(t012, permutation<3>().permute(0, 1), -1.0).
                perform(true, ctrlb.req_block(i102));
        ctrlb.ret_block(i102);
        to_copy<3, T>(t012, permutation<3>().permute(0, 1).permute(1, 2), 1.0).
                perform(true, ctrlb.req_block(i120));
        ctrlb.ret_block(i120);
        to_copy<3, T>(t012, permutation<3>().permute(0, 2), -1.0).
                perform(true, ctrlb.req_block(i210));
        ctrlb.ret_block(i210);
        to_copy<3, T>(t012, permutation<3>().permute(1, 2).permute(0, 1), 1.0).
                perform(true, ctrlb.req_block(i201));
        ctrlb.ret_block(i201);

        // Copy [0,2,2]
        //
        to_copy<3, T>(t022).perform(true, ctrla.req_block(i022));
        ctrla.ret_block(i022);
        to_copy<3, T>(t022).perform(true, ctrlb.req_block(i022));
        ctrlb.ret_block(i022);
        to_copy<3, T>(t022, permutation<3>().permute(0, 1), -1.0).
                perform(true, ctrlb.req_block(i202));
        ctrlb.ret_block(i202);
        to_copy<3, T>(t022, permutation<3>().permute(0, 1).permute(1, 2), 1.0).
                perform(true, ctrlb.req_block(i220));
        ctrlb.ret_block(i220);

        // Copy [1,1,1]
        //
        to_copy<3, T>(t111).perform(true, ctrla.req_block(i111));
        ctrla.ret_block(i111);
        to_copy<3, T>(t111).perform(true, ctrlb.req_block(i111));
        ctrlb.ret_block(i111);

        bta.set_immutable();
        btb.set_immutable();

        // Convert to simple tensors
        //
        dense_tensor<3, T, allocator_t> ta(dims), ta_ref(dims);
        to_btconv<3, T>(bta).perform(ta);
        to_btconv<3, T>(btb).perform(ta_ref);

        // Compare the result against the reference
        //
        compare_ref_x<3, T>::compare(testname, ta, ta_ref, 0.0);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}

template class to_btconv_test_x<double>;
template class to_btconv_test_x<float>;

} // namespace libtensor
