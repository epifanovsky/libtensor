#include <libtensor/core/allocator.h>
#include <libtensor/core/mask.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/btod_diag.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/se_label.h>
#include <libtensor/symmetry/se_part.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/dense_tensor/tod_btconv.h>
#include <libtensor/dense_tensor/tod_diag.h>
#include "btod_diag_test.h"
#include "../compare_ref.h"

namespace libtensor {


void btod_diag_test::perform() throw(libtest::test_exception) {

    allocator<double>::init();

    try {

    test_zero_1();
    test_zero_2();
    test_zero_3();

    test_nosym_1(false);
    test_nosym_2(false);
    test_nosym_3(false);
    test_nosym_4(false);
    test_nosym_5(false);
    test_nosym_6(false);

    test_nosym_1(true);
    test_nosym_2(true);
    test_nosym_3(true);
    test_nosym_4(true);
    test_nosym_5(true);
    test_nosym_6(true);

    test_sym_1(false);
    test_sym_1(true);

    test_sym_2(false);
    test_sym_2(true);

    test_sym_4(false);
    test_sym_4(true);

    test_sym_5(false);
    test_sym_5(true);

    test_sym_6(false);
    test_sym_6(true);

    test_sym_7(false);
    test_sym_7(true);

    test_sym_8(false);
    test_sym_8(true);

    test_sym_9(false);
    test_sym_9(true);

    test_sym_10(false);
    test_sym_10(true);

    test_sym_11(false);
    test_sym_11(true);

    } catch(...) {
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();
}

/** \test Extract diagonal: \f$ b_i = a_{ii} \f$, zero tensor, one block
 **/
void btod_diag_test::test_zero_1() throw(libtest::test_exception) {

    static const char *testname = "btod_diag_test::test_zero_1()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<1> i1a, i1b;
    i1b[0] = 10;
    libtensor::index<2> i2a, i2b;
    i2b[0] = 10; i2b[1] = 10;
    dimensions<1> dims1(index_range<1>(i1a, i1b));
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    block_index_space<1> bis1(dims1);
    block_index_space<2> bis2(dims2);

    block_tensor<2, double, allocator_t> bta(bis2);
    block_tensor<1, double, allocator_t> btb(bis1);

    sequence<2, size_t> msk;
    msk[0] = 1; msk[1] = 1;

    //  Fill in random data
    btod_random<1>().perform(btb);
    bta.set_immutable();

    //  Invoke the operation
    btod_diag<2, 1>(bta, msk).perform(btb);

    block_tensor_ctrl<1, double> ctrlb(btb);
    dimensions<1> bidims1 = bis1.get_block_index_dims();
    orbit_list<1, double> olb(ctrlb.req_const_symmetry());
    for (orbit_list<1, double>::iterator ib = olb.begin();
            ib != olb.end(); ib++) {
        orbit<1, double> ob(ctrlb.req_const_symmetry(), olb.get_abs_index(ib));
        abs_index<1> bidx(ob.get_acindex(), bidims1);
        if (! ctrlb.req_is_zero_block(bidx.get_index()))
            fail_test(testname, __FILE__, __LINE__, "Unexpected non-zero block.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Extract diagonal: \f$ b_i = a_{ii} \f$, zero tensor, multiple blocks
 **/
void btod_diag_test::test_zero_2() throw(libtest::test_exception) {

    static const char *testname = "btod_diag_test::test_zero_2()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<1> i1a, i1b;
    i1b[0] = 10;
    libtensor::index<2> i2a, i2b;
    i2b[0] = 10; i2b[1] = 10;
    dimensions<1> dims1(index_range<1>(i1a, i1b));
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    block_index_space<1> bis1(dims1);
    block_index_space<2> bis2(dims2);

    mask<1> msk1;
    msk1[0] = true;
    mask<2> msk2;
    msk2[0] = true; msk2[1] = true;
    bis1.split(msk1,3); bis1.split(msk1,6);
    bis2.split(msk2,3); bis2.split(msk2,6);

    block_tensor<2, double, allocator_t> bta(bis2);
    block_tensor<1, double, allocator_t> btb(bis1);

    sequence<2, size_t> msk;
    msk[0] = 1; msk[1] = 1;

    //  Fill in random data
    btod_random<1>().perform(btb);
    bta.set_immutable();

    //  Invoke the operation
    btod_diag<2, 1>(bta, msk).perform(btb);

    block_tensor_ctrl<1, double> ctrlb(btb);
    dimensions<1> bidims1 = bis1.get_block_index_dims();
    orbit_list<1, double> olb(ctrlb.req_const_symmetry());
    for (orbit_list<1, double>::iterator ib = olb.begin();
            ib != olb.end(); ib++) {
        orbit<1, double> ob(ctrlb.req_const_symmetry(), olb.get_abs_index(ib));
        abs_index<1> bidx(ob.get_acindex(), bidims1);
        if (! ctrlb.req_is_zero_block(bidx.get_index()))
            fail_test(testname, __FILE__, __LINE__, "Unexpected non-zero block.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Extract diagonal: \f$ b_ij = a_{ijji} \f$, zero tensor, multiple blocks
 **/
void btod_diag_test::test_zero_3() throw(libtest::test_exception) {

    static const char *testname = "btod_diag_test::test_zero_3()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<2> i2a, i2b;
    i2b[0] = 10; i2b[1] = 10;
    libtensor::index<4> i4a, i4b;
    i4b[0] = 10; i4b[1] = 10; i4b[2] = 10; i4b[3] = 10;
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    dimensions<4> dims4(index_range<4>(i4a, i4b));
    block_index_space<2> bis2(dims2);
    block_index_space<4> bis4(dims4);

    mask<2> msk2;
    msk2[0] = true; msk2[1] = true;
    mask<4> msk4;
    msk4[0] = true; msk4[1] = true; msk4[2] = true; msk4[3] = true;
    bis2.split(msk2,3); bis2.split(msk2,6);
    bis4.split(msk4,3); bis4.split(msk4,6);

    block_tensor<4, double, allocator_t> bta(bis4);
    block_tensor<2, double, allocator_t> btb(bis2);

    sequence<4, size_t> msk;
    msk[0] = 1; msk[1] = 2; msk[2] = 2; msk[3] = 1;

    //  Fill in random data
    btod_random<2>().perform(btb);
    bta.set_immutable();

    //  Invoke the operation
    btod_diag<4, 2>(bta, msk).perform(btb);

    block_tensor_ctrl<2, double> ctrlb(btb);
    dimensions<2> bidims2 = bis2.get_block_index_dims();
    orbit_list<2, double> olb(ctrlb.req_const_symmetry());
    for (orbit_list<2, double>::iterator ib = olb.begin();
            ib != olb.end(); ib++) {
        orbit<2, double> ob(ctrlb.req_const_symmetry(), olb.get_abs_index(ib));
        abs_index<2> bidx(ob.get_acindex(), bidims2);
        if (! ctrlb.req_is_zero_block(bidx.get_index()))
            fail_test(testname, __FILE__, __LINE__, "Unexpected non-zero block.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Extract diagonal: \f$ b_i = a_{ii} \f$, non-zero tensor,
     single block
 **/
void btod_diag_test::test_nosym_1(bool add) throw(libtest::test_exception) {

    static const char *testname = "btod_diag_test::test_nosym_1(bool)";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<1> i1a, i1b;
    i1b[0] = 10;
    libtensor::index<2> i2a, i2b;
    i2b[0] = 10; i2b[1] = 10;
    dimensions<1> dims1(index_range<1>(i1a, i1b));
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    block_index_space<1> bis1(dims1);
    block_index_space<2> bis2(dims2);

    block_tensor<2, double, allocator_t> bta(bis2);
    block_tensor<1, double, allocator_t> btb(bis1);

    dense_tensor<2, double, allocator_t> ta(dims2);
    dense_tensor<1, double, allocator_t> tb(dims1), tb_ref(dims1);

    sequence<2, size_t> msk;
    msk[0] = 1; msk[1] = 1;

    //  Fill in random data
    btod_random<2>().perform(bta);
    bta.set_immutable();
    tod_btconv<2>(bta).perform(ta);

    if (add) {
        //  Fill with random data
        btod_random<1>().perform(btb);

        //  Prepare the reference
        tod_btconv<1>(btb).perform(tb_ref);

        tod_diag<2, 1>(ta, msk).perform(false, tb_ref);

        //  Invoke the operation
        btod_diag<2, 1>(bta, msk).perform(btb, 1.0);
    }
    else {
        //  Prepare the reference
        tod_diag<2, 1>(ta, msk).perform(true, tb_ref);

        //  Invoke the operation
        btod_diag<2, 1>(bta, msk).perform(btb);
    }

    tod_btconv<1>(btb).perform(tb);

    //  Compare against the reference
    compare_ref<1>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Extract a single diagonal: \f$ b_{ija} = a_{iajb} \f$
 **/
void btod_diag_test::test_nosym_2(bool add) throw(libtest::test_exception) {

    static const char *testname = "btod_diag_test::test_nosym_2(bool)";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<3> i3a, i3b;
    i3b[0] = 5; i3b[1] = 5; i3b[2] = 10;
    libtensor::index<4> i4a, i4b;
    i4b[0] = 5; i4b[1] = 10; i4b[2] = 5; i4b[3] = 10;
    dimensions<3> dims3(index_range<3>(i3a, i3b));
    dimensions<4> dims4(index_range<4>(i4a, i4b));
    block_index_space<3> bis3(dims3);
    block_index_space<4> bis4(dims4);

    mask<3> msk3;
    msk3[2] = true;
    bis3.split(msk3, 6);
    mask<4> msk4;
    msk4[1] = true; msk4[3]=true;
    bis4.split(msk4, 6);

    block_tensor<4, double, allocator_t> bta(bis4);
    block_tensor<3, double, allocator_t> btb(bis3);

    dense_tensor<4, double, allocator_t> ta(dims4);
    dense_tensor<3, double, allocator_t> tb(dims3), tb_ref(dims3);

    permutation<3> pb;
    pb.permute(1,2);

    sequence<4, size_t> msk(0);
    msk[1] = 1; msk[3] = 1;

    //  Fill in random data
    btod_random<4>().perform(bta);
    bta.set_immutable();

    //  Prepare the reference
    tod_btconv<4>(bta).perform(ta);

    if (add) {
        //  Fill in random data
        btod_random<3>().perform(btb);

        //  Prepare the reference
        tod_btconv<3>(btb).perform(tb_ref);

        tod_diag<4, 3>(ta, msk,
                tensor_transf<3, double>(pb)).perform(false, tb_ref);

        //  Invoke the operation
        btod_diag<4, 3>(bta, msk, pb).perform(btb, 1.0);

    } else {
        tod_diag<4, 3>(ta, msk,
                tensor_transf<3, double>(pb)).perform(true, tb_ref);

        //  Invoke the operation
        btod_diag<4, 3>(bta, msk, pb).perform(btb);
    }

    tod_btconv<3>(btb).perform(tb);

    //  Compare against the reference
    compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Extract diagonal: \f$ b_i = a_{ii} \f$, non-zero tensor,
     multiple blocks
 **/
void btod_diag_test::test_nosym_3(bool add) throw(libtest::test_exception) {

    static const char *testname = "btod_diag_test::test_nosym_3(bool)";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<1> i1a, i1b;
    i1b[0] = 10;
    libtensor::index<2> i2a, i2b;
    i2b[0] = 10; i2b[1] = 10;
    dimensions<1> dims1(index_range<1>(i1a, i1b));
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    block_index_space<1> bis1(dims1);
    block_index_space<2> bis2(dims2);

    mask<1> msk1;
    msk1[0] = true;
    mask<2> msk2;
    msk2[0] = true; msk2[1] = true;
    bis1.split(msk1,3); bis1.split(msk1,6);
    bis2.split(msk2,3); bis2.split(msk2,6);

    block_tensor<2, double, allocator_t> bta(bis2);
    block_tensor<1, double, allocator_t> btb(bis1);

    dense_tensor<2, double, allocator_t> ta(dims2);
    dense_tensor<1, double, allocator_t> tb(dims1), tb_ref(dims1);

    sequence<2, size_t> msk;
    msk[0] = 1; msk[1] = 1;

    //  Fill in random data
    btod_random<2>().perform(bta);
    bta.set_immutable();

    //  Prepare the reference
    tod_btconv<2>(bta).perform(ta);

    if (add) {
        //  Fill in random data
        btod_random<1>().perform(btb);

        //  Prepare the reference
        tod_btconv<1>(btb).perform(tb_ref);

        tod_diag<2, 1>(ta, msk).perform(false, tb_ref);

        //  Invoke the operation
        btod_diag<2, 1>(bta, msk).perform(btb, 1.0);
    }
    else {
        tod_diag<2, 1>(ta, msk).perform(true, tb_ref);

        //  Invoke the operation
        btod_diag<2, 1>(bta, msk).perform(btb);
    }

    tod_btconv<1>(btb).perform(tb);

    //  Compare against the reference
    compare_ref<1>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Extract diagonal: \f$ b_{ija} = a_{iaja} \f$, non-zero tensor,
     multiple blocks with permutation
 **/
void btod_diag_test::test_nosym_4(bool add) throw(libtest::test_exception) {

    static const char *testname = "btod_diag_test::test_nosym_4(bool)";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<3> i3a, i3b;
    i3b[0] = 5; i3b[1] = 5; i3b[2] = 10;
    libtensor::index<4> i4a, i4b;
    i4b[0] = 5; i4b[1] = 10; i4b[2] = 5; i4b[3] = 10;
    dimensions<3> dims3(index_range<3>(i3a, i3b));
    dimensions<4> dims4(index_range<4>(i4a, i4b));
    block_index_space<3> bis3(dims3);
    block_index_space<4> bis4(dims4);

    mask<3> msk3;
    msk3[0] = true; msk3[1] = true;
    mask<4> msk4;
    msk4[0] = true; msk4[2] = true;
    bis3.split(msk3,2);
    bis4.split(msk4,2);
    msk3[0] = false; msk3[1] = false; msk3[2] = true;
    msk4[0] = false; msk4[1] = true; msk4[2] = false; msk4[3] = true;
    bis3.split(msk3,3);
    bis4.split(msk4,3);

    block_tensor<4, double, allocator_t> bta(bis4);
    block_tensor<3, double, allocator_t> btb(bis3);

    dense_tensor<4, double, allocator_t> ta(dims4);
    dense_tensor<3, double, allocator_t> tb(dims3), tb_ref(dims3);

    sequence<4, size_t> msk(0);
    msk[1] = 1; msk[3] = 1;

    permutation<3> pb;
    pb.permute(1,2);

    //  Fill in random data
    btod_random<4>().perform(bta);
    bta.set_immutable();

    //  Prepare the reference
    tod_btconv<4>(bta).perform(ta);

    if (add) {
        btod_random<3>().perform(btb);

        //  Prepare the reference
        tod_btconv<3>(btb).perform(tb_ref);

        tod_diag<4, 3>(ta, msk,
                tensor_transf<3, double>(pb)).perform(false, tb_ref);

        //  Invoke the operation
        btod_diag<4, 3>(bta, msk, pb).perform(btb, 1.0);
    }
    else {
        tod_diag<4, 3>(ta, msk,
                tensor_transf<3, double>(pb)).perform(true, tb_ref);

        //  Invoke the operation
        btod_diag<4, 3>(bta, msk, pb).perform(btb);
    }
    tod_btconv<3>(btb).perform(tb);

    //  Compare against the reference
    compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Extract diagonal: \f$ b_{ia} = a_{iiaa} \f$, non-zero tensor,
     multiple blocks
 **/
void btod_diag_test::test_nosym_5(bool add) throw(libtest::test_exception) {

    static const char *testname = "btod_diag_test::test_nosym_5(bool)";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<2> i2a, i2b;
    i2b[0] = 8; i2b[1] = 10;
    libtensor::index<4> i4a, i4b;
    i4b[0] = 8; i4a[1] = 8; i4a[2] = 10; i4a[3] = 10;
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    dimensions<4> dims4(index_range<4>(i4a, i4b));
    block_index_space<2> bis2(dims2);
    block_index_space<4> bis4(dims4);

    mask<2> msk2a, msk2b;
    msk2a[0] = true; msk2b[1] = true;
    mask<4> msk4a, msk4b;
    msk4a[0] = true; msk4a[1] = true;
    msk4b[2] = true; msk4b[3] = true;
    bis2.split(msk2a,2); bis2.split(msk2a,5);
    bis2.split(msk2b,3); bis2.split(msk2b,6);
    bis4.split(msk4a,2); bis4.split(msk4a,5);
    bis4.split(msk4b,3); bis4.split(msk4b,6);

    block_tensor<4, double, allocator_t> bta(bis4);
    block_tensor<2, double, allocator_t> btb(bis2);

    dense_tensor<4, double, allocator_t> ta(dims4);
    dense_tensor<2, double, allocator_t> tb(dims2), tb_ref(dims2);

    sequence<4, size_t> msk;
    msk[0] = 1; msk[1] = 1; msk[2] = 2; msk[3] = 2;

    //  Fill in random data
    btod_random<4>().perform(bta);
    bta.set_immutable();

    //  Prepare the reference
    tod_btconv<4>(bta).perform(ta);

    if (add) {
        //  Fill in random data
        btod_random<2>().perform(btb);

        //  Prepare the reference
        tod_btconv<2>(btb).perform(tb_ref);

        tod_diag<4, 2>(ta, msk).perform(false, tb_ref);

        //  Invoke the operation
        btod_diag<4, 2>(bta, msk).perform(btb, 1.0);
    }
    else {
        tod_diag<4, 2>(ta, msk).perform(true, tb_ref);

        //  Invoke the operation
        btod_diag<4, 2>(bta, msk).perform(btb);
    }

    tod_btconv<2>(btb).perform(tb);

    //  Compare against the reference
    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Extract diagonal: \f$ b_{ija} = a_{iaija} \f$, non-zero tensor,
     multiple blocks with permutation
 **/
void btod_diag_test::test_nosym_6(bool add) throw(libtest::test_exception) {

    static const char *testname = "btod_diag_test::test_nosym_6(bool)";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<3> i3a, i3b;
    i3b[0] = 5; i3b[1] = 5; i3b[2] = 10;
    libtensor::index<5> i5a, i5b;
    i5b[0] = 5; i5b[1] = 10; i5b[2] = 5; i5b[3] = 5; i5b[4] = 10;
    dimensions<3> dims3(index_range<3>(i3a, i3b));
    dimensions<5> dims5(index_range<5>(i5a, i5b));
    block_index_space<3> bis3(dims3);
    block_index_space<5> bis5(dims5);

    mask<3> msk3a, msk3b;
    msk3a[0] = true; msk3a[1] = true; msk3b[2] = true;
    bis3.split(msk3a, 2);
    bis3.split(msk3b, 6);
    mask<5> msk5a, msk5b;
    msk5a[0] = true; msk5a[2] = true; msk5a[3] = true;
    msk5b[1] = true; msk5b[4] = true;
    bis5.split(msk5a, 2);
    bis5.split(msk5b, 6);

    block_tensor<5, double, allocator_t> bta(bis5);
    block_tensor<3, double, allocator_t> btb(bis3);

    dense_tensor<5, double, allocator_t> ta(dims5);
    dense_tensor<3, double, allocator_t> tb(dims3), tb_ref(dims3);

    permutation<3> pb;
    pb.permute(1,2);

    sequence<5, size_t> msk(0);
    msk[0] = 1; msk[1] = 2; msk[2] = 1; msk[4] = 2;

    //  Fill in random data
    btod_random<5>().perform(bta);
    bta.set_immutable();

    //  Prepare the reference
    tod_btconv<5>(bta).perform(ta);

    if (add) {
        btod_random<3>().perform(btb);

        //  Prepare the reference
        tod_btconv<3>(btb).perform(tb_ref);

        tod_diag<5, 3>(ta, msk,
                tensor_transf<3, double>(pb)).perform(false, tb_ref);

        //  Invoke the operation
        btod_diag<5, 3>(bta, msk, pb).perform(btb, 1.0);
    }
    else {
        tod_diag<5, 3>(ta, msk,
                tensor_transf<3, double>(pb)).perform(true, tb_ref);

        //  Invoke the operation
        btod_diag<5, 3>(bta, msk, pb).perform(btb);
    }
    tod_btconv<3>(btb).perform(tb);

    //  Compare against the reference
    compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Extract diagonal: \f$ b_i = a_{ii} \f$, permutational symmetry,
     multiple blocks
 **/
void btod_diag_test::test_sym_1(bool add) throw(libtest::test_exception) {

    static const char *testname = "btod_diag_test::test_sym_1(bool)";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<1> i1a, i1b;
    i1b[0] = 10;
    libtensor::index<2> i2a, i2b;
    i2b[0] = 10; i2b[1] = 10;
    dimensions<1> dims1(index_range<1>(i1a, i1b));
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    block_index_space<1> bis1(dims1);
    block_index_space<2> bis2(dims2);

    mask<1> msk1;
    msk1[0] = true;
    mask<2> msk2;
    msk2[0] = true; msk2[1] = true;
    bis1.split(msk1,3); bis1.split(msk1,6);
    bis2.split(msk2,3); bis2.split(msk2,6);

    block_tensor<2, double, allocator_t> bta(bis2);
    block_tensor<1, double, allocator_t> btb(bis1);

    dense_tensor<2, double, allocator_t> ta(dims2);
    dense_tensor<1, double, allocator_t> tb(dims1), tb_ref(dims1);

    permutation<2> perm10;
    perm10.permute(0, 1);
    scalar_transf<double> tr0, tr1(-1.);
    se_perm<2, double> cycle1(perm10, tr0);
    block_tensor_ctrl<2, double> ctrla(bta);
    ctrla.req_symmetry().insert(cycle1);

    sequence<2, size_t> msk;
    msk[0] = 1; msk[1] = 1;

    //  Fill in random data
    btod_random<2>().perform(bta);
    bta.set_immutable();

    //  Prepare the reference
    tod_btconv<2>(bta).perform(ta);

    if (add) {
        //  Fill in random data
        btod_random<1>().perform(btb);

        //  Prepare the reference
        tod_btconv<1>(btb).perform(tb_ref);

        tod_diag<2, 1>(ta, msk).perform(false, tb_ref);

        //  Invoke the operation
        btod_diag<2, 1>(bta, msk).perform(btb, 1.0);
    }
    else {
        tod_diag<2, 1>(ta, msk).perform(true, tb_ref);

        //  Invoke the operation
        btod_diag<2, 1>(bta, msk).perform(btb);
    }
    tod_btconv<1>(btb).perform(tb);

    //  Compare against the reference
    compare_ref<1>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Extract diagonal: \f$ b_{ia} = a_{iia} \f$, permutational symmetry,
     multiple blocks
 **/
void btod_diag_test::test_sym_2(bool add) throw(libtest::test_exception) {

    static const char *testname = "btod_diag_test::test_sym_2(bool)";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<2> i2a, i2b;
    i2b[0] = 10; i2b[1] = 5;
    libtensor::index<3> i3a, i3b;
    i3b[0] = 10; i3b[1] = 10; i3b[2] = 5;
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    dimensions<3> dims3(index_range<3>(i3a, i3b));
    block_index_space<2> bis2(dims2);
    block_index_space<3> bis3(dims3);

    mask<2> msk2;
    msk2[0] = true; msk2[1] = false;
    mask<3> msk3;
    msk3[0] = true; msk3[1] = true; msk3[2] = false;
    bis2.split(msk2,3); bis2.split(msk2,6);
    bis3.split(msk3,3); bis3.split(msk3,6);
    msk2[0] = false; msk2[1] = true;
    msk3[0] = false; msk3[1] = false; msk3[2] = true;
    bis2.split(msk2,5);
    bis3.split(msk3,5);

    block_tensor<3, double, allocator_t> bta(bis3);
    block_tensor<2, double, allocator_t> btb(bis2);

    dense_tensor<3, double, allocator_t> ta(dims3);
    dense_tensor<2, double, allocator_t> tb(dims2), tb_ref(dims2);

    permutation<3> perm10;
    perm10.permute(0, 1);
    scalar_transf<double> tr0, tr1(-1.);
    se_perm<3, double> cycle1(perm10, tr0);
    block_tensor_ctrl<3, double> ctrla(bta);
    ctrla.req_symmetry().insert(cycle1);

    sequence<3, size_t> msk(0);
    msk[0] = 1; msk[1] = 1;

    //  Fill in random data
    btod_random<3>().perform(bta);
    bta.set_immutable();

    //  Prepare the reference
    tod_btconv<3>(bta).perform(ta);

    if (add) {
        //  Fill in random data
        btod_random<2>().perform(btb);

        //  Prepare the reference
        tod_btconv<2>(btb).perform(tb_ref);

        tod_diag<3, 2>(ta, msk).perform(false, tb_ref);

        //  Invoke the operation
        btod_diag<3, 2>(bta, msk).perform(btb, 1.0);
    }
    else {
        tod_diag<3, 2>(ta, msk).perform(true, tb_ref);

        //  Invoke the operation
        btod_diag<3, 2>(bta, msk).perform(btb);
    }

    tod_btconv<2>(btb).perform(tb);

    //  Compare against the reference
    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Extract diagonal: \f$ b_i = a_{ii} \f$, permutational anti-symmetry,
     multiple blocks

    This test will always fail, since the formation of the respective symmetry
    element fails by definition: (p0,-1.) is not valid.

    TODO: Remove this test!!!
 **/
void btod_diag_test::test_sym_3(bool add) throw(libtest::test_exception) {

    static const char *testname = "btod_diag_test::test_sym_3(bool)";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<1> i1a, i1b;
    i1b[0] = 10;
    libtensor::index<2> i2a, i2b;
    i2b[0] = 10; i2b[1] = 10;
    dimensions<1> dims1(index_range<1>(i1a, i1b));
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    block_index_space<1> bis1(dims1);
    block_index_space<2> bis2(dims2);

    mask<1> msk1;
    msk1[0] = true;
    mask<2> msk2;
    msk2[0] = true; msk2[1] = true;
    bis1.split(msk1,3); bis1.split(msk1,6);
    bis2.split(msk2,3); bis2.split(msk2,6);

    block_tensor<2, double, allocator_t> bta(bis2);
    block_tensor<1, double, allocator_t> btb(bis1);

    dense_tensor<2, double, allocator_t> ta(dims2);
    dense_tensor<1, double, allocator_t> tb(dims1), tb_ref(dims1);

    permutation<2> perm10;
    perm10.permute(0, 1);
    scalar_transf<double> tr0, tr1(-1.);
    se_perm<2, double> cycle1(perm10, tr1);
    block_tensor_ctrl<2, double> ctrla(bta);
    ctrla.req_symmetry().insert(cycle1);

    sequence<2, size_t> msk;
    msk[0] = 1; msk[1] = 1;

    //  Fill in random data
    btod_random<2>().perform(bta);
    bta.set_immutable();

    //  Prepare the reference
    tod_btconv<2>(bta).perform(ta);

    if (add) {
        //  Fill in random data
        btod_random<1>().perform(btb);

        //  Prepare the reference
        tod_btconv<1>(btb).perform(tb_ref);

        tod_diag<2, 1>(ta, msk).perform(false, tb_ref);

        //  Invoke the operation
        btod_diag<2, 1>(bta, msk).perform(btb, 1.0);
    }
    else {
        tod_diag<2, 1>(ta, msk).perform(true, tb_ref);

        //  Invoke the operation
        btod_diag<2, 1>(bta, msk).perform(btb);
    }

    tod_btconv<1>(btb).perform(tb);

    //  Compare against the reference
    compare_ref<1>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Extract diagonal: \f$ b_{ija} = a_{iaja} \f$, permutational anti-symmetry,
     multiple blocks
 **/
void btod_diag_test::test_sym_4(bool add) throw(libtest::test_exception) {

    static const char *testname = "btod_diag_test::test_sym_4(bool)";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<3> i3a, i3b;
    i3b[0] = 5; i3b[1] = 5; i3b[2] = 10;
    libtensor::index<4> i4a, i4b;
    i4b[0] = 5; i4b[1] = 10; i4b[2] = 5; i4b[3] = 10;
    dimensions<3> dims3(index_range<3>(i3a, i3b));
    dimensions<4> dims4(index_range<4>(i4a, i4b));
    block_index_space<3> bis3(dims3);
    block_index_space<4> bis4(dims4);

    mask<3> msk3;
    msk3[0] = true; msk3[1] = true;
    mask<4> msk4;
    msk4[0] = true; msk4[2] = true;
    bis3.split(msk3, 2);
    bis4.split(msk4, 2);
    msk3[0] = false; msk3[1] = false; msk3[2] = true;
    msk4[0] = false; msk4[1] = true; msk4[2] = false; msk4[3] = true;
    bis3.split(msk3,3);
    bis4.split(msk4,3);

    block_tensor<4, double, allocator_t> bta(bis4);
    block_tensor<3, double, allocator_t> btb(bis3);

    dense_tensor<4, double, allocator_t> ta(dims4);
    dense_tensor<3, double, allocator_t> tb(dims3), tb_ref(dims3);

    permutation<4> perm20;
    perm20.permute(0, 2);
    scalar_transf<double> tr0, tr1(-1.);
    se_perm<4, double> cycle1(perm20, tr1);
    block_tensor_ctrl<4, double> ctrla(bta);
    ctrla.req_symmetry().insert(cycle1);

    sequence<4, size_t> msk(0);
    msk[1] = 1; msk[3] = 1;

    permutation<3> pb;
    pb.permute(1,2);

    //  Fill in random data
    btod_random<4>().perform(bta);
    bta.set_immutable();

    //  Prepare the reference
    tod_btconv<4>(bta).perform(ta);

    if (add) {
        //  Fill in random data
        btod_random<3>().perform(btb);

        //  Prepare the reference
        tod_btconv<3>(btb).perform(tb_ref);

        tod_diag<4, 3>(ta, msk,
                tensor_transf<3, double>(pb)).perform(false, tb_ref);

        //  Invoke the operation
        btod_diag<4, 3>(bta, msk, pb).perform(btb, 1.0);
    }
    else {
        tod_diag<4, 3>(ta, msk,
                tensor_transf<3, double>(pb)).perform(true, tb_ref);

        //  Invoke the operation
        btod_diag<4, 3>(bta, msk, pb).perform(btb);
    }
    tod_btconv<3>(btb).perform(tb);

    //  Compare against the reference
    compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Extract diagonal: \f$ b_{iaj} = a_{iaja} \f$,
        permutational symmetry, multiple blocks
 **/
void btod_diag_test::test_sym_5(bool add) throw(libtest::test_exception) {

    static const char *testname = "btod_diag_test::test_sym_5(bool)";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<3> i3a, i3b;
    i3b[0] = 7; i3b[1] = 7; i3b[2] = 10;
    libtensor::index<4> i4a, i4b;
    i4b[0] = 7; i4b[1] = 10; i4b[2] = 7; i4b[3] = 10;
    dimensions<3> dims3(index_range<3>(i3a, i3b));
    dimensions<4> dims4(index_range<4>(i4a, i4b));
    block_index_space<3> bis3(dims3);
    block_index_space<4> bis4(dims4);

    mask<3> msk3;
    msk3[0] = true; msk3[1] = true;
    mask<4> msk4;
    msk4[0] = true; msk4[2] = true;
    bis3.split(msk3,2); bis3.split(msk3, 5);
    bis4.split(msk4,2); bis4.split(msk4, 5);
    msk3[0] = false; msk3[1] = false; msk3[2] = true;
    msk4[0] = false; msk4[1] = true; msk4[2] = false; msk4[3] = true;
    bis3.split(msk3, 5);
    bis4.split(msk4, 5);

    block_tensor<4, double, allocator_t> bta(bis4);
    block_tensor<3, double, allocator_t> btb(bis3);

    dense_tensor<4, double, allocator_t> ta(dims4);
    dense_tensor<3, double, allocator_t> tb(dims3), tb_ref(dims3);

    {
        scalar_transf<double> tr0, tr1(-1.);
        se_perm<4, double> cycle1(
                permutation<4>().permute(0, 2).permute(1, 3), tr0);
        block_tensor_ctrl<4, double> ctrla(bta);
        ctrla.req_symmetry().insert(cycle1);
    }

    sequence<4, size_t> msk(0);
    msk[1] = 1; msk[3] = 1;

    permutation<3> perm;
    perm.permute(1, 2);

    //  Fill in random data
    btod_random<4>().perform(bta);
    bta.set_immutable();

    //  Prepare the reference
    tod_btconv<4>(bta).perform(ta);

    if (add) {
        //  Fill in random data
        btod_random<3>().perform(btb);

        //  Prepare the reference
        tod_btconv<3>(btb).perform(tb_ref);

        tod_diag<4, 3>(ta, msk,
                tensor_transf<3, double>(perm)).perform(false, tb_ref);

        //  Invoke the operation
        btod_diag<4, 3>(bta, msk, perm).perform(btb, 1.0);
    }
    else {
        tod_diag<4, 3>(ta, msk,
                tensor_transf<3, double>(perm)).perform(true, tb_ref);

        //  Invoke the operation
        btod_diag<4, 3>(bta, msk, perm).perform(btb);
    }
    tod_btconv<3>(btb).perform(tb);

    //  Compare against the reference
    compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Extract diagonal: \f$ b_{ijk} = a_{ikjk} \f$,
        permutational anti-symmetry, multiple blocks
 **/
void btod_diag_test::test_sym_6(bool add) throw(libtest::test_exception) {

    static const char *testname = "btod_diag_test::test_sym_6(bool)";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<3> i3a, i3b;
    i3b[0] = 10; i3b[1] = 10; i3b[2] = 10;
    libtensor::index<4> i4a, i4b;
    i4b[0] = 10; i4b[1] = 10; i4b[2] = 10; i4b[3] = 10;
    dimensions<3> dims3(index_range<3>(i3a, i3b));
    dimensions<4> dims4(index_range<4>(i4a, i4b));
    block_index_space<3> bis3(dims3);
    block_index_space<4> bis4(dims4);

    mask<3> msk3;
    msk3[0] = true; msk3[1] = true; msk3[2] = true;
    mask<4> msk4;
    msk4[0] = true; msk4[1] = true; msk4[2] = true; msk4[3] = true;
    bis3.split(msk3, 2);
    bis3.split(msk3, 4);
    bis3.split(msk3, 8);
    bis4.split(msk4, 2);
    bis4.split(msk4, 4);
    bis4.split(msk4, 8);

    block_tensor<4, double, allocator_t> bta(bis4);
    block_tensor<3, double, allocator_t> btb(bis3);

    dense_tensor<4, double, allocator_t> ta(dims4);
    dense_tensor<3, double, allocator_t> tb(dims3), tb_ref(dims3);

    block_tensor_ctrl<4, double> ctrla(bta);

    scalar_transf<double> tr0, tr1(-1.);
    ctrla.req_symmetry().insert(se_perm<4, double>(permutation<4>().
        permute(0, 1), tr1));
    ctrla.req_symmetry().insert(se_perm<4, double>(permutation<4>().
        permute(2, 3), tr1));

    sequence<4, size_t> msk(0);
    msk[1] = 1; msk[3] = 1;

    //  Fill in random data
    btod_random<4>().perform(bta);
    btod_random<3>().perform(btb);
    bta.set_immutable();

    //  Prepare the reference
    tod_btconv<4>(bta).perform(ta);
    tod_btconv<3>(btb).perform(tb_ref);

    //  Invoke the operation
    if(add) {
        btod_diag<4, 3>(bta, msk).perform(btb, 1.0);
        tod_diag<4, 3>(ta, msk).perform(false, tb_ref);
    } else {
        btod_diag<4, 3>(bta, msk).perform(btb);
        tod_diag<4, 3>(ta, msk).perform(true, tb_ref);
    }
    tod_btconv<3>(btb).perform(tb);

    //  Compare against the reference
    compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Extract diagonal: \f$ b_i = a_{ii} \f$, non-zero tensor,
        multiple blocks, label symmetry
 **/
void btod_diag_test::test_sym_7(bool add) throw(libtest::test_exception) {

    static const char *testname = "btod_diag_test::test_sym_7(bool)";

    typedef allocator<double> allocator_t;

    const char *pgtid = "point_group_cs";

    try {

    point_group_table::label_t ap = 0, app = 1;
    std::vector<std::string> irnames(2);
    irnames[0] = "Ap"; irnames[1] = "App";
    point_group_table cs(pgtid, irnames, irnames[0]);
    cs.add_product(ap, ap, ap);
    cs.add_product(ap, app, app);
    cs.add_product(app, ap, app);
    cs.add_product(app, app, ap);
    cs.check();
    product_table_container::get_instance().add(cs);

    {

    libtensor::index<1> i1a, i1b;
    i1b[0] = 10;
    libtensor::index<2> i2a, i2b;
    i2b[0] = 10; i2b[1] = 10;
    dimensions<1> dims1(index_range<1>(i1a, i1b));
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    block_index_space<1> bis1(dims1);
    block_index_space<2> bis2(dims2);

    mask<1> msk1;
    msk1[0] = true;
    mask<2> msk2;
    msk2[0] = true; msk2[1] = true;
    bis1.split(msk1, 3); bis1.split(msk1, 6);
    bis2.split(msk2, 3); bis2.split(msk2, 6);

    se_label<2, double> elem1(bis2.get_block_index_dims(), pgtid);
    block_labeling<2> &bl1 = elem1.get_labeling();
    bl1.assign(msk2, 0, ap);
    bl1.assign(msk2, 1, ap);
    bl1.assign(msk2, 2, app);
    elem1.set_rule(ap);

    block_tensor<2, double, allocator_t> bta(bis2);
    block_tensor<1, double, allocator_t> btb(bis1);

    {
        block_tensor_ctrl<2, double> ca(bta);
        ca.req_symmetry().insert(elem1);
    }

    dense_tensor<2, double, allocator_t> ta(dims2);
    dense_tensor<1, double, allocator_t> tb(dims1), tb_ref(dims1);

    sequence<2, size_t> msk(1);
    msk[0] = 1; msk[1] = 1;

    //  Fill in random data
    btod_random<2>().perform(bta);
    bta.set_immutable();

    //  Prepare the reference
    tod_btconv<2>(bta).perform(ta);

    if (add) {
        //  Fill in random data
        btod_random<1>().perform(btb);

        //  Prepare the reference
        tod_btconv<1>(btb).perform(tb_ref);

        tod_diag<2, 1>(ta, msk).perform(false, tb_ref);

        //  Invoke the operation
        btod_diag<2, 1>(bta, msk).perform(btb, 1.0);
    }
    else {
        tod_diag<2, 1>(ta, msk).perform(true, tb_ref);

        //  Invoke the operation
        btod_diag<2, 1>(bta, msk).perform(btb);
    }

    tod_btconv<1>(btb).perform(tb);

    //  Compare against the reference
    compare_ref<1>::compare(testname, tb, tb_ref, 1e-15);

    }

    } catch(exception &e) {
        product_table_container::get_instance().erase(pgtid);
        fail_test(testname, __FILE__, __LINE__, e.what());
    } catch(...) {
        product_table_container::get_instance().erase(pgtid);
        throw;
    }

    product_table_container::get_instance().erase(pgtid);
}


/** \test Extract diagonal: \f$ b_ijk = a_{ijkj} \f$, non-zero tensor,
        multiple blocks, perm and partition symmetry
 **/
void btod_diag_test::test_sym_8(bool add) throw(libtest::test_exception) {

    static const char *testname = "btod_diag_test::test_sym_8(bool)";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<3> i1a, i1b;
    i1b[0] = i1b[1] = i1b[2] = 9;
    libtensor::index<4> i2a, i2b;
    i2b[0] = i2b[1] = i2b[2] = i2b[3] = 9;
    dimensions<3> dims1(index_range<3>(i1a, i1b));
    dimensions<4> dims2(index_range<4>(i2a, i2b));
    block_index_space<3> bis1(dims1);
    block_index_space<4> bis2(dims2);

    mask<3> msk1;
    msk1[0] = msk1[1] = msk1[2] = true;
    mask<4> msk2;
    msk2[0] = msk2[1] = msk2[2] = msk2[3] = true;
    bis1.split(msk1, 3);
    bis1.split(msk1, 5);
    bis1.split(msk1, 8);
    bis2.split(msk2, 3);
    bis2.split(msk2, 5);
    bis2.split(msk2, 8);

    scalar_transf<double> tr;
    permutation<4> p1, p2, p3;
    p1.permute(0, 1);
    p2.permute(0, 2).permute(1, 3);
    p3.permute(2, 3);
    se_perm<4, double> se1a(p1, tr), se1b(p2, tr), se1c(p3, tr);
    se_part<4, double> se2(bis2, msk2, 2);
    libtensor::index<4> i0, i1;
    i1[2] = i1[3] = 1;
    se2.add_map(i0, i1, tr);
    i0[0] = i0[1] = 1;
    se2.add_map(i1, i0, tr);
    i1[0] = i1[1] = 1;
    se2.add_map(i0, i1, tr);
    i0[2] = 1;
    se2.mark_forbidden(i0);
    i0[2] = 0; i0[3] = 1;
    se2.mark_forbidden(i0);
    i0[1] = 0; i0[2] = 1;
    se2.mark_forbidden(i0);
    i0[0] = 0; i0[1] = 1;
    se2.mark_forbidden(i0);
    i0[2] = 0;
    se2.mark_forbidden(i0);
    i0[2] = 1; i0[3] = 0;
    se2.mark_forbidden(i0);
    i0[0] = 1; i0[1] = 0;
    se2.mark_forbidden(i0);
    i0[2] = 0; i0[3] = 1;
    se2.mark_forbidden(i0);
    i0[0] = 0;
    se2.mark_forbidden(i0);
    i0[2] = 1; i0[3] = 0;
    se2.mark_forbidden(i0);
    i0[1] = 1; i0[2] = 0;
    se2.mark_forbidden(i0);
    i0[0] = 1; i0[1] = 0;
    se2.mark_forbidden(i0);

    block_tensor<4, double, allocator_t> bta(bis2);
    block_tensor<3, double, allocator_t> btb(bis1);

    {
        block_tensor_ctrl<4, double> ca(bta);
        symmetry<4, double> &sym = ca.req_symmetry();
        sym.insert(se1a);
        sym.insert(se1b);
        sym.insert(se1c);
        sym.insert(se2);
    }

    dense_tensor<4, double, allocator_t> ta(dims2);
    dense_tensor<3, double, allocator_t> tb(dims1), tb_ref(dims1);

    sequence<4, size_t> msk(0);
    msk[1] = 1; msk[3] = 1;

    //  Fill in random data
    btod_random<4>().perform(bta);
    bta.set_immutable();

    //  Prepare the reference
    tod_btconv<4>(bta).perform(ta);

    if (add) {
        //  Fill in random data
        btod_random<3>().perform(btb);

        //  Prepare the reference
        tod_btconv<3>(btb).perform(tb_ref);

        tod_diag<4, 3>(ta, msk).perform(false, tb_ref);

        //  Invoke the operation
        btod_diag<4, 3>(bta, msk).perform(btb, 1.0);
    }
    else {
        tod_diag<4, 3>(ta, msk).perform(true, tb_ref);

        //  Invoke the operation
        btod_diag<4, 3> diag(bta, msk);
        diag.perform(btb);
    }

    tod_btconv<3>(btb).perform(tb);

    //  Compare against the reference
    compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Extract diagonal: \f$ b_{ij} = a_{ijij} \f$,
        permutational anti-symmetry, multiple blocks
 **/
void btod_diag_test::test_sym_9(bool add) throw(libtest::test_exception) {

    static const char *testname = "btod_diag_test::test_sym_9(bool)";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<2> i2a, i2b;
    i2b[0] = 10; i2b[1] = 10;
    libtensor::index<4> i4a, i4b;
    i4b[0] = 10; i4b[1] = 10; i4b[2] = 10; i4b[3] = 10;
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    dimensions<4> dims4(index_range<4>(i4a, i4b));
    block_index_space<2> bis2(dims2);
    block_index_space<4> bis4(dims4);

    mask<2> msk2;
    msk2[0] = true; msk2[1] = true;
    mask<4> msk4;
    msk4[0] = true; msk4[1] = true; msk4[2] = true; msk4[3] = true;
    bis2.split(msk2, 2);
    bis2.split(msk2, 4);
    bis2.split(msk2, 8);
    bis4.split(msk4, 2);
    bis4.split(msk4, 4);
    bis4.split(msk4, 8);

    block_tensor<4, double, allocator_t> bta(bis4);
    block_tensor<2, double, allocator_t> btb(bis2);

    symmetry<2, double> sym_ref(bis2);

    dense_tensor<4, double, allocator_t> ta(dims4);
    dense_tensor<2, double, allocator_t> tb(dims2), tb_ref(dims2);

    block_tensor_ctrl<4, double> ctrla(bta);
    block_tensor_ctrl<2, double> ctrlb(btb);

    scalar_transf<double> tr0, tr1(-1.);
    ctrla.req_symmetry().insert(se_perm<4, double>(permutation<4>().
        permute(0, 1), tr1));
    ctrla.req_symmetry().insert(se_perm<4, double>(permutation<4>().
        permute(2, 3), tr1));
    ctrla.req_symmetry().insert(se_perm<4, double>(permutation<4>().
        permute(0, 2).permute(1, 3), tr0));
    ctrlb.req_symmetry().insert(se_perm<2, double>(permutation<2>().
        permute(0, 1), tr0));
    sym_ref.insert(se_perm<2, double>(permutation<2>().permute(0, 1), tr0));

    sequence<4, size_t> msk(0);
    msk[0] = 1; msk[1] = 2; msk[2] = 1; msk[3] = 2;

    //  Fill in random data
    btod_random<4>().perform(bta);
    btod_random<2>().perform(btb);
    bta.set_immutable();

    //  Prepare the reference
    tod_btconv<4>(bta).perform(ta);
    tod_btconv<2>(btb).perform(tb_ref);

    //  Invoke the operation
    if(add) {
        btod_diag<4, 2>(bta, msk).perform(btb, 1.0);
        tod_diag<4, 2>(ta, msk).perform(false, tb_ref);
    } else {
        btod_diag<4, 2>(bta, msk).perform(btb);
        tod_diag<4, 2>(ta, msk).perform(true, tb_ref);
    }
    tod_btconv<2>(btb).perform(tb);

    //  Compare against the reference

    compare_ref<2>::compare(testname, ctrlb.req_const_symmetry(), sym_ref);
    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Extract diagonal: \f$ b_{ia} = a_{aaii} \f$, non-zero tensor,
        multiple blocks, label symmetry
 **/
void btod_diag_test::test_sym_10(bool add) throw(libtest::test_exception) {

    static const char *testname = "btod_diag_test::test_sym_10(bool)";

    typedef allocator<double> allocator_t;

    const char *pgtid = "point_group_cs";

    try {

    point_group_table::label_t ap = 0, app = 1;
    std::vector<std::string> irnames(2);
    irnames[0] = "Ap"; irnames[1] = "App";
    point_group_table cs(pgtid, irnames, irnames[0]);
    cs.add_product(ap, ap, ap);
    cs.add_product(ap, app, app);
    cs.add_product(app, ap, app);
    cs.add_product(app, app, ap);
    cs.check();
    product_table_container::get_instance().add(cs);

    {

    libtensor::index<2> i2a, i2b;
    i2b[0] = 8; i2b[1] = 10;
    libtensor::index<4> i4a, i4b;
    i4b[0] = 10; i4b[1] = 10; i4b[2] = 8; i4b[3] = 8;
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    dimensions<4> dims4(index_range<4>(i4a, i4b));
    block_index_space<2> bis2(dims2);
    block_index_space<4> bis4(dims4);

    mask<4> msk4;
    msk4[0] = true; msk4[1] = true;
    mask<2> msk2;
    msk2[1] = true;
    bis4.split(msk4, 3); bis4.split(msk4, 6);
    bis2.split(msk2, 3); bis2.split(msk2, 6);
    msk4[0] = false; msk4[1] = false; msk4[2] = true; msk4[3] = true;
    msk2[1] = false; msk2[0] = true;
    bis4.split(msk4, 2); bis4.split(msk4, 5);
    bis2.split(msk2, 2); bis2.split(msk2, 5);

    se_label<4, double> elem4(bis4.get_block_index_dims(), pgtid);
    block_labeling<4> &bl4 = elem4.get_labeling();
    se_label<2, double> elem2(bis2.get_block_index_dims(), pgtid);
    block_labeling<2> &bl2 = elem2.get_labeling();
    se_label<2, double> elem2_ref(bis2.get_block_index_dims(), pgtid);
    block_labeling<2> &bl2_ref = elem2_ref.get_labeling();
    bl4.assign(msk4, 0, ap);
    bl4.assign(msk4, 1, ap);
    bl4.assign(msk4, 2, app);
    bl2.assign(msk2, 0, ap);
    bl2.assign(msk2, 1, ap);
    bl2.assign(msk2, 2, app);
    bl2_ref.assign(msk2, 0, ap);
    bl2_ref.assign(msk2, 1, ap);
    bl2_ref.assign(msk2, 2, app);
    msk4[0] = true; msk4[1] = true; msk4[2] = false; msk4[3] = false;
    msk2[1] = true; msk2[0] = false;
    bl4.assign(msk4, 0, ap);
    bl4.assign(msk4, 1, app);
    bl4.assign(msk4, 2, app);
    bl2.assign(msk2, 0, ap);
    bl2.assign(msk2, 1, app);
    bl2.assign(msk2, 2, app);
    bl2_ref.assign(msk2, 0, ap);
    bl2_ref.assign(msk2, 1, app);
    bl2_ref.assign(msk2, 2, app);
    elem4.set_rule(ap);
    elem2.set_rule(ap);
    elem2_ref.set_rule(product_table_i::k_invalid);

    block_tensor<4, double, allocator_t> bta(bis4);
    block_tensor<2, double, allocator_t> btb(bis2);

    {
        block_tensor_ctrl<4, double> ca(bta);
        ca.req_symmetry().insert(elem4);
        block_tensor_ctrl<2, double> cb(btb);
        cb.req_symmetry().insert(elem2);
    }

    symmetry<2, double> sym_ref(bis2);
    sym_ref.insert(elem2_ref);

    dense_tensor<4, double, allocator_t> ta(dims4);
    dense_tensor<2, double, allocator_t> tb(dims2), tb_ref(dims2);

    sequence<4, size_t> msk(1);
    msk[0] = 1; msk[1] = 1; msk[2] = 2; msk[3] = 2;

    //  Fill in random data
    btod_random<4>().perform(bta);
    bta.set_immutable();

    //  Prepare the reference
    tod_btconv<4>(bta).perform(ta);

    tensor_transf<2, double> tr(permutation<2>().permute(0, 1));
    if (add) {
        //  Fill in random data
        btod_random<2>().perform(btb);

        //  Prepare the reference
        tod_btconv<2>(btb).perform(tb_ref);

        tod_diag<4, 2>(ta, msk, tr).perform(false, tb_ref);

        //  Invoke the operation
        btod_diag<4, 2>(bta, msk, tr.get_perm()).perform(btb, 1.0);
    }
    else {
        tod_diag<4, 2>(ta, msk, tr).perform(true, tb_ref);

        //  Invoke the operation
        btod_diag<4, 2>(bta, msk, tr.get_perm()).perform(btb);
    }

    tod_btconv<2>(btb).perform(tb);

    //  Compare against the reference
    {
    block_tensor_ctrl<2, double> cb(btb);
    compare_ref<2>::compare(testname, cb.req_const_symmetry(), sym_ref);
    }
    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    }

    } catch(exception &e) {
        product_table_container::get_instance().erase(pgtid);
        fail_test(testname, __FILE__, __LINE__, e.what());
    } catch(...) {
        product_table_container::get_instance().erase(pgtid);
        throw;
    }

    product_table_container::get_instance().erase(pgtid);
}


/** \test Extract diagonal: \f$ b_{ij} = a_{ijji} \f$, non-zero tensor,
        multiple blocks, perm and partition symmetry
 **/
void btod_diag_test::test_sym_11(bool add) throw(libtest::test_exception) {

    static const char *testname = "btod_diag_test::test_sym_11(bool)";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<2> i2a, i2b;
    i2b[0] = i2b[1] = 9;
    libtensor::index<4> i4a, i4b;
    i4b[0] = i4b[1] = i4b[2] = i4b[3] = 9;
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    dimensions<4> dims4(index_range<4>(i4a, i4b));
    block_index_space<2> bis2(dims2);
    block_index_space<4> bis4(dims4);

    mask<2> msk2;
    msk2[0] = msk2[1] = true;
    mask<4> msk4;
    msk4[0] = msk4[1] = msk4[2] = msk4[3] = true;
    bis2.split(msk2, 3);
    bis2.split(msk2, 5);
    bis2.split(msk2, 8);
    bis4.split(msk4, 3);
    bis4.split(msk4, 5);
    bis4.split(msk4, 8);

    scalar_transf<double> tr;
    se_perm<4, double> se4a(permutation<4>().permute(0, 1), tr);
    se_perm<4, double> se4b(permutation<4>().permute(0, 2).permute(1, 3), tr);
    se_perm<4, double> se4c(permutation<4>().permute(2, 3), tr);
    se_part<4, double> se4d(bis4, msk4, 2);
    se_perm<2, double> se2a(permutation<2>().permute(0, 1), tr);
    se_part<2, double> se2b(bis2, msk2, 2);
    i4b[0] = i4b[1] = i4b[2] = i4b[3] = 1;
    se4d.add_map(i4a, i4b, tr); // 0000->1111
    i4a[1] = i4a[3] = 1; i4b[0] = i4b[3] = 0;
    se4d.add_map(i4a, i4b, tr); // 0101->0110
    i4a[0] = 1; i4a[1] = 0;
    se4d.add_map(i4b, i4a, tr); // 0110->1001
    i4b[0] = 1; i4b[1] = 0;
    se4d.add_map(i4a, i4b, tr); // 1001->1010
    i4a[0] = 0;
    se4d.mark_forbidden(i4a); // 0001->x
    i4a[2] = 1; i4a[3] = 0;
    se4d.mark_forbidden(i4a); // 0010->x
    i4a[1] = 1; i4a[2] = 0;
    se4d.mark_forbidden(i4a); // 0100->x
    i4a[0] = 1; i4a[1] = 0;
    se4d.mark_forbidden(i4a); // 1000->x
    i4a[1] = 1;
    se4d.mark_forbidden(i4a); // 1100->x
    i4a[2] = 1;
    se4d.mark_forbidden(i4a); // 1110->x
    i4a[2] = 0; i4a[3] = 1;
    se4d.mark_forbidden(i4a); // 1101->x
    i4a[1] = 0; i4a[2] = 1;
    se4d.mark_forbidden(i4a); // 1011->x
    i4a[0] = 0; i4a[1] = 1;
    se4d.mark_forbidden(i4a); // 0111->x
    i4a[1] = 0;
    se4d.mark_forbidden(i4a); // 0011->x
    i2b[0] = i2b[1] = 1;
    se2b.add_map(i2a, i2b, tr); // 00->11
    i2a[1] = 1; i2b[1] = 0;
    se2b.add_map(i2a, i2b, tr); // 01->10

    block_tensor<4, double, allocator_t> bta(bis4);
    block_tensor<2, double, allocator_t> btb(bis2);

    symmetry<2, double> sym_ref(bis2);
    {
        block_tensor_ctrl<4, double> ca(bta);
        symmetry<4, double> &syma = ca.req_symmetry();
        syma.insert(se4a);
        syma.insert(se4b);
        syma.insert(se4c);
        syma.insert(se4d);
        block_tensor_ctrl<2, double> cb(btb);
        symmetry<2, double> &symb = cb.req_symmetry();
        symb.insert(se2a);
        symb.insert(se2b);
        sym_ref.insert(se2a);
        sym_ref.insert(se2b);
    }

    dense_tensor<4, double, allocator_t> ta(dims4);
    dense_tensor<2, double, allocator_t> tb(dims2), tb_ref(dims2);

    sequence<4, size_t> msk(0);
    msk[0] = 1; msk[1] = 2; msk[2] = 2; msk[3] = 1;

    //  Fill in random data
    btod_random<4>().perform(bta);
    bta.set_immutable();

    //  Prepare the reference
    tod_btconv<4>(bta).perform(ta);

    if (add) {
        //  Fill in random data
        btod_random<2>().perform(btb);

        //  Prepare the reference
        tod_btconv<2>(btb).perform(tb_ref);

        tod_diag<4, 2>(ta, msk).perform(false, tb_ref);

        //  Invoke the operation
        btod_diag<4, 2>(bta, msk).perform(btb, 1.0);
    }
    else {
        tod_diag<4, 2>(ta, msk).perform(true, tb_ref);

        //  Invoke the operation
        btod_diag<4, 2>(bta, msk).perform(btb);
    }

    tod_btconv<2>(btb).perform(tb);

    //  Compare against the reference
    {
    block_tensor_ctrl<2, double> cb(btb);
    compare_ref<2>::compare(testname, cb.req_const_symmetry(), sym_ref);
    }
    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
