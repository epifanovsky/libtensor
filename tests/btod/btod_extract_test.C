#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/btod/btod_extract.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/dense_tensor/tod_btconv.h>
#include <libtensor/dense_tensor/tod_extract.h>
#include <libtensor/symmetry/se_perm.h>
#include "btod_extract_test.h"
#include "../compare_ref.h"

namespace libtensor {


void btod_extract_test::perform() throw(libtest::test_exception) {

    allocator<double>::vmm().init(16, 16, 16777216, 16777216);

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
    test_11();
    test_12a();
    test_12b();
    test_12c();
    test_13a();

    } catch(...) {
        allocator<double>::vmm().shutdown();
        throw;
    }

    allocator<double>::vmm().shutdown();
}


/** \test Extract a single column b_j = a_ij where i is constant
 **/
void btod_extract_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "btod_extract_test::test_1()";

    typedef std_allocator<double> allocator_t;

    try {

    index<1> i1a, i1b;
    i1b[0] = 10;
    index<2> i2a, i2b;
    i2b[0] = 10; i2b[1] = 10;
    dimensions<1> dims1(index_range<1>(i1a, i1b));
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    block_index_space<1> bis1(dims1);
    block_index_space<2> bis2(dims2);

    block_tensor<2, double, allocator_t> bta(bis2);
    block_tensor<1, double, allocator_t> btb(bis1);

    dense_tensor<2, double, allocator_t> ta(dims2);
    dense_tensor<1, double, allocator_t> tb(dims1), tb_ref(dims1);

    mask<2> msk;
    msk[0] = false; msk[1] = true;
    index<2> idx;
    idx[0] = 2; idx [1] = 0;
    index<2> idxbl;
    idxbl[0] = 0; idxbl[1] = 0;
    index<2> idxibl;
    idxibl[0] = idx[0]; idxibl[1] = idx[1];

    //  Fill in random data
    btod_random<2>().perform(bta);
    bta.set_immutable();

    //  Prepare the reference
    tod_btconv<2>(bta).perform(ta);
    tod_extract<2, 1>(ta, msk, idx).perform(tb_ref);

    //  Invoke the operation
    btod_extract<2, 1>(bta, msk, idxbl, idxibl).perform(btb);
    tod_btconv<1>(btb).perform(tb);

    //  Compare against the reference
    compare_ref<1>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Extract a matrix from the 3rd order tensor \f$ b_{ia} = a_{iba} \f$
 **/
void btod_extract_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "btod_extract_test::test_2()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i2a, i2b;
    i2b[0] = 10; i2b[1] = 5;
    index<3> i3a, i3b;
    i3b[0] = 10; i3b[1] = 10; i3b[2] = 5;
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    dimensions<3> dims3(index_range<3>(i3a, i3b));
    block_index_space<2> bis2(dims2);
    block_index_space<3> bis3(dims3);

    block_tensor<3, double, allocator_t> bta(bis3);
    block_tensor<2, double, allocator_t> btb(bis2);

    dense_tensor<3, double, allocator_t> ta(dims3);
    dense_tensor<2, double, allocator_t> tb(dims2), tb_ref(dims2);

    mask<3> msk;
    msk[0] = true; msk[1] = false; msk[2] = true;

    index<3> idx;
    idx[0] = 0; idx [1] = 4; idx[2] =0;
    index<3> idxbl;
    idxbl[0] = 0; idxbl[1] = 0;idxbl[2] = 0;
    index<3> idxibl;
    idxibl[0] = idx[0]; idxibl[1] = idx[1];idxibl[2] = idx[2];

    //  Fill in random data
    btod_random<3>().perform(bta);
    bta.set_immutable();

    //  Prepare the reference
    tod_btconv<3>(bta).perform(ta);
    tod_extract<3, 1>(ta, msk, idx).perform(tb_ref);

    //  Invoke the operation
    btod_extract<3, 1>(bta, msk, idxbl, idxibl).perform(btb);
    tod_btconv<2>(btb).perform(tb);

    //  Compare against the reference
    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Extract a 3rd order tensor from the 5rd order tensor
 * \f$ b_{jkm} = a_{ijklm} \f$
 **/

void btod_extract_test::test_3() throw(libtest::test_exception) {

    static const char *testname = "btod_extract_test::test_3()";

    typedef std_allocator<double> allocator_t;

    try {

    index<3> i3a, i3b;
    i3b[0] = 10; i3b[1] = 5;i3b[2]=7;
    index<5> i5a, i5b;
    i5b[0] = 12; i5b[1] = 10; i5b[2] = 5;i5b[3] = 8;i5b[4] = 7;
    dimensions<3> dims3(index_range<3>(i3a, i3b));
    dimensions<5> dims5(index_range<5>(i5a, i5b));
    block_index_space<3> bis3(dims3);
    block_index_space<5> bis5(dims5);

    block_tensor<5, double, allocator_t> bta(bis5);
    block_tensor<3, double, allocator_t> btb(bis3);

    dense_tensor<5, double, allocator_t> ta(dims5);
    dense_tensor<3, double, allocator_t> tb(dims3), tb_ref(dims3);

    mask<5> msk;
    msk[0] = false; msk[1] = true;msk[2] = true;msk[3] = false; msk[4] = true;

    index<5> idx;
    idx[0] = 11; idx [1] = 0; idx[2] = 0;idx[3] = 1;idx[4] = 0;
    index<5> idxbl;
    idxbl[0] = 0; idxbl[1] = 0;idxbl[2] = 0;idxbl[3]=0;idxbl[4]=0;
    index<5> idxibl;
    idxibl[0] = idx[0]; idxibl[1] = idx[1];idxibl[2] = idx[2];
    idxibl[3] = idx[3]; idxibl[4] = idx[4];

    //  Fill in random data
    btod_random<5>().perform(bta);
    bta.set_immutable();

    //  Prepare the reference
    tod_btconv<5>(bta).perform(ta);
    tod_extract<5, 2>(ta, msk, idx).perform(tb_ref);

    //  Invoke the operation
    btod_extract<5, 2>(bta, msk, idxbl, idxibl).perform(btb);
    tod_btconv<3>(btb).perform(tb);

    //  Compare against the reference
    compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}
/** \test Extract a matrix from the 5rd order tensor
 * \f$ b_{jl} = a_{ijklm} \f$
 **/
void btod_extract_test::test_4() throw(libtest::test_exception) {

    static const char *testname = "btod_extract_test::test_4()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i2a, i2b;
    i2b[0] = 11; i2b[1] = 8;
    index<5> i5a, i5b;
    i5b[0] = 7; i5b[1] = 11; i5b[2] = 20;i5b[3] = 8;i5b[4] = 7;
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    dimensions<5> dims5(index_range<5>(i5a, i5b));
    block_index_space<2> bis2(dims2);
    block_index_space<5> bis5(dims5);

    block_tensor<5, double, allocator_t> bta(bis5);
    block_tensor<2, double, allocator_t> btb(bis2);

    dense_tensor<5, double, allocator_t> ta(dims5);
    dense_tensor<2, double, allocator_t> tb(dims2), tb_ref(dims2);

    mask<5> msk;
    msk[0] = false; msk[1] = true;msk[2] = false;msk[3] = true; msk[4] = false;

    index<5> idx;
    idx[0] = 0; idx [1] = 0; idx[2] = 19;idx[3] = 0;idx[4] = 5;
    index<5> idxbl;
    idxbl[0] = 0; idxbl[1] = 0;idxbl[2] = 0;idxbl[3]=0;idxbl[4]=0;
    index<5> idxibl;
    idxibl[0] = idx[0]; idxibl[1] = idx[1];idxibl[2] = idx[2];
    idxibl[3] = idx[3]; idxibl[4] = idx[4];

    //  Fill in random data
    btod_random<5>().perform(bta);
    bta.set_immutable();

    //  Prepare the reference
    tod_btconv<5>(bta).perform(ta);
    tod_extract<5, 3>(ta, msk, idx).perform(tb_ref);

    //  Invoke the operation
    btod_extract<5, 3>(bta, msk, idxbl, idxibl).perform(btb);
    tod_btconv<2>(btb).perform(tb);

    //  Compare against the reference
    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Extract a vector from the matrix with splitting
 * \f$ b_{j} = a_{ij} \f$
 **/
void btod_extract_test::test_5() throw(libtest::test_exception) {

    static const char *testname = "btod_extract_test::test_5()";

    typedef std_allocator<double> allocator_t;

    try {

    index<1> i1a, i1b;
    i1b[0] = 10;
    index<2> i2a, i2b;
    i2b[0] = 10; i2b[1] = 10;
    dimensions<1> dims1(index_range<1>(i1a, i1b));
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    block_index_space<1> bis1(dims1);
    block_index_space<2> bis2(dims2);

    mask<1> splmsk1; splmsk1[0] = true;
    bis1.split(splmsk1, 4);

    mask<2> splmsk2; splmsk2[0] = true;splmsk2[1] = true;
    bis2.split(splmsk2, 4);



    block_tensor<2, double, allocator_t> bta(bis2);
    block_tensor<1, double, allocator_t> btb(bis1);

    dense_tensor<2, double, allocator_t> ta(dims2);
    dense_tensor<1, double, allocator_t> tb(dims1), tb_ref(dims1);

    mask<2> msk;
    msk[0] = false; msk[1] = true;
    index<2> idx;
    idx[0] = 7; idx [1] = 0;
    index<2> idxbl;
    idxbl[0] = 1; idxbl[1] = 0;
    index<2> idxibl;
    idxibl[0] = idx[0]-4; idxibl[1] = idx[1];

    //  Fill in random data
    btod_random<2>().perform(bta);
    bta.set_immutable();

    //  Prepare the reference
    tod_btconv<2>(bta).perform(ta);
    tod_extract<2, 1>(ta, msk, idx).perform(tb_ref);

    //  Invoke the operation
    btod_extract<2, 1>(bta, msk, idxbl, idxibl).perform(btb);
    tod_btconv<1>(btb).perform(tb);

    //  Compare against the reference
    compare_ref<1>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Extract a matrix from the 3rd order tensor with splitting
 * \f$ b_{ik} = a_{ijk} \f$
 **/
void btod_extract_test::test_6() throw(libtest::test_exception) {

    static const char *testname = "btod_extract_test::test_6()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i2a, i2b;
    i2b[0] = 10; i2b[1] = 5;
    index<3> i3a, i3b;
    i3b[0] = 10; i3b[1] = 10; i3b[2] = 5;
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    dimensions<3> dims3(index_range<3>(i3a, i3b));
    block_index_space<2> bis2(dims2);
    block_index_space<3> bis3(dims3);

    mask<2> splmsk2; splmsk2[0] = true;splmsk2[1] = false;
    bis2.split(splmsk2, 5);

    mask<3> splmsk3; splmsk3[0] = true;splmsk3[1] = true;splmsk3[2] = false;
    bis3.split(splmsk3, 5);

    mask<2> splmsk22; splmsk22[0] = false;splmsk22[1] = true;
    bis2.split(splmsk22, 3);

    mask<3> splmsk33; splmsk33[0] = false;splmsk33[1] = false;splmsk33[2] = true;
    bis3.split(splmsk33, 3);

    block_tensor<3, double, allocator_t> bta(bis3);
    block_tensor<2, double, allocator_t> btb(bis2);

    dense_tensor<3, double, allocator_t> ta(dims3);
    dense_tensor<2, double, allocator_t> tb(dims2), tb_ref(dims2);

    mask<3> msk;
    msk[0] = true; msk[1] = false;msk[2] = true;

    index<3> idx;
    idx[0] = 0; idx [1] = 7; idx[2] =0;
    index<3> idxbl;
    idxbl[0] = 0; idxbl[1] = 1;idxbl[2] = 0;
    index<3> idxibl;
    idxibl[0] = idx[0]; idxibl[1] = idx[1]-5;idxibl[2] = idx[2];

    //  Fill in random data
    btod_random<3>().perform(bta);
    bta.set_immutable();

    //  Prepare the reference
    tod_btconv<3>(bta).perform(ta);
    tod_extract<3, 1>(ta, msk, idx).perform(tb_ref);

    //  Invoke the operation
    btod_extract<3, 1>(bta, msk, idxbl, idxibl).perform(btb);
    tod_btconv<2>(btb).perform(tb);

    //  Compare against the reference
    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Extract a matrix from the 4rd order tensor with splitting
 * \f$ b_{il} = a_{ijkl} \f$
 **/


void btod_extract_test::test_7() throw(libtest::test_exception) {

    static const char *testname = "btod_extract_test::test_7()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i2a, i2b;
    i2b[0] = 10; i2b[1] = 5;
    index<4> i4a, i4b;
    i4b[0] = 10; i4b[1] = 10; i4b[2] = 5;i4b[3] = 5;
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    dimensions<4> dims4(index_range<4>(i4a, i4b));
    block_index_space<2> bis2(dims2);
    block_index_space<4> bis4(dims4);

    mask<2> splmsk2; splmsk2[0] = true;splmsk2[1] = false;
    bis2.split(splmsk2, 5);

    mask<4> splmsk3; splmsk3[0] = true;splmsk3[1] = true;splmsk3[2] = false;
    splmsk3[3] = false;
    bis4.split(splmsk3, 5);

    mask<2> splmsk22; splmsk22[0] = false;splmsk22[1] = true;
    bis2.split(splmsk22, 3);

    mask<4> splmsk33; splmsk33[0] = false;splmsk33[1] = false;splmsk33[2] = true;
    splmsk33[3] = true;
    bis4.split(splmsk33, 3);

    block_tensor<4, double, allocator_t> bta(bis4);
    block_tensor<2, double, allocator_t> btb(bis2);

    dense_tensor<4, double, allocator_t> ta(dims4);
    dense_tensor<2, double, allocator_t> tb(dims2), tb_ref(dims2);

    mask<4> msk;
    msk[0] = true; msk[1] = false;msk[2] = false;msk[3] = true;

    index<4> idx;
    idx[0] = 0; idx [1] = 7; idx[2] =2;idx[3] = 0;
    index<4> idxbl;
    idxbl[0] = 0; idxbl[1] = 1;idxbl[2] = 0;idxbl[3] = 0;
    index<4> idxibl;
    idxibl[0] = idx[0]; idxibl[1] = idx[1] -5;idxibl[2] = idx[2];
    idxibl[3] = idx[3];

    //  Fill in random data
    btod_random<4>().perform(bta);
    bta.set_immutable();

    //  Prepare the reference
    tod_btconv<4>(bta).perform(ta);
    tod_extract<4, 2>(ta, msk, idx).perform(tb_ref);

    //  Invoke the operation
    btod_extract<4, 2>(bta, msk, idxbl, idxibl).perform(btb);
    tod_btconv<2>(btb).perform(tb);

    //  Compare against the reference
    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Extract a matrix from the 4rd order tensor with splitting
 * \f$ b_{il} = a_{ijkl} \f$
 **/
void btod_extract_test::test_8() throw(libtest::test_exception) {

    static const char *testname = "btod_extract_test::test_8()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i2a, i2b;
    i2b[0] = 12; i2b[1] = 12;
    index<4> i4a, i4b;
    i4b[0] = 12; i4b[1] = 12; i4b[2] = 12;i4b[3] = 12;
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    dimensions<4> dims4(index_range<4>(i4a, i4b));
    block_index_space<2> bis2(dims2);
    block_index_space<4> bis4(dims4);

    mask<2> splmsk2; splmsk2[0] = true;splmsk2[1] = true;
    bis2.split(splmsk2, 3);
    bis2.split(splmsk2, 6);
    bis2.split(splmsk2, 9);
    bis2.split(splmsk2, 12);

    mask<4> splmsk3; splmsk3[0] = true;splmsk3[1] = true;splmsk3[2] = true;
    splmsk3[3] = true;
    bis4.split(splmsk3, 3);
    bis4.split(splmsk3, 6);
    bis4.split(splmsk3, 9);
    bis4.split(splmsk3, 12);

    block_tensor<4, double, allocator_t> bta(bis4);
    block_tensor<2, double, allocator_t> btb(bis2);

    dense_tensor<4, double, allocator_t> ta(dims4);
    dense_tensor<2, double, allocator_t> tb(dims2), tb_ref(dims2);

    mask<4> msk;
    msk[0] = true; msk[1] = false;msk[2] = false;msk[3] = true;

    index<4> idx;
    idx[0] = 0; idx [1] = 5; idx[2] =11;idx[3] = 0;
    index<4> idxbl;
    idxbl[0] = 0; idxbl[1] = 1;idxbl[2] = 3;idxbl[3] = 0;
    index<4> idxibl;
    idxibl[0] = idx[0]; idxibl[1] = idx[1] -3;idxibl[2] = idx[2] - 9;
    idxibl[3] = idx[3];

    //  Fill in random data
    btod_random<4>().perform(bta);
    bta.set_immutable();

    //  Prepare the reference
    tod_btconv<4>(bta).perform(ta);
    tod_extract<4, 2>(ta, msk, idx).perform(tb_ref);

    //  Invoke the operation
    btod_extract<4, 2>(bta, msk, idxbl, idxibl).perform(btb);
    tod_btconv<2>(btb).perform(tb);

    //  Compare against the reference
    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}



/** \test Extract a matrix from the 3rd order tensor with splitting and
 * permutation \f$ b_{ki} = a_{ijk} \f$
 **/
void btod_extract_test::test_9() throw(libtest::test_exception) {

    static const char *testname = "btod_extract_test::test_9()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i2a, i2b;
    i2b[0] = 5; i2b[1] = 10;
    index<3> i3a, i3b;
    i3b[0] = 10; i3b[1] = 10; i3b[2] = 5;
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    dimensions<3> dims3(index_range<3>(i3a, i3b));
    block_index_space<2> bis2(dims2);
    block_index_space<3> bis3(dims3);

    block_tensor<3, double, allocator_t> bta(bis3);
    block_tensor<2, double, allocator_t> btb(bis2);

    dense_tensor<3, double, allocator_t> ta(dims3);
    dense_tensor<2, double, allocator_t> tb(dims2), tb_ref(dims2);

    mask<3> msk;
    msk[0] = true; msk[1] = false; msk[2] = true;

    index<3> idx;
    idx[0] = 0; idx [1] = 4; idx[2] =0;
    index<3> idxbl;
    idxbl[0] = 0; idxbl[1] = 0;idxbl[2] = 0;
    index<3> idxibl;
    idxibl[0] = idx[0]; idxibl[1] = idx[1];idxibl[2] = idx[2];

    //  Fill in random data
    btod_random<3>().perform(bta);
    bta.set_immutable();

    // Set the permutation
    permutation<2> perm;
    perm.permute(0, 1);

    //  Prepare the reference
    tod_btconv<3>(bta).perform(ta);
    tod_extract<3, 1>(ta, msk, perm, idx).perform(tb_ref);

    //  Invoke the operation
    btod_extract<3, 1>(bta, msk,perm, idxbl, idxibl).perform(btb);
    tod_btconv<2>(btb).perform(tb);

    //  Compare against the reference
    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Extract a matrix from the 4rd order tensor with splitting and symmetry
        \f$ b_{il} = a_{ijkl} \f$
 **/
void btod_extract_test::test_10() throw(libtest::test_exception) {

    static const char *testname = "btod_extract_test::test_10()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i2a, i2b;
    i2b[0] = 12; i2b[1] = 12;
    index<4> i4a, i4b;
    i4b[0] = 12; i4b[1] = 12; i4b[2] = 12;i4b[3] = 12;
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    dimensions<4> dims4(index_range<4>(i4a, i4b));
    block_index_space<2> bis2(dims2);
    block_index_space<4> bis4(dims4);

    mask<2> splmsk2; splmsk2[0] = true; splmsk2[1] = true;
    bis2.split(splmsk2, 3);
    bis2.split(splmsk2, 6);
    bis2.split(splmsk2, 9);
    bis2.split(splmsk2, 12);

    mask<4> splmsk3; splmsk3[0] = true; splmsk3[1] = true; splmsk3[2] = true;
    splmsk3[3] = true;
    bis4.split(splmsk3, 3);
    bis4.split(splmsk3, 6);
    bis4.split(splmsk3, 9);
    bis4.split(splmsk3, 12);

    block_tensor<4, double, allocator_t> bta(bis4);
    block_tensor<2, double, allocator_t> btb(bis2);

    //  Add symmetries

    permutation<4> p30, p21;
    p30.permute(0, 3);
    p21.permute(2, 1);
    scalar_transf<double> tr0, tr1(-1.);
    se_perm<4, double> sp30(p30, tr0);
    se_perm<4, double> ap21(p21, tr1);
    block_tensor_ctrl<4, double> cbta(bta);
    cbta.req_symmetry().insert(sp30);
    cbta.req_symmetry().insert(ap21);

    permutation<2> p10;
    p10.permute(0, 1);
    se_perm<2, double> sp10(p10, tr0);
    block_tensor_ctrl<2, double> cbtb(btb);
    cbtb.req_symmetry().insert(sp10);

    dense_tensor<4, double, allocator_t> ta(dims4);
    dense_tensor<2, double, allocator_t> tb(dims2), tb_ref(dims2);

    mask<4> msk;
    msk[0] = true; msk[1] = false; msk[2] = false; msk[3] = true;

    index<4> idx;
    idx[0] = 0; idx[1] = 5; idx[2] =11; idx[3] = 0;
    index<4> idxbl;
    idxbl[0] = 0; idxbl[1] = 1; idxbl[2] = 3; idxbl[3] = 0;
    index<4> idxibl;
    idxibl[0] = idx[0]; idxibl[1] = idx[1] -3; idxibl[2] = idx[2] - 9;
    idxibl[3] = idx[3];

    //  Fill in random data

    btod_random<4>().perform(bta);
    bta.set_immutable();

    //  Prepare the reference

    tod_btconv<4>(bta).perform(ta);
    tod_extract<4, 2>(ta, msk, idx).perform(tb_ref);

    //  Invoke the operation
    btod_extract<4, 2>(bta, msk, idxbl, idxibl).perform(btb);
    tod_btconv<2>(btb).perform(tb);

    //  Compare against the reference
    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}



/** \test Extract a matrix from the 3rd order tensor with splitting,
 * permutation, and symmetry \f$ b_{ki} = a_{ijk} \f$
 **/
void btod_extract_test::test_11() throw(libtest::test_exception) {

    static const char *testname = "btod_extract_test::test_11()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i2a, i2b;
    i2b[0] = 5; i2b[1] = 10;
    index<3> i3a, i3b;
    i3b[0] = 10; i3b[1] = 10; i3b[2] = 5;
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    dimensions<3> dims3(index_range<3>(i3a, i3b));
    block_index_space<2> bis2(dims2);
    block_index_space<3> bis3(dims3);

    block_tensor<3, double, allocator_t> bta(bis3);
    block_tensor<2, double, allocator_t> btb(bis2);

    //  Add symmetry

    permutation<3> p21;
    p21.permute(0, 1);
    scalar_transf<double> tr0, tr1(-1.);
    se_perm<3, double> ap21(p21, tr1);
    block_tensor_ctrl<3, double> cbta(bta);
    cbta.req_symmetry().insert(ap21);

    dense_tensor<3, double, allocator_t> ta(dims3);
    dense_tensor<2, double, allocator_t> tb(dims2), tb_ref(dims2);

    mask<3> msk;
    msk[0] = true; msk[1] = false; msk[2] = true;

    index<3> idx;
    idx[0] = 0; idx [1] = 4; idx[2] =0;
    index<3> idxbl;
    idxbl[0] = 0; idxbl[1] = 0; idxbl[2] = 0;
    index<3> idxibl;
    idxibl[0] = idx[0]; idxibl[1] = idx[1]; idxibl[2] = idx[2];

    //  Fill in random data
    btod_random<3>().perform(bta);
    bta.set_immutable();

    // Set the permutation
    permutation<2> perm;
    perm.permute(0, 1);

    //  Prepare the reference
    tod_btconv<3>(bta).perform(ta);
    tod_extract<3, 1>(ta, msk, perm, idx).perform(tb_ref);

    //  Invoke the operation
    btod_extract<3, 1>(bta, msk,perm, idxbl, idxibl).perform(btb);
    tod_btconv<2>(btb).perform(tb);

    //  Compare against the reference
    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Extract a matrix from a fourth-order tensor with splitting
        and perm symmetry
          \f$ b_{jl} = a_{ijkl} \qquad a_{ijkl} = a_{klij} \f$
 **/
void btod_extract_test::test_12a() throw(libtest::test_exception) {

    static const char *testname = "btod_extract_test::test_12a()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i2a, i2b;
    i2b[0] = 12; i2b[1] = 12;
    index<4> i4a, i4b;
    i4b[0] = 12; i4b[1] = 12; i4b[2] = 12; i4b[3] = 12;
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    dimensions<4> dims4(index_range<4>(i4a, i4b));
    block_index_space<2> bis2(dims2);
    block_index_space<4> bis4(dims4);

    mask<2> splmsk2;
    splmsk2[0] = true; splmsk2[1] = true;
    bis2.split(splmsk2, 3);
    bis2.split(splmsk2, 6);
    bis2.split(splmsk2, 9);
    bis2.split(splmsk2, 12);

    mask<4> splmsk3;
    splmsk3[0] = true; splmsk3[1] = true; splmsk3[2] = true;
    splmsk3[3] = true;
    bis4.split(splmsk3, 3);
    bis4.split(splmsk3, 6);
    bis4.split(splmsk3, 9);
    bis4.split(splmsk3, 12);

    block_tensor<4, double, allocator_t> bta(bis4);
    block_tensor<2, double, allocator_t> btb(bis2);

    //  Add perm symmetry

    block_tensor_ctrl<4, double> cbta(bta);
    scalar_transf<double> tr0, tr1(-1.);
    cbta.req_symmetry().insert(se_perm<4, double>(
        permutation<4>().permute(2, 3), tr1));
    cbta.req_symmetry().insert(se_perm<4, double>(
        permutation<4>().permute(0, 2).permute(1, 3), tr0));

    dense_tensor<4, double, allocator_t> ta(dims4);
    dense_tensor<2, double, allocator_t> tb(dims2), tb_ref(dims2);

    mask<4> msk;
    msk[0] = false; msk[1] = true; msk[2] = false; msk[3] = true;

    index<4> idx;
    idx[0] = 5; idx[1] = 0; idx[2] = 11; idx[3] = 0;
    index<4> idxbl;
    idxbl[0] = 1; idxbl[1] = 0; idxbl[2] = 3; idxbl[3] = 0;
    index<4> idxibl;
    idxibl[0] = idx[0] - 3; idxibl[1] = idx[1]; idxibl[2] = idx[2] - 9;
    idxibl[3] = idx[3];

    //  Fill in random data

    btod_random<4>().perform(bta);
    bta.set_immutable();

    //  Prepare the reference

    tod_btconv<4>(bta).perform(ta);
    tod_extract<4, 2>(ta, msk, idx).perform(tb_ref);

    //  Invoke the operation

    btod_extract<4, 2>(bta, msk, idxbl, idxibl).perform(btb);
    tod_btconv<2>(btb).perform(tb);

    //  Compare against the reference

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Extract a matrix from a fourth-order tensor with splitting
        and perm symmetry (additive, b has perm symmetry)
          \f$ b_{jl} = b_{jl} + a_{ijkl} \qquad a_{ijkl} = a_{klij} \f$
 **/
void btod_extract_test::test_12b() throw(libtest::test_exception) {

    static const char *testname = "btod_extract_test::test_12b()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i2a, i2b;
    i2b[0] = 12; i2b[1] = 12;
    index<4> i4a, i4b;
    i4b[0] = 12; i4b[1] = 12; i4b[2] = 12; i4b[3] = 12;
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    dimensions<4> dims4(index_range<4>(i4a, i4b));
    block_index_space<2> bis2(dims2);
    block_index_space<4> bis4(dims4);

    mask<2> splmsk2;
    splmsk2[0] = true; splmsk2[1] = true;
    bis2.split(splmsk2, 3);
    bis2.split(splmsk2, 6);
    bis2.split(splmsk2, 9);
    bis2.split(splmsk2, 12);

    mask<4> splmsk3;
    splmsk3[0] = true; splmsk3[1] = true; splmsk3[2] = true;
    splmsk3[3] = true;
    bis4.split(splmsk3, 3);
    bis4.split(splmsk3, 6);
    bis4.split(splmsk3, 9);
    bis4.split(splmsk3, 12);

    block_tensor<4, double, allocator_t> bta(bis4);
    block_tensor<2, double, allocator_t> btb(bis2);

    //  Add perm symmetry

    block_tensor_ctrl<4, double> cbta(bta);
    scalar_transf<double> tr0, tr1(-1.);
    cbta.req_symmetry().insert(se_perm<4, double>(
        permutation<4>().permute(2, 3), tr1));
    cbta.req_symmetry().insert(se_perm<4, double>(
        permutation<4>().permute(0, 2).permute(1, 3), tr0));

    block_tensor_ctrl<2, double> cbtb(btb);
    cbtb.req_symmetry().insert(se_perm<2, double>(
        permutation<2>().permute(0, 1), tr0));

    dense_tensor<4, double, allocator_t> ta(dims4);
    dense_tensor<2, double, allocator_t> tb(dims2), tb_ref(dims2);

    mask<4> msk;
    msk[0] = false; msk[1] = true; msk[2] = false; msk[3] = true;

    index<4> idx;
    idx[0] = 5; idx[1] = 0; idx[2] = 11; idx[3] = 0;
    index<4> idxbl;
    idxbl[0] = 1; idxbl[1] = 0; idxbl[2] = 3; idxbl[3] = 0;
    index<4> idxibl;
    idxibl[0] = idx[0] - 3; idxibl[1] = idx[1]; idxibl[2] = idx[2] - 9;
    idxibl[3] = idx[3];

    //  Fill in random data

    btod_random<4>().perform(bta);
    btod_random<2>().perform(btb);
    bta.set_immutable();

    //  Prepare the reference

    tod_btconv<4>(bta).perform(ta);
    tod_btconv<2>(btb).perform(tb_ref);
    tod_extract<4, 2>(ta, msk, idx).perform(tb_ref, 2.3);

    //  Invoke the operation

    btod_extract<4, 2>(bta, msk, idxbl, idxibl).perform(btb, 2.3);
    tod_btconv<2>(btb).perform(tb);

    //  Compare against the reference

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Extract a matrix from a fourth-order tensor with splitting
        and perm symmetry (additive, b has no symmetry)
          \f$ b_{jl} = b_{jl} + a_{ijkl} \qquad a_{ijkl} = a_{klij} \f$
 **/
void btod_extract_test::test_12c() throw(libtest::test_exception) {

    static const char *testname = "btod_extract_test::test_12c()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i2a, i2b;
    i2b[0] = 12; i2b[1] = 12;
    index<4> i4a, i4b;
    i4b[0] = 12; i4b[1] = 12; i4b[2] = 12; i4b[3] = 12;
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    dimensions<4> dims4(index_range<4>(i4a, i4b));
    block_index_space<2> bis2(dims2);
    block_index_space<4> bis4(dims4);

    mask<2> splmsk2;
    splmsk2[0] = true; splmsk2[1] = true;
    bis2.split(splmsk2, 3);
    bis2.split(splmsk2, 6);
    bis2.split(splmsk2, 9);
    bis2.split(splmsk2, 12);

    mask<4> splmsk3;
    splmsk3[0] = true; splmsk3[1] = true; splmsk3[2] = true;
    splmsk3[3] = true;
    bis4.split(splmsk3, 3);
    bis4.split(splmsk3, 6);
    bis4.split(splmsk3, 9);
    bis4.split(splmsk3, 12);

    block_tensor<4, double, allocator_t> bta(bis4);
    block_tensor<2, double, allocator_t> btb(bis2);

    //  Add perm symmetry

    block_tensor_ctrl<4, double> cbta(bta);
    scalar_transf<double> tr0, tr1(-1.);
    cbta.req_symmetry().insert(se_perm<4, double>(
        permutation<4>().permute(2, 3), tr1));
    cbta.req_symmetry().insert(se_perm<4, double>(
        permutation<4>().permute(0, 2).permute(1, 3), tr0));

    dense_tensor<4, double, allocator_t> ta(dims4);
    dense_tensor<2, double, allocator_t> tb(dims2), tb_ref(dims2);

    mask<4> msk;
    msk[0] = false; msk[1] = true; msk[2] = false; msk[3] = true;

    index<4> idx;
    idx[0] = 5; idx[1] = 0; idx[2] = 11; idx[3] = 0;
    index<4> idxbl;
    idxbl[0] = 1; idxbl[1] = 0; idxbl[2] = 3; idxbl[3] = 0;
    index<4> idxibl;
    idxibl[0] = idx[0] - 3; idxibl[1] = idx[1]; idxibl[2] = idx[2] - 9;
    idxibl[3] = idx[3];

    //  Fill in random data

    btod_random<4>().perform(bta);
    btod_random<2>().perform(btb);
    bta.set_immutable();

    //  Prepare the reference

    tod_btconv<4>(bta).perform(ta);
    tod_btconv<2>(btb).perform(tb_ref);
    tod_extract<4, 2>(ta, msk, idx).perform(tb_ref, -2.3);

    //  Invoke the operation

    btod_extract<4, 2>(bta, msk, idxbl, idxibl).perform(btb, -2.3);
    tod_btconv<2>(btb).perform(tb);

    //  Compare against the reference

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Extract a matrix from a fourth-order tensor with splitting
        and perm symmetry
          \f$ b_{ak} = a_{ijka} \qquad a_{ijka} = -a_{jika} \f$
 **/
void btod_extract_test::test_13a() throw(libtest::test_exception) {

    static const char *testname = "btod_extract_test::test_13a()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i2a, i2b;
    i2b[0] = 11; i2b[1] = 7;
    index<4> i4a, i4b;
    i4b[0] = 7; i4b[1] = 7; i4b[2] = 7; i4b[3] = 11;
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    dimensions<4> dims4(index_range<4>(i4a, i4b));
    block_index_space<2> bis2(dims2);
    block_index_space<4> bis4(dims4);

    mask<2> m2a, m2b;
    m2a[0] = true; m2b[1] = true;
    bis2.split(m2a, 3);
    bis2.split(m2a, 6);
    bis2.split(m2a, 9);
    bis2.split(m2b, 4);

    mask<4> m4a, m4b;
    m4b[0] = true; m4b[1] = true; m4b[2] = true; m4a[3] = true;
    bis4.split(m4a, 3);
    bis4.split(m4a, 6);
    bis4.split(m4a, 9);
    bis4.split(m4b, 4);

    block_tensor<4, double, allocator_t> bta(bis4);
    block_tensor<2, double, allocator_t> btb(bis2);

    //  Add perm symmetry

    block_tensor_ctrl<4, double> cbta(bta);
    scalar_transf<double> tr0, tr1(-1.);
    cbta.req_symmetry().insert(se_perm<4, double>(
        permutation<4>().permute(0, 1), tr1));

    dense_tensor<4, double, allocator_t> ta(dims4);
    dense_tensor<2, double, allocator_t> tb(dims2), tb_ref(dims2);

    mask<4> msk;
    msk[0] = false; msk[1] = true; msk[2] = false; msk[3] = true;
    permutation<2> perm;
    perm.permute(0, 1);

    index<4> idx;
    idx[0] = 4; idx[1] = 0; idx[2] = 3; idx[3] = 0;
    index<4> idxbl;
    idxbl[0] = 1; idxbl[1] = 0; idxbl[2] = 0; idxbl[3] = 0;
    index<4> idxibl;
    idxibl[0] = idx[0] - 4; idxibl[1] = idx[1]; idxibl[2] = idx[2];
    idxibl[3] = idx[3];

    //  Fill in random data

    btod_random<4>().perform(bta);
    bta.set_immutable();

    //  Prepare the reference

    tod_btconv<4>(bta).perform(ta);
    tod_extract<4, 2>(ta, msk, perm, idx).perform(tb_ref);

    //  Invoke the operation

    btod_extract<4, 2>(bta, msk, perm, idxbl, idxibl).perform(btb);
    tod_btconv<2>(btb).perform(tb);

    //  Compare against the reference

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
