#include <libtensor/core/allocator.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/core/direct_block_tensor.h>
#include <libtensor/dense_tensor/tod_btconv.h>
#include <libtensor/dense_tensor/tod_contract2.h>
#include <libtensor/dense_tensor/tod_dirsum.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/btod/btod_add.h>
#include <libtensor/btod/btod_copy.h>
#include <libtensor/btod/btod_contract2.h>
#include <libtensor/btod/btod_dirsum.h>
#include <libtensor/btod/btod_random.h>
#include "../compare_ref.h"
#include "direct_block_tensor_test.h"

namespace libtensor {


void direct_block_tensor_test::perform() throw(libtest::test_exception) {

    test_op_1();
    test_op_2();
    test_op_3();
    test_op_4();
}


/** \test Installs a simple copy operation in a direct block %tensor.
 **/
void direct_block_tensor_test::test_op_1() throw(libtest::test_exception) {

    static const char *testname = "direct_block_tensor_test::test_op_1()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);

    block_tensor<2, double, allocator_t> bta(bis);
    btod_random<2>().perform(bta);
    bta.set_immutable();

    btod_copy<2> op_copy(bta);
    direct_block_tensor<2, double, allocator_t> btb(op_copy);

    dense_tensor<2, double, allocator_t> tc(dims), tc_ref(dims);
    tod_btconv<2>(bta).perform(tc_ref);
    tod_btconv<2>(btb).perform(tc);
    compare_ref<2>::compare(testname, tc, tc_ref, 0.0);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Installs a simple copy operation in a direct block %tensor
        (multiple blocks).
 **/
void direct_block_tensor_test::test_op_2() throw(libtest::test_exception) {

    static const char *testname = "direct_block_tensor_test::test_op_2()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> msk;
    msk[0]=true; msk[1]=true;
    bis.split(msk,4);

    block_tensor<2, double, allocator_t> bta(bis);
    btod_random<2>().perform(bta);
    block_tensor_ctrl<2,double> ctrl(bta);
    i1[1]=1;
    i2[0]=1; i2[1]=0;
    ctrl.req_zero_block(i1);
    ctrl.req_zero_block(i2);
    bta.set_immutable();

    btod_copy<2> op_copy(bta);
    direct_block_tensor<2, double, allocator_t> btb(op_copy);

    dense_tensor<2, double, allocator_t> tc(dims), tc_ref(dims);
    tod_btconv<2>(bta).perform(tc_ref);
    tod_btconv<2>(btb).perform(tc);
    compare_ref<2>::compare(testname, tc, tc_ref, 0.0);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Installs a copy operation in a direct block %tensor and runs
        a contraction.
 **/
void direct_block_tensor_test::test_op_3() throw(libtest::test_exception) {

    static const char *testname = "direct_block_tensor_test::test_op_3()";

    typedef std_allocator<double> allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> m;
    m[0] = true; m[1] = true; m[2] = true; m[3] = true;
    bis.split(m, 2);
    bis.split(m, 4);
    bis.split(m, 6);
    bis.split(m, 8);

    block_tensor<4, double, allocator_t> bta1(bis), bta2(bis), bta3(bis),
        bta4(bis), bta(bis), btb(bis), btc(bis);
    btod_random<4>().perform(bta1);
    btod_random<4>().perform(bta2);
    btod_random<4>().perform(bta3);
    btod_random<4>().perform(bta4);
    btod_random<4>().perform(btb);
    bta1.set_immutable();
    bta2.set_immutable();
    bta3.set_immutable();
    bta4.set_immutable();
    btb.set_immutable();

    btod_add<4> op(bta1, 2.0);
    op.add_op(bta2, -2.0);
    op.add_op(bta3, permutation<4>().permute(1, 3), 2.0);
    op.add_op(bta4, permutation<4>().permute(0, 2), -2.0);
    op.perform(bta);
    bta.set_immutable();

    direct_block_tensor<4, double, allocator_t> dbta(op);

    contraction2<2, 2, 2> contr;
    contr.contract(1, 0);
    contr.contract(3, 2);

    btod_contract2<2, 2, 2>(contr, dbta, dbta).perform(btc);

    dense_tensor<4, double, allocator_t> ta(dims), tb(dims), tc(dims),
        tc_ref(dims);
    tod_btconv<4>(bta).perform(ta);
    tod_btconv<4>(btb).perform(tb);
    tod_btconv<4>(btc).perform(tc);
    tod_contract2<2, 2, 2>(contr, ta, ta).perform(true, 1.0, tc_ref);

    compare_ref<4>::compare(testname, tc, tc_ref, 2e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Nested use of direct block tensors
 **/
void direct_block_tensor_test::test_op_4() throw(libtest::test_exception) {

    static const char *testname = "direct_block_tensor_test::test_op_4()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i2a, i2b;
    i2b[0] = 9; i2b[1] = 9;
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    block_index_space<2> bis2(dims2);
    mask<2> m2; m2[0] = true; m2[1] = true;
    bis2.split(m2, 2);
    bis2.split(m2, 4);
    bis2.split(m2, 6);
    bis2.split(m2, 8);

    index<4> i4a, i4b;
    i4b[0] = 9; i4b[1] = 9; i4b[2] = 9; i4b[3] = 9;
    dimensions<4> dims4(index_range<4>(i4a, i4b));
    block_index_space<4> bis4(dims4);
    mask<4> m4; m4[0] = true; m4[1] = true; m4[2] = true; m4[3] = true;
    bis4.split(m4, 2);
    bis4.split(m4, 4);
    bis4.split(m4, 6);
    bis4.split(m4, 8);

    block_tensor<2, double, allocator_t> bta1(bis2), bta2(bis2), bta3(bis2),
        bta4(bis2);
    btod_random<2>().perform(bta1);
    btod_random<2>().perform(bta2);
    btod_random<2>().perform(bta3);
    btod_random<2>().perform(bta4);
    bta1.set_immutable();
    bta2.set_immutable();
    bta3.set_immutable();
    bta4.set_immutable();

    btod_add<2> add1(bta1, 2.0); add1.add_op(bta2, -2.0);
    btod_add<2> add2(bta3, -3.0); add2.add_op(bta4, 2.5);
    direct_block_tensor<2, double, allocator_t> dbta1(add1), dbta2(add2);

    btod_dirsum<2, 2> dirsum1(dbta1, 1.0, dbta2, -2.0);
    btod_dirsum<2, 2> dirsum2(dbta1, -2.0, dbta2, 1.0);
    direct_block_tensor<4, double, allocator_t> dbtb1(dirsum1),
        dbtb2(dirsum2);

    contraction2<2, 2, 2> contr;
    contr.contract(1, 0);
    contr.contract(3, 2);

    block_tensor<4, double, allocator_t> btc(bis4);
    btod_contract2<2, 2, 2>(contr, dbtb1, dbtb2).perform(btc);

    dense_tensor<2, double, allocator_t> ta1(dims2), ta2(dims2), ta3(dims2),
        ta4(dims2), ta5(dims2), ta6(dims2);
    tod_btconv<2>(bta1).perform(ta1);
    tod_btconv<2>(bta2).perform(ta2);
    tod_btconv<2>(bta3).perform(ta3);
    tod_btconv<2>(bta4).perform(ta4);
    tod_copy<2>(ta1, 2.0).perform(true, 1.0, ta5);
    tod_copy<2>(ta2, -2.0).perform(false, 1.0, ta5);
    tod_copy<2>(ta3, -3.0).perform(true, 1.0, ta6);
    tod_copy<2>(ta4, 2.5).perform(false, 1.0, ta6);

	dense_tensor<4, double, allocator_t> tb1(dims4), tb2(dims4), tc(dims4),
		tc_ref(dims4);
	tod_dirsum<2, 2>(ta5, 1.0, ta6, -2.0).perform(true, 1.0, tb1);
	tod_dirsum<2, 2>(ta5, -2.0, ta6, 1.0).perform(true, 1.0, tb2);
	tod_contract2<2, 2, 2>(contr, tb1, tb2).perform(true, 1.0, tc_ref);
	tod_btconv<4>(btc).perform(tc);

    compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

