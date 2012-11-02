#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/btod_contract2.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/symmetry/permutation_group.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/se_label.h>
#include <libtensor/symmetry/se_part.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/dense_tensor/tod_btconv.h>
#include <libtensor/dense_tensor/tod_contract2.h>
#include <libtensor/dense_tensor/tod_set.h>
#include "../compare_ref.h"
#include "btod_contract2_test.h"

namespace libtensor {

void btod_contract2_test::perform() throw(libtest::test_exception) {

    allocator<double>::vmm().init(16, 16, 16777216, 16777216);

    try {

    test_bis_1();
    test_bis_2();
    test_bis_3();
    test_bis_4();
    test_bis_5();

std::cout << __FILE__ << ": " << __LINE__ << std::endl;
    //  Tests for zero block structure
    test_zeroblk_1();
    test_zeroblk_2();
    test_zeroblk_3();
    test_zeroblk_4();
    test_zeroblk_5();
    test_zeroblk_6();

    //  Tests for contractions

    std::cout << __FILE__ << ": " << __LINE__ << std::endl;
    test_contr_1();
    test_contr_2();
    test_contr_3();
    test_contr_4();
    test_contr_5();
    test_contr_6();
    test_contr_7();
    test_contr_8();
    test_contr_9();
    std::cout << __FILE__ << ": " << __LINE__ << std::endl;
    test_contr_10();
    test_contr_11();
    test_contr_12();
    test_contr_13();
    test_contr_14(0.0);
    test_contr_14(1.0);
    test_contr_14(-2.2);
    test_contr_15(0.0);
    test_contr_15(1.0);
    test_contr_15(-2.2);
    test_contr_16(0.0);
    test_contr_16(1.0);
    test_contr_16(-2.2);
    test_contr_17(0.0);
    test_contr_17(1.5);
    std::cout << __FILE__ << ": " << __LINE__ << std::endl;
    test_contr_18(0.0);
    test_contr_18(-1.5);
    test_contr_19();
    test_contr_20a();
    test_contr_20b();
    test_contr_21();
    test_contr_22();
    test_contr_23();
    std::cout << __FILE__ << ": " << __LINE__ << std::endl;

    //  Tests for the contraction of a block tensor with itself

    test_self_1();
    test_self_2();
    test_self_3();

    } catch(...) {
        allocator<double>::vmm().shutdown();
        throw;
    }

    allocator<double>::vmm().shutdown();
}


void btod_contract2_test::test_bis_1() throw(libtest::test_exception) {

    //
    //  c_ijkl = a_ijkp b_lp
    //  [ij] = 5  (no splits)
    //  [kl] = 10 (no splits)
    //  [p]  = 4  (no splits)
    //
    //  Expected block index space:
    //  [ijkl] have correct dimensions, no splits
    //

    static const char *testname = "btod_contract2_test::test_bis_1()";

    typedef std_allocator<double> allocator_t;

    try {

        index<4> ia1, ia2;
        ia2[0] = 4; ia2[1] = 4; ia2[2] = 9; ia2[3] = 3;
        dimensions<4> dimsa(index_range<4>(ia1, ia2));
        block_index_space<4> bisa(dimsa);

        index<2> ib1, ib2;
        ib2[0] = 9; ib2[1] = 3;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));
        block_index_space<2> bisb(dimsb);

        index<4> ic1, ic2;
        ic2[0] = 4; ic2[1] = 4; ic2[2] = 9; ic2[3] = 9;
        dimensions<4> dimsc(index_range<4>(ic1, ic2));
        block_index_space<4> bisc_ref(dimsc);

        block_tensor<4, double, allocator_t> bta(bisa);
        block_tensor<2, double, allocator_t> btb(bisb);
        contraction2<3, 1, 1> contr;
        contr.contract(3, 1);

        btod_contract2<3, 1, 1> op(contr, bta, btb);

        if(!op.get_bis().equals(bisc_ref)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Incorrect output block index space.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract2_test::test_bis_2() throw(libtest::test_exception) {

    //
    //  c_ijkl = a_ijkp b_lp
    //  [ij] = 5  (no splits)
    //  [kl] = 10 (4, 6)
    //  [p]  = 4  (no splits)
    //
    //  Expected block index space:
    //  [ijkl] have correct dimensions, [ij] have no splits,
    //  [kl] split identically
    //

    static const char *testname = "btod_contract2_test::test_bis_2()";

    typedef std_allocator<double> allocator_t;

    try {

        index<4> ia1, ia2;
        ia2[0] = 4; ia2[1] = 4; ia2[2] = 9; ia2[3] = 3;
        dimensions<4> dimsa(index_range<4>(ia1, ia2));
        block_index_space<4> bisa(dimsa);
        mask<4> ma1;
        ma1[2] = true;
        bisa.split(ma1, 4);

        index<2> ib1, ib2;
        ib2[0] = 9; ib2[1] = 3;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));
        block_index_space<2> bisb(dimsb);
        mask<2> mb1;
        mb1[0] = true;
        bisb.split(mb1, 4);

        index<4> ic1, ic2;
        ic2[0] = 4; ic2[1] = 4; ic2[2] = 9; ic2[3] = 9;
        dimensions<4> dimsc(index_range<4>(ic1, ic2));
        block_index_space<4> bisc_ref(dimsc);
        mask<4> mc1;
        mc1[2] = true; mc1[3] = true;
        bisc_ref.split(mc1, 4);

        block_tensor<4, double, allocator_t> bta(bisa);
        block_tensor<2, double, allocator_t> btb(bisb);
        contraction2<3, 1, 1> contr;
        contr.contract(3, 1);

        btod_contract2<3, 1, 1> op(contr, bta, btb);

        if(!op.get_bis().equals(bisc_ref)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Incorrect output block index space.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract2_test::test_bis_3() throw(libtest::test_exception) {

    //
    //  c_ijkl = a_ijkp b_lp
    //  [ij] = 5  (2, 3)
    //  [kl] = 10 (4, 6)
    //  [p]  = 4  (no splits)
    //
    //  Expected block index space:
    //  [ijkl] have correct dimensions,
    //  spaces in pairs [ij] and [kl] are split identically
    //

    static const char *testname = "btod_contract2_test::test_bis_3()";

    typedef std_allocator<double> allocator_t;

    try {

        index<4> ia1, ia2;
        ia2[0] = 4; ia2[1] = 4; ia2[2] = 9; ia2[3] = 3;
        dimensions<4> dimsa(index_range<4>(ia1, ia2));
        block_index_space<4> bisa(dimsa);
        mask<4> ma1, ma2;
        ma1[0] = true; ma1[1] = true;
        ma2[2] = true;
        bisa.split(ma1, 2);
        bisa.split(ma2, 4);

        index<2> ib1, ib2;
        ib2[0] = 9; ib2[1] = 3;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));
        block_index_space<2> bisb(dimsb);
        mask<2> mb1;
        mb1[0] = true;
        bisb.split(mb1, 4);

        index<4> ic1, ic2;
        ic2[0] = 4; ic2[1] = 4; ic2[2] = 9; ic2[3] = 9;
        dimensions<4> dimsc(index_range<4>(ic1, ic2));
        block_index_space<4> bisc_ref(dimsc);
        mask<4> mc1, mc2;
        mc1[0] = true; mc1[1] = true;
        mc2[2] = true; mc2[3] = true;
        bisc_ref.split(mc1, 2);
        bisc_ref.split(mc2, 4);

        block_tensor<4, double, allocator_t> bta(bisa);
        block_tensor<2, double, allocator_t> btb(bisb);
        contraction2<3, 1, 1> contr;
        contr.contract(3, 1);

        btod_contract2<3, 1, 1> op(contr, bta, btb);

        if(!op.get_bis().equals(bisc_ref)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Incorrect output block index space.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract2_test::test_bis_4() throw(libtest::test_exception) {

    //
    //  c_ijkl = a_ijpq b_klpq
    //  [ijklpq] = 11 (3, 2, 5)
    //
    //  Expected block index space:
    //  [ijkl] have correct dimensions, one splitting pattern
    //

    static const char *testname = "btod_contract2_test::test_bis_4()";

    typedef std_allocator<double> allocator_t;

    try {

        index<4> ia1, ia2;
        ia2[0] = 10; ia2[1] = 10; ia2[2] = 10; ia2[3] = 10;
        dimensions<4> dimsa(index_range<4>(ia1, ia2));
        block_index_space<4> bisa(dimsa);
        mask<4> ma1;
        ma1[0] = true; ma1[1] = true; ma1[2] = true; ma1[3] = true;
        bisa.split(ma1, 3);
        bisa.split(ma1, 5);

        block_index_space<4> bisb(bisa), bisc_ref(bisa);

        block_tensor<4, double, allocator_t> bta(bisa), btb(bisb);
        contraction2<2, 2, 2> contr;
        contr.contract(0, 2);
        contr.contract(1, 3);

        btod_contract2<2, 2, 2> op(contr, bta, btb);

        if(!op.get_bis().equals(bisc_ref)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Incorrect output block index space.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract2_test::test_bis_5() throw(libtest::test_exception) {

    //
    //  c_ijk = a_ipqr b_jpqrk
    //  [ijpqr] = 11 (3, 2, 5)
    //  [k]     = 9  (4, 5)
    //
    //  Expected block index space:
    //  [ijk] have correct dimensions,
    //  [ij] and [k] preserve the splitting pattern
    //

    static const char *testname = "btod_contract2_test::test_bis_5()";

    typedef std_allocator<double> allocator_t;

    try {

        index<4> ia1, ia2;
        ia2[0] = 10; ia2[1] = 10; ia2[2] = 10; ia2[3] = 10;
        dimensions<4> dimsa(index_range<4>(ia1, ia2));
        block_index_space<4> bisa(dimsa);
        mask<4> ma1;
        ma1[0] = true; ma1[1] = true; ma1[2] = true; ma1[3] = true;
        bisa.split(ma1, 3);
        bisa.split(ma1, 5);

        index<5> ib1, ib2;
        ib2[0] = 10; ib2[1] = 10; ib2[2] = 10; ib2[3] = 10; ib2[4] = 8;
        dimensions<5> dimsb(index_range<5>(ib1, ib2));
        block_index_space<5> bisb(dimsb);
        mask<5> mb1, mb2;
        mb1[0] = true; mb1[1] = true; mb1[2] = true; mb1[3] = true;
        mb2[4] = true;
        bisb.split(mb1, 3);
        bisb.split(mb1, 5);
        bisb.split(mb2, 4);

        index<3> ic1, ic2;
        ic2[0] = 10; ic2[1] = 10; ic2[2] = 8;
        dimensions<3> dimsc(index_range<3>(ic1, ic2));
        block_index_space<3> bisc_ref(dimsc);
        mask<3> mc1, mc2;
        mc1[0] = true; mc1[1] = true;
        mc2[2] = true;
        bisc_ref.split(mc1, 3);
        bisc_ref.split(mc1, 5);
        bisc_ref.split(mc2, 4);

        block_tensor<4, double, allocator_t> bta(bisa);
        block_tensor<5, double, allocator_t> btb(bisb);
        contraction2<1, 2, 3> contr;
        contr.contract(1, 1);
        contr.contract(2, 2);
        contr.contract(3, 3);

        btod_contract2<1, 2, 3> op(contr, bta, btb);

        if(!op.get_bis().equals(bisc_ref)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Invalid output block index space.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Runs \f$ c_{ij} = \sum_p a_{ip} b_{jp} \f$.
Dimensions: [ijp] = 10. No splitting points. No symmetry.

The single block of a is zero, b is non-zero. Initially, c is zero.
The result c is expected to have a single zero block.
 **/
void btod_contract2_test::test_zeroblk_1() throw(libtest::test_exception) {

    static const char *testname = "btod_contract2_test::test_zeroblk_1()";

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i21, i22;
        i22[0] = 9; i22[1] = 9;
        dimensions<2> dimsa(index_range<2>(i21, i22));
        dimensions<2> dimsb(index_range<2>(i21, i22));
        dimensions<2> dimsc(index_range<2>(i21, i22));
        block_index_space<2> bisa(dimsa), bisb(dimsb);
        block_index_space<2> bisc(dimsc);

        block_tensor<2, double, allocator_t> bta(bisa);
        block_tensor<2, double, allocator_t> btb(bisb);
        block_tensor<2, double, allocator_t> btc(bisc);

        //  Load random data for input

        index<2> i_00;
        btod_random<2>().perform(btb, i_00);
        bta.set_immutable();
        btb.set_immutable();

        //  Run contraction and compute the reference

        contraction2<1, 1, 1> contr;
        contr.contract(1, 1);
        btod_contract2<1, 1, 1>(contr, bta, btb).perform(btc);

        //  Check the zero block structure

        block_tensor_ctrl<2, double> btcc(btc);
        if(!btcc.req_is_zero_block(i_00)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Block [0,0] is expected to be zero.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Runs \f$ c_{ij} = \sum_p a_{ip} b_{jp} \f$.
Dimensions: [ijp] = 10. No splitting points. No symmetry.

The single block of a is zero, b is non-zero. Initially, c is non-zero.
The result c is expected to have a single zero block.
 **/
void btod_contract2_test::test_zeroblk_2() throw(libtest::test_exception) {

    static const char *testname = "btod_contract2_test::test_zeroblk_2()";

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i21, i22;
        i22[0] = 9; i22[1] = 9;
        dimensions<2> dimsa(index_range<2>(i21, i22));
        dimensions<2> dimsb(index_range<2>(i21, i22));
        dimensions<2> dimsc(index_range<2>(i21, i22));
        block_index_space<2> bisa(dimsa), bisb(dimsb);
        block_index_space<2> bisc(dimsc);

        block_tensor<2, double, allocator_t> bta(bisa);
        block_tensor<2, double, allocator_t> btb(bisb);
        block_tensor<2, double, allocator_t> btc(bisc);

        //  Load random data for input

        index<2> i_00;
        btod_random<2>().perform(btb, i_00);
        btod_random<2>().perform(btc);
        bta.set_immutable();
        btb.set_immutable();

        //  Run contraction and compute the reference

        contraction2<1, 1, 1> contr;
        contr.contract(1, 1);
        btod_contract2<1, 1, 1>(contr, bta, btb).perform(btc);

        //  Check the zero block structure

        block_tensor_ctrl<2, double> btcc(btc);
        if(!btcc.req_is_zero_block(i_00)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Block [0,0] is expected to be zero.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Runs \f$ c_{ij} = \sum_p a_{ip} b_{jp} \f$.
Dimensions: [ijp] = 10. No splitting points. No symmetry.

The single block of a is non-zero, b is zero. Initially, c is zero.
The result c is expected to have a single zero block.
 **/
void btod_contract2_test::test_zeroblk_3() throw(libtest::test_exception) {

    static const char *testname = "btod_contract2_test::test_zeroblk_3()";

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i21, i22;
        i22[0] = 9; i22[1] = 9;
        dimensions<2> dimsa(index_range<2>(i21, i22));
        dimensions<2> dimsb(index_range<2>(i21, i22));
        dimensions<2> dimsc(index_range<2>(i21, i22));
        block_index_space<2> bisa(dimsa), bisb(dimsb);
        block_index_space<2> bisc(dimsc);

        block_tensor<2, double, allocator_t> bta(bisa);
        block_tensor<2, double, allocator_t> btb(bisb);
        block_tensor<2, double, allocator_t> btc(bisc);

        //  Load random data for input

        index<2> i_00;
        btod_random<2>().perform(bta, i_00);
        bta.set_immutable();
        btb.set_immutable();

        //  Run contraction and compute the reference

        contraction2<1, 1, 1> contr;
        contr.contract(1, 1);
        btod_contract2<1, 1, 1>(contr, bta, btb).perform(btc);

        //  Check the zero block structure

        block_tensor_ctrl<2, double> btcc(btc);
        if(!btcc.req_is_zero_block(i_00)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Block [0,0] is expected to be zero.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Runs \f$ c_{ij} = \sum_p a_{ip} b_{jp} \f$.
Dimensions: [ijp] = 10. No splitting points. No symmetry.

The single block of a is non-zero, b is zero. Initially, c is non-zero.
The result c is expected to have a single zero block.
 **/
void btod_contract2_test::test_zeroblk_4() throw(libtest::test_exception) {

    static const char *testname = "btod_contract2_test::test_zeroblk_4()";

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i21, i22;
        i22[0] = 9; i22[1] = 9;
        dimensions<2> dimsa(index_range<2>(i21, i22));
        dimensions<2> dimsb(index_range<2>(i21, i22));
        dimensions<2> dimsc(index_range<2>(i21, i22));
        block_index_space<2> bisa(dimsa), bisb(dimsb);
        block_index_space<2> bisc(dimsc);

        block_tensor<2, double, allocator_t> bta(bisa);
        block_tensor<2, double, allocator_t> btb(bisb);
        block_tensor<2, double, allocator_t> btc(bisc);

        //  Load random data for input

        index<2> i_00;
        btod_random<2>().perform(bta, i_00);
        btod_random<2>().perform(btc);
        bta.set_immutable();
        btb.set_immutable();

        //  Run contraction and compute the reference

        contraction2<1, 1, 1> contr;
        contr.contract(1, 1);
        btod_contract2<1, 1, 1>(contr, bta, btb).perform(btc);

        //  Check the zero block structure

        block_tensor_ctrl<2, double> btcc(btc);
        if(!btcc.req_is_zero_block(i_00)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Block [0,0] is expected to be zero.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Runs \f$ c_{ij} = \sum_p a_{ip} b_{jp} \f$.
Dimensions: [ijp] = 10. Splitting points: [ijp] = {5}. No symmetry.

Only diagonal blocks in a and b are non-zero. Initially, c is zero.
The result c is expected to have diagonal blocks non-zero and
off-diagonal blocks zero.
 **/
void btod_contract2_test::test_zeroblk_5() throw(libtest::test_exception) {

    static const char *testname = "btod_contract2_test::test_zeroblk_5()";

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i21, i22;
        i22[0] = 9; i22[1] = 9;
        dimensions<2> dimsa(index_range<2>(i21, i22));
        dimensions<2> dimsb(index_range<2>(i21, i22));
        dimensions<2> dimsc(index_range<2>(i21, i22));
        block_index_space<2> bisa(dimsa);
        mask<2> m2; m2[0] = true; m2[1] = true;
        bisa.split(m2, 5);
        block_index_space<2> bisb(bisa), bisc(bisa);

        block_tensor<2, double, allocator_t> bta(bisa);
        block_tensor<2, double, allocator_t> btb(bisb);
        block_tensor<2, double, allocator_t> btc(bisc);

        //  Load random data for input

        index<2> i_00, i_01, i_10, i_11;
        i_01[1] = 1; i_10[0] = 1;
        i_11[0] = 1; i_11[1] = 1;
        btod_random<2>().perform(bta, i_00);
        btod_random<2>().perform(bta, i_11);
        btod_random<2>().perform(btb, i_00);
        btod_random<2>().perform(btb, i_11);
        bta.set_immutable();
        btb.set_immutable();

        //  Run contraction and compute the reference

        contraction2<1, 1, 1> contr;
        contr.contract(1, 1);
        btod_contract2<1, 1, 1>(contr, bta, btb).perform(btc);

        //  Check the zero block structure

        block_tensor_ctrl<2, double> btcc(btc);
        if(btcc.req_is_zero_block(i_00)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Block [0,0] is expected to be non-zero.");
        }
        if(!btcc.req_is_zero_block(i_01)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Block [0,1] is expected to be zero.");
        }
        if(!btcc.req_is_zero_block(i_10)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Block [1,0] is expected to be zero.");
        }
        if(btcc.req_is_zero_block(i_11)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Block [1,1] is expected to be non-zero.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Runs \f$ c_{ij} = \sum_p a_{ip} b_{jp} \f$.
Dimensions: [ijp] = 10. Splitting points: [ijp] = {5}. No symmetry.

Only diagonal blocks in a and b are non-zero. Initially, c is non-zero.
The result c is expected to have diagonal blocks non-zero and
off-diagonal blocks zero.
 **/
void btod_contract2_test::test_zeroblk_6() throw(libtest::test_exception) {

    static const char *testname = "btod_contract2_test::test_zeroblk_6()";

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i21, i22;
        i22[0] = 9; i22[1] = 9;
        dimensions<2> dimsa(index_range<2>(i21, i22));
        dimensions<2> dimsb(index_range<2>(i21, i22));
        dimensions<2> dimsc(index_range<2>(i21, i22));
        block_index_space<2> bisa(dimsa);
        mask<2> m2; m2[0] = true; m2[1] = true;
        bisa.split(m2, 5);
        block_index_space<2> bisb(bisa), bisc(bisa);

        block_tensor<2, double, allocator_t> bta(bisa);
        block_tensor<2, double, allocator_t> btb(bisb);
        block_tensor<2, double, allocator_t> btc(bisc);

        //  Load random data for input

        index<2> i_00, i_01, i_10, i_11;
        i_01[1] = 1; i_10[0] = 1;
        i_11[0] = 1; i_11[1] = 1;
        btod_random<2>().perform(bta, i_00);
        btod_random<2>().perform(bta, i_11);
        btod_random<2>().perform(btb, i_00);
        btod_random<2>().perform(btb, i_11);
        btod_random<2>().perform(btc);
        bta.set_immutable();
        btb.set_immutable();

        //  Run contraction and compute the reference

        contraction2<1, 1, 1> contr;
        contr.contract(1, 1);
        btod_contract2<1, 1, 1>(contr, bta, btb).perform(btc);

        //  Check the zero block structure

        block_tensor_ctrl<2, double> btcc(btc);
        if(btcc.req_is_zero_block(i_00)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Block [0,0] is expected to be non-zero.");
        }
        if(!btcc.req_is_zero_block(i_01)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Block [0,1] is expected to be zero.");
        }
        if(!btcc.req_is_zero_block(i_10)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Block [1,0] is expected to be zero.");
        }
        if(btcc.req_is_zero_block(i_11)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Block [1,1] is expected to be non-zero.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract2_test::test_contr_1() throw(libtest::test_exception) {

    //
    //  c_ijkl = a_ijpq b_klpq
    //  All dimensions are identical, no symmetry
    //

    static const char *testname = "btod_contract2_test::test_contr_1()";

    typedef std_allocator<double> allocator_t;

    try {

        index<4> i1, i2;
        i2[0] = 10; i2[1] = 10; i2[2] = 10; i2[3] = 10;
        dimensions<4> dims(index_range<4>(i1, i2));
        block_index_space<4> bisa(dims);
        mask<4> m1;
        m1[0] = true; m1[1] = true; m1[2] = true; m1[3] = true;
        bisa.split(m1, 3);
        bisa.split(m1, 5);

        block_index_space<4> bisb(bisa), bisc(bisa);

        block_tensor<4, double, allocator_t> bta(bisa), btb(bisb), btc(bisc);

        //  Load random data for input

        btod_random<4> rand;
        rand.perform(bta);
        rand.perform(btb);
        bta.set_immutable();
        btb.set_immutable();

        //  Run contraction

        contraction2<2, 2, 2> contr;
        contr.contract(2, 2);
        contr.contract(3, 3);

        btod_contract2<2, 2, 2> op(contr, bta, btb);
        op.perform(btc);

        //  Convert block tensors to regular tensors

        dense_tensor<4, double, allocator_t> ta(dims), tb(dims), tc(dims),
                tc_ref(dims);
        tod_btconv<4> conva(bta);
        conva.perform(ta);
        tod_btconv<4> convb(btb);
        convb.perform(tb);
        tod_btconv<4> convc(btc);
        convc.perform(tc);

        //  Compute reference tensor

        tod_contract2<2, 2, 2> op_ref(contr, ta, tb);
        op_ref.perform(true, tc_ref);

        //  Compare against reference

        compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract2_test::test_contr_2() throw(libtest::test_exception) {

    //
    //  c_ikjl = a_ijpq b_klqp
    //  All dimensions are identical, no symmetry
    //

    static const char *testname = "btod_contract2_test::test_contr_2()";

    typedef std_allocator<double> allocator_t;

    try {

        index<4> i1, i2;
        i2[0] = 10; i2[1] = 10; i2[2] = 10; i2[3] = 10;
        dimensions<4> dims(index_range<4>(i1, i2));
        block_index_space<4> bisa(dims);
        mask<4> m1;
        m1[0] = true; m1[1] = true; m1[2] = true; m1[3] = true;
        bisa.split(m1, 3);
        bisa.split(m1, 5);

        block_index_space<4> bisb(bisa), bisc(bisa);

        block_tensor<4, double, allocator_t> bta(bisa), btb(bisb), btc(bisc);

        //  Load random data for input

        btod_random<4> rand;
        rand.perform(bta);
        rand.perform(btb);
        bta.set_immutable();
        btb.set_immutable();

        //  Run contraction

        permutation<4> permc; permc.permute(1, 2);
        contraction2<2, 2, 2> contr(permc);
        contr.contract(2, 3);
        contr.contract(3, 2);

        btod_contract2<2, 2, 2> op(contr, bta, btb);
        op.perform(btc);

        //  Convert block tensors to regular tensors

        dense_tensor<4, double, allocator_t> ta(dims), tb(dims), tc(dims),
                tc_ref(dims);
        tod_btconv<4> conva(bta);
        conva.perform(ta);
        tod_btconv<4> convb(btb);
        convb.perform(tb);
        tod_btconv<4> convc(btc);
        convc.perform(tc);

        //  Compute reference tensor

        tod_contract2<2, 2, 2> op_ref(contr, ta, tb);
        op_ref.perform(true, tc_ref);

        //  Compare against reference

        compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract2_test::test_contr_3() throw(libtest::test_exception) {

    //
    //  c_ijkl = a_ijpq b_pqkl
    //  Dimensions [ij]=10, [kl]=12, [pq]=6, no symmetry
    //

    static const char *testname = "btod_contract2_test::test_contr_3()";

    typedef std_allocator<double> allocator_t;

    try {

        index<4> i1, i2;
        i2[0] = 9; i2[1] = 9; i2[2] = 5; i2[3] = 5;
        dimensions<4> dimsa(index_range<4>(i1, i2));
        i2[0] = 5; i2[1] = 5; i2[2] = 11; i2[3] = 11;
        dimensions<4> dimsb(index_range<4>(i1, i2));
        i2[0] = 9; i2[1] = 9; i2[2] = 11; i2[3] = 11;
        dimensions<4> dimsc(index_range<4>(i1, i2));
        block_index_space<4> bisa(dimsa), bisb(dimsb), bisc(dimsc);

        mask<4> msk1, msk2;
        msk1[0] = true; msk1[1] = true;
        msk2[2] = true; msk2[3] = true;

        bisa.split(msk1, 3);
        bisa.split(msk1, 5);
        bisa.split(msk2, 4);

        bisb.split(msk1, 4);
        bisb.split(msk2, 6);

        bisc.split(msk1, 3);
        bisc.split(msk1, 5);
        bisc.split(msk2, 6);

        block_tensor<4, double, allocator_t> bta(bisa), btb(bisb), btc(bisc);

        //  Load random data for input

        btod_random<4> rand;
        rand.perform(bta);
        rand.perform(btb);
        bta.set_immutable();
        btb.set_immutable();

        //  Run contraction

        contraction2<2, 2, 2> contr;
        contr.contract(2, 0);
        contr.contract(3, 1);

        btod_contract2<2, 2, 2> op(contr, bta, btb);
        op.perform(btc);

        //  Convert block tensors to regular tensors

        dense_tensor<4, double, allocator_t> ta(dimsa), tb(dimsb), tc(dimsc),
                tc_ref(dimsc);
        tod_btconv<4> conva(bta);
        conva.perform(ta);
        tod_btconv<4> convb(btb);
        convb.perform(tb);
        tod_btconv<4> convc(btc);
        convc.perform(tc);

        //  Compute reference tensor

        tod_contract2<2, 2, 2> op_ref(contr, ta, tb);
        op_ref.perform(true, tc_ref);

        //  Compare against reference

        compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract2_test::test_contr_4() throw(libtest::test_exception) {

    //
    //  c_ijkl = a_ijpq b_pqkl
    //  Dimensions [ij]=10, [kl]=12, [pq]=6, permutational symmetry
    //

    static const char *testname = "btod_contract2_test::test_contr_4()";

    typedef std_allocator<double> allocator_t;

    try {

        index<4> i1, i2;
        i2[0] = 9; i2[1] = 9; i2[2] = 5; i2[3] = 5;
        dimensions<4> dimsa(index_range<4>(i1, i2));
        i2[0] = 5; i2[1] = 5; i2[2] = 11; i2[3] = 11;
        dimensions<4> dimsb(index_range<4>(i1, i2));
        i2[0] = 9; i2[1] = 9; i2[2] = 11; i2[3] = 11;
        dimensions<4> dimsc(index_range<4>(i1, i2));
        block_index_space<4> bisa(dimsa), bisb(dimsb), bisc(dimsc);

        mask<4> msk1, msk2;
        msk1[0] = true; msk1[1] = true;
        msk2[2] = true; msk2[3] = true;

        bisa.split(msk1, 3);
        bisa.split(msk1, 5);
        bisa.split(msk2, 4);

        bisb.split(msk1, 4);
        bisb.split(msk2, 6);

        bisc.split(msk1, 3);
        bisc.split(msk1, 5);
        bisc.split(msk2, 6);

        block_tensor<4, double, allocator_t> bta(bisa), btb(bisb), btc(bisc);

        //  Set up symmetry

        permutation<4> p1023, p0132;
        p1023.permute(0, 1);
        p0132.permute(2, 3);
        scalar_transf<double> tr0, tr1(-1.);
        se_perm<4, double> cycle1(p1023, tr0), cycle2(p0132, tr0);
        block_tensor_ctrl<4, double> ctrla(bta), ctrlb(btb);
        ctrla.req_symmetry().insert(cycle1);
        ctrla.req_symmetry().insert(cycle2);
        ctrlb.req_symmetry().insert(cycle1);
        ctrlb.req_symmetry().insert(cycle2);

        //  Load random data for input

        btod_random<4> rand;
        rand.perform(bta);
        rand.perform(btb);
        bta.set_immutable();
        btb.set_immutable();

        //  Run contraction

        contraction2<2, 2, 2> contr;
        contr.contract(2, 0);
        contr.contract(3, 1);

        btod_contract2<2, 2, 2> op(contr, bta, btb);
        op.perform(btc);

        //  Convert block tensors to regular tensors

        dense_tensor<4, double, allocator_t> ta(dimsa), tb(dimsb), tc(dimsc),
                tc_ref(dimsc);
        tod_btconv<4> conva(bta);
        conva.perform(ta);
        tod_btconv<4> convb(btb);
        convb.perform(tb);
        tod_btconv<4> convc(btc);
        convc.perform(tc);

        //  Compute reference tensor

        tod_contract2<2, 2, 2> op_ref(contr, ta, tb);
        op_ref.perform(true, tc_ref);

        //  Compare against reference

        compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract2_test::test_contr_5() throw(libtest::test_exception) {

    //
    //  c_ijkl = c_ijkl + a_ijpq b_pqkl
    //  Dimensions [ij]=10, [kl]=12, [pq]=6, permutational symmetry
    //  Sym(C) = Sym(A*B)
    //

    static const char *testname = "btod_contract2_test::test_contr_5()";

    typedef std_allocator<double> allocator_t;

    try {

        index<4> i1, i2;
        i2[0] = 9; i2[1] = 9; i2[2] = 5; i2[3] = 5;
        dimensions<4> dimsa(index_range<4>(i1, i2));
        i2[0] = 5; i2[1] = 5; i2[2] = 11; i2[3] = 11;
        dimensions<4> dimsb(index_range<4>(i1, i2));
        i2[0] = 9; i2[1] = 9; i2[2] = 11; i2[3] = 11;
        dimensions<4> dimsc(index_range<4>(i1, i2));
        block_index_space<4> bisa(dimsa), bisb(dimsb), bisc(dimsc);

        mask<4> msk1, msk2;
        msk1[0] = true; msk1[1] = true;
        msk2[2] = true; msk2[3] = true;

        bisa.split(msk1, 3);
        bisa.split(msk1, 5);
        bisa.split(msk2, 4);

        bisb.split(msk1, 4);
        bisb.split(msk2, 6);

        bisc.split(msk1, 3);
        bisc.split(msk1, 5);
        bisc.split(msk2, 6);

        block_tensor<4, double, allocator_t> bta(bisa), btb(bisb), btc(bisc);

        //  Set up symmetry

        permutation<4> p1023, p0132;
        p1023.permute(0, 1);
        p0132.permute(2, 3);
        scalar_transf<double> tr0, tr1(-1.);
        se_perm<4, double> cycle1(p1023, tr0), cycle2(p0132, tr0);
        block_tensor_ctrl<4, double> ctrla(bta), ctrlb(btb), ctrlc(btc);
        ctrla.req_symmetry().insert(cycle1);
        ctrla.req_symmetry().insert(cycle2);
        ctrlb.req_symmetry().insert(cycle1);
        ctrlb.req_symmetry().insert(cycle2);
        ctrlc.req_symmetry().insert(cycle1);
        ctrlc.req_symmetry().insert(cycle2);

        //  Load random data for input

        btod_random<4> rand;
        rand.perform(bta);
        rand.perform(btb);
        rand.perform(btc);
        bta.set_immutable();
        btb.set_immutable();

        //  Convert input block tensors to regular tensors

        dense_tensor<4, double, allocator_t> ta(dimsa), tb(dimsb), tc(dimsc),
                tc_ref(dimsc);
        tod_btconv<4> conva(bta);
        conva.perform(ta);
        tod_btconv<4> convb(btb);
        convb.perform(tb);
        tod_btconv<4> convc_ref(btc);
        convc_ref.perform(tc_ref);

        //  Run contraction

        contraction2<2, 2, 2> contr;
        contr.contract(2, 0);
        contr.contract(3, 1);

        btod_contract2<2, 2, 2> op(contr, bta, btb);
        op.perform(btc, 2.0);

        tod_btconv<4> convc(btc);
        convc.perform(tc);

        //  Compute reference tensor

        tod_contract2<2, 2, 2> op_ref(contr, ta, tb, 2.0);
        op_ref.perform(false, tc_ref);

        //  Compare against reference

        compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract2_test::test_contr_6() throw(libtest::test_exception) {

    //
    //  c_ijkl = c_ijkl + a_ijpq b_pqkl
    //  Dimensions [ij]=10, [kl]=12, [pq]=6, permutational symmetry
    //  Sym(C) > Sym(A*B)
    //

    static const char *testname = "btod_contract2_test::test_contr_6()";

    typedef std_allocator<double> allocator_t;

    try {

        index<4> i1, i2;
        i2[0] = 9; i2[1] = 9; i2[2] = 5; i2[3] = 5;
        dimensions<4> dimsa(index_range<4>(i1, i2));
        i2[0] = 5; i2[1] = 5; i2[2] = 11; i2[3] = 11;
        dimensions<4> dimsb(index_range<4>(i1, i2));
        i2[0] = 9; i2[1] = 9; i2[2] = 11; i2[3] = 11;
        dimensions<4> dimsc(index_range<4>(i1, i2));
        block_index_space<4> bisa(dimsa), bisb(dimsb), bisc(dimsc);

        mask<4> msk1, msk2;
        msk1[0] = true; msk1[1] = true;
        msk2[2] = true; msk2[3] = true;

        bisa.split(msk1, 3);
        bisa.split(msk1, 5);
        bisa.split(msk2, 4);

        bisb.split(msk1, 4);
        bisb.split(msk2, 6);

        bisc.split(msk1, 3);
        bisc.split(msk1, 5);
        bisc.split(msk2, 6);

        block_tensor<4, double, allocator_t> bta(bisa), btb(bisb), btc(bisc);

        //  Set up symmetry

        permutation<4> p1023, p0132;
        p1023.permute(0, 1);
        p0132.permute(2, 3);
        scalar_transf<double> tr0;
        se_perm<4, double> cycle1(p1023, tr0), cycle2(p0132, tr0);
        block_tensor_ctrl<4, double> ctrla(bta), ctrlb(btb), ctrlc(btc);
        ctrla.req_symmetry().insert(cycle2);
        ctrlb.req_symmetry().insert(cycle1);
        ctrlb.req_symmetry().insert(cycle2);
        ctrlc.req_symmetry().insert(cycle1);
        ctrlc.req_symmetry().insert(cycle2);

        //  Load random data for input

        btod_random<4> rand;
        rand.perform(bta);
        rand.perform(btb);
        rand.perform(btc);
        bta.set_immutable();
        btb.set_immutable();

        //  Convert input block tensors to regular tensors

        dense_tensor<4, double, allocator_t> ta(dimsa), tb(dimsb), tc(dimsc),
                tc_ref(dimsc);
        tod_btconv<4> conva(bta);
        conva.perform(ta);
        tod_btconv<4> convb(btb);
        convb.perform(tb);
        tod_btconv<4> convc_ref(btc);
        convc_ref.perform(tc_ref);

        //  Run contraction

        contraction2<2, 2, 2> contr;
        contr.contract(2, 0);
        contr.contract(3, 1);

        btod_contract2<2, 2, 2> op(contr, bta, btb);
        op.perform(btc, 2.0);

        tod_btconv<4> convc(btc);
        convc.perform(tc);

        //  Compute reference tensor

        tod_contract2<2, 2, 2> op_ref(contr, ta, tb, 2.0);
        op_ref.perform(false, tc_ref);

        //  Compare against reference

        compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract2_test::test_contr_7() throw(libtest::test_exception) {

    //
    //  c_ijkl = a_pi b_jklp
    //  Dimensions [ijkl]=10, [p]=6, no symmetry
    //

    static const char *testname = "btod_contract2_test::test_contr_7()";

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i21, i22;
        i22[0] = 5; i22[1] = 9;
        dimensions<2> dimsa(index_range<2>(i21, i22));
        index<4> i41, i42;
        i42[0] = 9; i42[1] = 9; i42[2] = 9; i42[3] = 5;
        dimensions<4> dimsb(index_range<4>(i41, i42));
        i42[0] = 9; i42[1] = 9; i42[2] = 9; i42[3] = 9;
        dimensions<4> dimsc(index_range<4>(i41, i42));
        block_index_space<2> bisa(dimsa);
        block_index_space<4> bisb(dimsb), bisc(dimsc);

        mask<2> mska;
        mask<4> mskb, mskc;
        mska[1] = true;
        mskb[0] = true; mskb[1] = true; mskb[2] = true;
        mskc[0] = true; mskc[1] = true; mskc[2] = true; mskc[3] = true;

        bisa.split(mska, 3);
        bisb.split(mskb, 3);
        bisc.split(mskc, 3);

        block_tensor<2, double, allocator_t> bta(bisa);
        block_tensor<4, double, allocator_t> btb(bisb), btc(bisc);

        //  Load random data for input

        btod_random<2>().perform(bta);
        btod_random<4>().perform(btb);
        bta.set_immutable();
        btb.set_immutable();

        //  Run contraction

        contraction2<1, 3, 1> contr;
        contr.contract(0, 3);

        btod_contract2<1, 3, 1> op(contr, bta, btb);
        op.perform(btc);

        //  Convert block tensors to regular tensors

        dense_tensor<2, double, allocator_t> ta(dimsa);
        dense_tensor<4, double, allocator_t> tb(dimsb), tc(dimsc), tc_ref(dimsc);
        tod_btconv<2> conva(bta);
        conva.perform(ta);
        tod_btconv<4> convb(btb);
        convb.perform(tb);
        tod_btconv<4> convc(btc);
        convc.perform(tc);

        //  Compute reference tensor

        tod_contract2<1, 3, 1> op_ref(contr, ta, tb);
        op_ref.perform(true, tc_ref);

        //  Compare against reference

        compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract2_test::test_contr_8() throw(libtest::test_exception) {

    //
    //  c_ijkl = a_pi b_jklp
    //  Dimensions [ijkl]=10, [p]=6, no symmetry
    //

    static const char *testname = "btod_contract2_test::test_contr_8()";

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i21, i22;
        i22[0] = 9; i22[1] = 19;
        dimensions<2> dimsa(index_range<2>(i21, i22));
        index<4> i41, i42;
        i42[0] = 19; i42[1] = 19; i42[2] = 19; i42[3] = 9;
        dimensions<4> dimsb(index_range<4>(i41, i42));
        i42[0] = 19; i42[1] = 19; i42[2] = 19; i42[3] = 19;
        dimensions<4> dimsc(index_range<4>(i41, i42));
        block_index_space<2> bisa(dimsa);
        block_index_space<4> bisb(dimsb), bisc(dimsc);

        block_tensor<2, double, allocator_t> bta(bisa);
        block_tensor<4, double, allocator_t> btb(bisb), btc(bisc);

        //  Load random data for input

        btod_random<2>().perform(bta);
        btod_random<4>().perform(btb);
        btod_random<4>().perform(btc);
        bta.set_immutable();
        btb.set_immutable();

        //  Convert block tensors to regular tensors

        dense_tensor<2, double, allocator_t> ta(dimsa);
        dense_tensor<4, double, allocator_t> tb(dimsb), tc(dimsc), tc_ref(dimsc);
        tod_btconv<2>(bta).perform(ta);
        tod_btconv<4>(btb).perform(tb);
        tod_btconv<4>(btc).perform(tc_ref);

        //  Run contraction

        contraction2<1, 3, 1> contr;
        contr.contract(0, 3);

        btod_contract2<1, 3, 1> op(contr, bta, btb);
        op.perform(btc, 1.0);

        tod_btconv<4>(btc).perform(tc);

        //  Compute reference tensor

        tod_contract2<1, 3, 1> op_ref(contr, ta, tb);
        op_ref.perform(false, tc_ref);

        //  Compare against reference

        compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract2_test::test_contr_9() throw(libtest::test_exception) {

    //
    //  c_ijkl = - a_pi b_jklp
    //  Dimensions [ijkl]=10, [p]=6, no symmetry
    //

    static const char *testname = "btod_contract2_test::test_contr_9()";

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i21, i22;
        i22[0] = 9; i22[1] = 19;
        dimensions<2> dimsa(index_range<2>(i21, i22));
        index<4> i41, i42;
        i42[0] = 19; i42[1] = 19; i42[2] = 19; i42[3] = 9;
        dimensions<4> dimsb(index_range<4>(i41, i42));
        i42[0] = 19; i42[1] = 19; i42[2] = 19; i42[3] = 19;
        dimensions<4> dimsc(index_range<4>(i41, i42));
        block_index_space<2> bisa(dimsa);
        block_index_space<4> bisb(dimsb), bisc(dimsc);

        block_tensor<2, double, allocator_t> bta(bisa);
        block_tensor<4, double, allocator_t> btb(bisb), btc(bisc);

        //  Load random data for input

        btod_random<2>().perform(bta);
        btod_random<4>().perform(btb);
        bta.set_immutable();
        btb.set_immutable();

        //  Convert block tensors to regular tensors

        dense_tensor<2, double, allocator_t> ta(dimsa);
        dense_tensor<4, double, allocator_t> tb(dimsb), tc(dimsc), tc_ref(dimsc);
        tod_btconv<2>(bta).perform(ta);
        tod_btconv<4>(btb).perform(tb);
        tod_btconv<4>(btc).perform(tc_ref);

        //  Run contraction

        contraction2<1, 3, 1> contr;
        contr.contract(0, 3);

        btod_contract2<1, 3, 1> op(contr, bta, btb);
        op.perform(btc, -1.0);

        tod_btconv<4>(btc).perform(tc);

        //  Compute reference tensor

        tod_contract2<1, 3, 1> op_ref(contr, ta, tb, -1.0);
        op_ref.perform(false, tc_ref);

        //  Compare against reference

        compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract2_test::test_contr_10() throw(libtest::test_exception) {

    //
    //  c_ijkl = - a_pi b_jklp
    //  Dimensions [ijkl]=10, [p]=6, no symmetry
    //  Copy with -1.0 vs. a coefficient in btod_contract2
    //

    static const char *testname = "btod_contract2_test::test_contr_10()";

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i21, i22;
        i22[0] = 9; i22[1] = 19;
        dimensions<2> dimsa(index_range<2>(i21, i22));
        index<4> i41, i42;
        i42[0] = 19; i42[1] = 19; i42[2] = 19; i42[3] = 9;
        dimensions<4> dimsb(index_range<4>(i41, i42));
        i42[0] = 19; i42[1] = 19; i42[2] = 19; i42[3] = 19;
        dimensions<4> dimsc(index_range<4>(i41, i42));
        block_index_space<2> bisa(dimsa);
        block_index_space<4> bisb(dimsb), bisc(dimsc);

        block_tensor<2, double, allocator_t> bta(bisa);
        block_tensor<4, double, allocator_t> btb(bisb);
        block_tensor<4, double, allocator_t> btc(bisc), btc_ref(bisc),
                btc_ref_tmp(bisc);

        //  Load random data for input

        btod_random<2>().perform(bta);
        btod_random<4>().perform(btb);
        bta.set_immutable();
        btb.set_immutable();

        //  Convert block tensors to regular tensors

        //  Run contraction and compute the reference

        contraction2<1, 3, 1> contr;
        contr.contract(0, 3);

        btod_contract2<1, 3, 1>(contr, bta, btb).perform(btc, -1.0);
        btod_contract2<1, 3, 1>(contr, bta, btb).perform(btc_ref_tmp);
        btod_copy<4>(btc_ref_tmp, -1.0).perform(btc_ref);

        //  Compare against reference

        compare_ref<4>::compare(testname, btc, btc_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract2_test::test_contr_11() throw(libtest::test_exception) {

    //
    //  c_ijkl = a_ij b_kl
    //  Dimensions [ij] = 10, [kl]=20, no symmetry
    //

    static const char *testname = "btod_contract2_test::test_contr_11()";

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i21, i22;
        i22[0] = 9; i22[1] = 9;
        dimensions<2> dimsa(index_range<2>(i21, i22));
        i22[0] = 19; i22[1] = 19;
        dimensions<2> dimsb(index_range<2>(i21, i22));
        index<4> i41, i42;
        i42[0] = 9; i42[1] = 9; i42[2] = 19; i42[3] = 19;
        dimensions<4> dimsc(index_range<4>(i41, i42));
        block_index_space<2> bisa(dimsa), bisb(dimsb);
        block_index_space<4> bisc(dimsc);

        block_tensor<2, double, allocator_t> bta(bisa);
        block_tensor<2, double, allocator_t> btb(bisb);
        block_tensor<4, double, allocator_t> btc(bisc), btc_ref(bisc),
                btc_ref_tmp(bisc);

        //  Load random data for input

        btod_random<2>().perform(bta);
        btod_random<2>().perform(btb);
        bta.set_immutable();
        btb.set_immutable();

        //  Convert block tensors to regular tensors

        dense_tensor<2, double, allocator_t> ta(dimsa);
        dense_tensor<2, double, allocator_t> tb(dimsb);
        dense_tensor<4, double, allocator_t> tc(dimsc), tc_ref(dimsc);
        tod_btconv<2>(bta).perform(ta);
        tod_btconv<2>(btb).perform(tb);

        //  Run contraction and compute the reference

        contraction2<2, 2, 0> contr;

        btod_contract2<2, 2, 0>(contr, bta, btb).perform(btc);
        tod_btconv<4>(btc).perform(tc);
        tod_contract2<2, 2, 0>(contr, ta, tb).perform(true, tc_ref);

        //  Compare against reference

        compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract2_test::test_contr_12() throw(libtest::test_exception) {

    //
    //  c_ijkl = a_ij b_lk
    //  Dimensions [ij] = 10, [kl]=20, no symmetry
    //

    static const char *testname = "btod_contract2_test::test_contr_12()";

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i21, i22;
        i22[0] = 9; i22[1] = 9;
        dimensions<2> dimsa(index_range<2>(i21, i22));
        i22[0] = 19; i22[1] = 19;
        dimensions<2> dimsb(index_range<2>(i21, i22));
        index<4> i41, i42;
        i42[0] = 9; i42[1] = 9; i42[2] = 19; i42[3] = 19;
        dimensions<4> dimsc(index_range<4>(i41, i42));
        block_index_space<2> bisa(dimsa), bisb(dimsb);
        block_index_space<4> bisc(dimsc);

        block_tensor<2, double, allocator_t> bta(bisa);
        block_tensor<2, double, allocator_t> btb(bisb);
        block_tensor<4, double, allocator_t> btc(bisc), btc_ref(bisc),
                btc_ref_tmp(bisc);

        //  Load random data for input

        btod_random<2>().perform(bta);
        btod_random<2>().perform(btb);
        bta.set_immutable();
        btb.set_immutable();

        //  Convert block tensors to regular tensors

        dense_tensor<2, double, allocator_t> ta(dimsa);
        dense_tensor<2, double, allocator_t> tb(dimsb);
        dense_tensor<4, double, allocator_t> tc(dimsc), tc_ref(dimsc);
        tod_btconv<2>(bta).perform(ta);
        tod_btconv<2>(btb).perform(tb);

        //  Run contraction and compute the reference

        permutation<4> permc;
        permc.permute(2, 3);
        contraction2<2, 2, 0> contr(permc);

        btod_contract2<2, 2, 0>(contr, bta, btb).perform(btc);
        tod_btconv<4>(btc).perform(tc);
        tod_contract2<2, 2, 0>(contr, ta, tb).perform(true, tc_ref);

        //  Compare against reference

        compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract2_test::test_contr_13() throw(libtest::test_exception) {

    //
    //  c_ij = a_kijl b_kl
    //  Dimensions [ij] = 10, [kl]=20, no symmetry
    //

    static const char *testname = "btod_contract2_test::test_contr_13()";

    typedef std_allocator<double> allocator_t;

    try {

        index<4> i41, i42;
        i42[0] = 19; i42[1] = 9; i42[2] = 9; i42[3] = 19;
        dimensions<4> dimsa(index_range<4>(i41, i42));
        index<2> i21, i22;
        i22[0] = 19; i22[1] = 19;
        dimensions<2> dimsb(index_range<2>(i21, i22));
        i22[0] = 9; i22[1] = 9;
        dimensions<2> dimsc(index_range<2>(i21, i22));
        block_index_space<4> bisa(dimsa);
        block_index_space<2> bisb(dimsb);
        block_index_space<2> bisc(dimsc);

        block_tensor<4, double, allocator_t> bta(bisa);
        block_tensor<2, double, allocator_t> btb(bisb);
        block_tensor<2, double, allocator_t> btc(bisc), btc_ref(bisc),
                btc_ref_tmp(bisc);

        //  Load random data for input

        btod_random<4>().perform(bta);
        btod_random<2>().perform(btb);
        bta.set_immutable();
        btb.set_immutable();

        //  Convert block tensors to regular tensors

        dense_tensor<4, double, allocator_t> ta(dimsa);
        dense_tensor<2, double, allocator_t> tb(dimsb);
        dense_tensor<2, double, allocator_t> tc(dimsc), tc_ref(dimsc);
        tod_btconv<4>(bta).perform(ta);
        tod_btconv<2>(btb).perform(tb);

        //  Run contraction and compute the reference

        contraction2<2, 0, 2> contr;
        contr.contract(0, 0);
        contr.contract(3, 1);

        btod_contract2<2, 0, 2> op(contr, bta, btb);
        op.perform(btc);
        tod_btconv<2>(btc).perform(tc);
        tod_contract2<2, 0, 2>(contr, ta, tb).perform(true, tc_ref);

        //  Compare against reference

        compare_ref<2>::compare(testname, tc, tc_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract2_test::test_contr_14(double c)
throw(libtest::test_exception) {

    //
    //  c_ijkl = a_ijmn b_klmn
    //  Dimensions [ijlkmn] = 15 (three blocks), no symmetry
    //  bis of the operation and the output tensor are not equal
    //

    std::ostringstream ss;
    ss << "btod_contract2_test::test_contr_14(" << c << ")";
    std::string tn = ss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<4> i1, i2;
        i2[0] = 14; i2[1] = 14; i2[2] = 14; i2[3] = 14;
        dimensions<4> dims(index_range<4>(i1, i2));
        block_index_space<4> bisa(dims);
        mask<4> m1, m2, m3, m4;
        m1[0] = true; m2[1] = true; m3[2] = true; m4[3] = true;
        bisa.split(m1, 5); bisa.split(m1, 10);
        bisa.split(m2, 5); bisa.split(m2, 10);
        bisa.split(m3, 5); bisa.split(m3, 10);
        bisa.split(m4, 5); bisa.split(m4, 10);
        block_index_space<4> bisb(bisa), bisc(bisa);

        block_tensor<4, double, allocator_t> bta(bisa);
        block_tensor<4, double, allocator_t> btb(bisb);
        block_tensor<4, double, allocator_t> btc(bisc);

        //  Load random data for input

        btod_random<4>().perform(bta);
        btod_random<4>().perform(btb);
        btod_random<4>().perform(btc);
        bta.set_immutable();
        btb.set_immutable();

        //  Convert block tensors to regular tensors

        dense_tensor<4, double, allocator_t> ta(dims);
        dense_tensor<4, double, allocator_t> tb(dims);
        dense_tensor<4, double, allocator_t> tc(dims), tc_ref(dims);
        tod_btconv<4>(bta).perform(ta);
        tod_btconv<4>(btb).perform(tb);
        if(c != 0.0) tod_btconv<4>(btc).perform(tc_ref);

        //  Run contraction and compute the reference

        contraction2<2, 2, 2> contr;
        contr.contract(2, 2);
        contr.contract(3, 3);

        if(c == 0.0) btod_contract2<2, 2, 2>(contr, bta, btb).perform(btc);
        else btod_contract2<2, 2, 2>(contr, bta, btb).perform(btc, c);
        tod_btconv<4>(btc).perform(tc);
        if(c == 0.0) {
            tod_contract2<2, 2, 2>(contr, ta, tb).perform(true, tc_ref);
        } else {
            tod_contract2<2, 2, 2>(contr, ta, tb, c).perform(false, tc_ref);
        }

        //  Compare against reference

        compare_ref<4>::compare(tn.c_str(), tc, tc_ref, 2e-13);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void btod_contract2_test::test_contr_15(double c)
throw(libtest::test_exception) {

    //
    //  c_ijkl = a_ijmn b_klmn
    //  Dimensions [ijlkmn] = 15 (three blocks), no symmetry,
    //  only diagonal blocks are non-zero
    //  bis of the operation and the output tensor are not equal
    //

    std::ostringstream ss;
    ss << "btod_contract2_test::test_contr_15(" << c << ")";
    std::string tn = ss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<4> i1, i2;
        i2[0] = 14; i2[1] = 14; i2[2] = 14; i2[3] = 14;
        dimensions<4> dims(index_range<4>(i1, i2));
        block_index_space<4> bisa(dims);
        mask<4> m1, m2, m3, m4;
        m1[0] = true; m2[1] = true; m3[2] = true; m4[3] = true;
        bisa.split(m1, 5); bisa.split(m1, 10);
        bisa.split(m2, 5); bisa.split(m2, 10);
        bisa.split(m3, 5); bisa.split(m3, 10);
        bisa.split(m4, 5); bisa.split(m4, 10);
        block_index_space<4> bisb(bisa), bisc(bisa);

        block_tensor<4, double, allocator_t> bta(bisa);
        block_tensor<4, double, allocator_t> btb(bisb);
        block_tensor<4, double, allocator_t> btc(bisc);

        //  Load random data for input

        for(size_t i = 0; i < 3; i++) {
            index<4> blkidx;
            blkidx[0] = i; blkidx[1] = i; blkidx[2] = i;
            btod_random<4>().perform(bta, blkidx);
            btod_random<4>().perform(btb, blkidx);
            btod_random<4>().perform(btc, blkidx);
        }
        bta.set_immutable();
        btb.set_immutable();

        //  Convert block tensors to regular tensors

        dense_tensor<4, double, allocator_t> ta(dims);
        dense_tensor<4, double, allocator_t> tb(dims);
        dense_tensor<4, double, allocator_t> tc(dims), tc_ref(dims);
        tod_btconv<4>(bta).perform(ta);
        tod_btconv<4>(btb).perform(tb);
        if(c != 0.0) tod_btconv<4>(btc).perform(tc_ref);

        //  Run contraction and compute the reference

        contraction2<2, 2, 2> contr;
        contr.contract(2, 2);
        contr.contract(3, 3);

        if(c == 0.0) btod_contract2<2, 2, 2>(contr, bta, btb).perform(btc);
        else btod_contract2<2, 2, 2>(contr, bta, btb).perform(btc, c);
        tod_btconv<4>(btc).perform(tc);
        if(c == 0.0) {
            tod_contract2<2, 2, 2>(contr, ta, tb).perform(true, tc_ref);
        } else {
            tod_contract2<2, 2, 2>(contr, ta, tb, c).perform(false, tc_ref);
        }

        //  Compare against reference

        compare_ref<4>::compare(tn.c_str(), tc, tc_ref, 2e-13);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void btod_contract2_test::test_contr_16(double c)
throw(libtest::test_exception) {

    //
    //  c_iabc = a_kcad b_ikbd
    //  Dimensions [ik] = 13 (three blocks), [abcd] = 7 (three blocks),
    //  no symmetry, all blocks non-zero, empty initial result tensor
    //

    std::ostringstream ss;
    ss << "btod_contract2_test::test_contr_16(" << c << ")";
    std::string tn = ss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<4> i1, i2;
        i2[0] = 12; i2[1] = 12; i2[2] = 6; i2[3] = 6;
        dimensions<4> dims_iiaa(index_range<4>(i1, i2));
        i2[0] = 12; i2[1] = 6; i2[2] = 6; i2[3] = 6;
        dimensions<4> dims_iaaa(index_range<4>(i1, i2));
        block_index_space<4> bis_iiaa(dims_iiaa), bis_iaaa(dims_iaaa);
        mask<4> m1, m2, m3, m4;
        m1[0] = true; m1[1] = true; m2[2] = true; m2[3] = true;
        m3[0] = true; m4[1] = true; m4[2] = true; m4[3] = true;
        bis_iiaa.split(m1, 3);
        bis_iiaa.split(m1, 7);
        bis_iiaa.split(m1, 10);
        bis_iiaa.split(m2, 2);
        bis_iiaa.split(m2, 3);
        bis_iiaa.split(m2, 5);
        bis_iaaa.split(m3, 3);
        bis_iaaa.split(m3, 7);
        bis_iaaa.split(m3, 10);
        bis_iaaa.split(m4, 2);
        bis_iaaa.split(m4, 3);
        bis_iaaa.split(m4, 5);

        block_tensor<4, double, allocator_t> bta(bis_iaaa);
        block_tensor<4, double, allocator_t> btb(bis_iiaa);
        block_tensor<4, double, allocator_t> btc(bis_iaaa);

        //  Load random data for input

        btod_random<4>().perform(bta);
        btod_random<4>().perform(btb);
        bta.set_immutable();
        btb.set_immutable();

        //  Convert block tensors to regular tensors

        dense_tensor<4, double, allocator_t> ta(dims_iaaa);
        dense_tensor<4, double, allocator_t> tb(dims_iiaa);
        dense_tensor<4, double, allocator_t> tc(dims_iaaa), tc_ref(dims_iaaa);
        tod_btconv<4>(bta).perform(ta);
        tod_btconv<4>(btb).perform(tb);
        tod_set<4>().perform(tc_ref);

        //  Run contraction and compute the reference

        //  iabc = kcad ikbd
        //  caib->iabc
        contraction2<2, 2, 2> contr(permutation<4>().permute(0, 2).
                permute(2, 3));
        contr.contract(0, 1);
        contr.contract(3, 3);

        if(c == 0.0) btod_contract2<2, 2, 2>(contr, bta, btb).perform(btc);
        else btod_contract2<2, 2, 2>(contr, bta, btb).perform(btc, c);
        tod_btconv<4>(btc).perform(tc);
        if(c == 0.0) {
            tod_contract2<2, 2, 2>(contr, ta, tb).perform(true, tc_ref);
        } else {
            tod_contract2<2, 2, 2>(contr, ta, tb, c).perform(false, tc_ref);
        }

        //  Compare against reference

        compare_ref<4>::compare(tn.c_str(), tc, tc_ref, 5e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void btod_contract2_test::test_contr_17(double c)
throw(libtest::test_exception) {

    //
    //  c_ij = a_jkab b_ikab
    //  Dimensions [ijk] = 13 (three blocks), [ab] = 7 (three blocks),
    //  Perm anti-symmetry in a, no symmetry in b,
    //  all blocks non-zero
    //

    std::ostringstream ss;
    ss << "btod_contract2_test::test_contr_17(" << c << ")";
    std::string tn = ss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<4> i1, i2;
        i2[0] = 12; i2[1] = 12; i2[2] = 6; i2[3] = 6;
        dimensions<4> dims_iiaa(index_range<4>(i1, i2));
        index<2> i3, i4;
        i4[0] = 12; i4[1] = 12;
        dimensions<2> dims_ii(index_range<2>(i3, i4));
        block_index_space<4> bis_iiaa(dims_iiaa);
        block_index_space<2> bis_ii(dims_ii);
        mask<4> m1, m2;
        mask<2> m3;
        m1[0] = true; m1[1] = true; m2[2] = true; m2[3] = true;
        m3[0] = true; m3[1] = true;
        bis_iiaa.split(m1, 3);
        bis_iiaa.split(m1, 7);
        bis_iiaa.split(m1, 10);
        bis_iiaa.split(m2, 2);
        bis_iiaa.split(m2, 3);
        bis_iiaa.split(m2, 5);
        bis_ii.split(m3, 3);
        bis_ii.split(m3, 7);
        bis_ii.split(m3, 10);

        block_tensor<4, double, allocator_t> bta(bis_iiaa);
        block_tensor<4, double, allocator_t> btb(bis_iiaa);
        block_tensor<2, double, allocator_t> btc(bis_ii);

        //  Install symmetry
        {
            scalar_transf<double> tr0;
            block_tensor_ctrl<4, double> ca(bta);
            ca.req_symmetry().insert(se_perm<4, double>(permutation<4>().
                    permute(0, 1), tr0));
            ca.req_symmetry().insert(se_perm<4, double>(permutation<4>().
                    permute(2, 3), tr0));
        }

        //  Load random data for input

        btod_random<4>().perform(bta);
        btod_random<4>().perform(btb);
        if(c != 0.0) btod_random<2>().perform(btc);
        bta.set_immutable();
        btb.set_immutable();

        //  Convert block tensors to regular tensors

        dense_tensor<4, double, allocator_t> ta(dims_iiaa);
        dense_tensor<4, double, allocator_t> tb(dims_iiaa);
        dense_tensor<2, double, allocator_t> tc(dims_ii), tc_ref(dims_ii);
        tod_btconv<4>(bta).perform(ta);
        tod_btconv<4>(btb).perform(tb);
        tod_btconv<2>(btc).perform(tc_ref);

        //  Run contraction and compute the reference

        contraction2<1, 1, 3> contr(permutation<2>().permute(0, 1));
        contr.contract(1, 1);
        contr.contract(2, 2);
        contr.contract(3, 3);

        if(c == 0.0) btod_contract2<1, 1, 3>(contr, bta, btb).perform(btc);
        else btod_contract2<1, 1, 3>(contr, bta, btb).perform(btc, c);
        tod_btconv<2>(btc).perform(tc);
        if(c == 0.0) {
            tod_contract2<1, 1, 3>(contr, ta, tb).perform(true, tc_ref);
        } else {
            tod_contract2<1, 1, 3>(contr, ta, tb, c).perform(false, tc_ref);
        }

        //  Compare against reference

        compare_ref<2>::compare(tn.c_str(), tc, tc_ref, 1e-14);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void btod_contract2_test::test_contr_18(double c)
throw(libtest::test_exception) {

    //
    //  c_ij = a_jkab b_iakb
    //  Dimensions [ijk] = 13 (two blocks), [ab] = 7 (three blocks),
    //  no symmetry,
    //  all blocks non-zero
    //

    std::ostringstream ss;
    ss << "btod_contract2_test::test_contr_18(" << c << ")";
    std::string tn = ss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<4> i1, i2;
        i2[0] = 12; i2[1] = 12; i2[2] = 6; i2[3] = 6;
        dimensions<4> dims_iiaa(index_range<4>(i1, i2));
        i2[0] = 12; i2[1] = 6; i2[2] = 12; i2[3] = 6;
        dimensions<4> dims_iaia(index_range<4>(i1, i2));
        index<2> i3, i4;
        i4[0] = 12; i4[1] = 12;
        dimensions<2> dims_ii(index_range<2>(i3, i4));
        block_index_space<4> bis_iiaa(dims_iiaa), bis_iaia(dims_iaia);
        block_index_space<2> bis_ii(dims_ii);
        mask<4> m1, m2, m3, m4;
        mask<2> m5;
        m1[0] = true; m1[1] = true; m2[2] = true; m2[3] = true;
        m3[0] = true; m4[1] = true; m3[2] = true; m4[3] = true;
        m5[0] = true; m5[1] = true;
        bis_iiaa.split(m1, 3);
        bis_iiaa.split(m1, 7);
        bis_iiaa.split(m2, 2);
        bis_iiaa.split(m2, 3);
        bis_iiaa.split(m2, 5);
        bis_iaia.split(m3, 3);
        bis_iaia.split(m3, 7);
        bis_iaia.split(m4, 2);
        bis_iaia.split(m4, 3);
        bis_iaia.split(m4, 5);
        bis_ii.split(m5, 3);
        bis_ii.split(m5, 7);

        block_tensor<4, double, allocator_t> bta(bis_iiaa);
        block_tensor<4, double, allocator_t> btb(bis_iaia);
        block_tensor<2, double, allocator_t> btc(bis_ii);

        //  Load random data for input

        btod_random<4>().perform(bta);
        btod_random<4>().perform(btb);
        if(c != 0.0) btod_random<2>().perform(btc);
        bta.set_immutable();
        btb.set_immutable();

        //  Convert block tensors to regular tensors

        dense_tensor<4, double, allocator_t> ta(dims_iiaa);
        dense_tensor<4, double, allocator_t> tb(dims_iaia);
        dense_tensor<2, double, allocator_t> tc(dims_ii), tc_ref(dims_ii);
        tod_btconv<4>(bta).perform(ta);
        tod_btconv<4>(btb).perform(tb);
        tod_btconv<2>(btc).perform(tc_ref);

        //  Run contraction and compute the reference

        contraction2<1, 1, 3> contr(permutation<2>().permute(0, 1));
        contr.contract(1, 2);
        contr.contract(2, 1);
        contr.contract(3, 3);

        if(c == 0.0) btod_contract2<1, 1, 3>(contr, bta, btb).perform(btc);
        else btod_contract2<1, 1, 3>(contr, bta, btb).perform(btc, c);
        tod_btconv<2>(btc).perform(tc);
        if(c == 0.0) {
            tod_contract2<1, 1, 3>(contr, ta, tb).perform(true, tc_ref);
        } else {
            tod_contract2<1, 1, 3>(contr, ta, tb, c).perform(false, tc_ref);
        }

        //  Compare against reference

        compare_ref<2>::compare(tn.c_str(), tc, tc_ref, 1e-14);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}

void btod_contract2_test::test_contr_19()
throw(libtest::test_exception) {

    //
    //  c_ijab = a_ijkl b_klab
    //  Dimensions [ijkl] = 10 (four blocks), [ab] = 16 (six blocks),
    //  perm and label symmetry
    //

    std::ostringstream ss;
    ss << "btod_contract2_test::test_contr_19()";
    std::string tn = ss.str();

    std::vector<std::string> irreps(2);
    irreps[0] = "g"; irreps[1] = "u";
    point_group_table pg(ss.str(), irreps, irreps[0]);
    pg.add_product(1, 1, 0);

    product_table_container::get_instance().add(pg);

    typedef std_allocator<double> allocator_t;

    try {

    index<4> i1a, i2a, i1b, i2b;
    i2a[0] = 9; i2a[1] = 9; i2a[2] = 9; i2a[3] = 9;
    i2b[0] = 9; i2b[1] = 9; i2b[2] = 15; i2b[3] = 15;
    dimensions<4> dims_iiii(index_range<4>(i1a, i2a));
    dimensions<4> dims_iiaa(index_range<4>(i1b, i2b));

    block_index_space<4> bis_iiaa(dims_iiaa), bis_iiii(dims_iiii);
    mask<4> m1, m2, m3;
    m1[0] = true; m1[1] = true; m1[2] = true; m1[3] = true;
    m2[0] = true; m2[1] = true; m3[2] = true; m3[3] = true;
    bis_iiii.split(m1, 4);
    bis_iiii.split(m1, 7);
    bis_iiaa.split(m2, 4);
    bis_iiaa.split(m2, 7);
    bis_iiaa.split(m3, 5);
    bis_iiaa.split(m3, 9);
    bis_iiaa.split(m3, 15);

    block_tensor<4, double, allocator_t> bta(bis_iiii);
    block_tensor<4, double, allocator_t> btb(bis_iiaa);
    block_tensor<4, double, allocator_t> btc(bis_iiaa);
    symmetry<4, double> symc_ref(bis_iiaa);

    { // set symmetry
    scalar_transf<double> tr0, tr1(-1.);
    se_perm<4, double> sp1023(permutation<4>().permute(0, 1), tr1);
    se_perm<4, double> sp0132(permutation<4>().permute(2, 3), tr1);
    se_perm<4, double> sp2301(
            permutation<4>().permute(0, 2).permute(1, 3), tr0);

    se_label<4, double> sla(bis_iiii.get_block_index_dims(), ss.str());
    block_labeling<4> &bla = sla.get_labeling();
    bla.assign(m1, 0, 0);
    bla.assign(m1, 1, 1);
    bla.assign(m1, 2, 1);
    sla.set_rule(0);
    se_label<4, double> slb(bis_iiaa.get_block_index_dims(), ss.str());
    block_labeling<4> &blb = slb.get_labeling();
    blb.assign(m2, 0, 0);
    blb.assign(m2, 1, 1);
    blb.assign(m2, 2, 1);
    blb.assign(m3, 0, 0);
    blb.assign(m3, 1, 0);
    blb.assign(m3, 2, 1);
    blb.assign(m3, 3, 1);
    slb.set_rule(0);

    block_tensor_ctrl<4, double> ca(bta), cb(btb);
    ca.req_symmetry().insert(sp1023);
    ca.req_symmetry().insert(sp0132);
    ca.req_symmetry().insert(sp2301);
    ca.req_symmetry().insert(sla);
    cb.req_symmetry().insert(sp1023);
    cb.req_symmetry().insert(sp0132);
    cb.req_symmetry().insert(slb);

    symc_ref.insert(sp1023);
    symc_ref.insert(sp0132);
    symc_ref.insert(slb);

    }

    //  Load random data for input

    btod_random<4>().perform(bta);
    btod_random<4>().perform(btb);
    bta.set_immutable();
    btb.set_immutable();

    //  Convert block tensors to regular tensors

    dense_tensor<4, double, allocator_t> ta(dims_iiii);
    dense_tensor<4, double, allocator_t> tb(dims_iiaa);
    dense_tensor<4, double, allocator_t> tc(dims_iiaa), tc_ref(dims_iiaa);
    tod_btconv<4>(bta).perform(ta);
    tod_btconv<4>(btb).perform(tb);
    tod_btconv<4>(btc).perform(tc_ref);

    //  Run contraction and compute the reference

    contraction2<2, 2, 2> contr;
    contr.contract(2, 0);
    contr.contract(3, 1);
    btod_contract2<2, 2, 2>(contr, bta, btb).perform(btc);
    tod_btconv<4>(btc).perform(tc);
    tod_contract2<2, 2, 2>(contr, ta, tb).perform(true, tc_ref);

    //  Compare against reference

    block_tensor_ctrl<4, double> cc(btc);
    compare_ref<4>::compare(tn.c_str(), cc.req_const_symmetry(), symc_ref);

    compare_ref<4>::compare(tn.c_str(), tc, tc_ref, 1e-14);

    } catch(exception &e) {
        product_table_container::get_instance().erase(ss.str());
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    } catch(libtest::test_exception &e) {
        product_table_container::get_instance().erase(ss.str());
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }

    product_table_container::get_instance().erase(ss.str());
}


void btod_contract2_test::test_contr_20a()
throw(libtest::test_exception) {

    //
    //  c_iy = a_ix b_xy
    //  Dimensions [i] = 10 (four blocks), [xy] = 16 (two blocks),
    //  perm and part symmetry
    //

    std::ostringstream ss;
    ss << "btod_contract2_test::test_contr_20a()";
    std::string tn = ss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i1a, i2a, i1b, i2b;
        i2a[0] = 9; i2a[1] = 15;
        i2b[0] = 15; i2b[1] = 15;

        block_index_space<2> bisa(dimensions<2>(index_range<2>(i1a, i2a)));
        block_index_space<2> bisb(dimensions<2>(index_range<2>(i1b, i2b)));

        mask<2> m01, m10, m11;
        m10[0] = true; m01[1] = true;
        m11[0] = true; m11[1] = true;
        bisa.split(m10, 3);
        bisa.split(m10, 5);
        bisa.split(m10, 8);
        bisa.split(m01, 8);
        bisb.split(m11, 8);

        block_tensor<2, double, allocator_t> bta(bisa);
        block_tensor<2, double, allocator_t> btb(bisb);
        block_tensor<2, double, allocator_t> btc(bisa);

        { // set symmetry
            scalar_transf<double> tr0;
            se_perm<2, double> sp10(permutation<2>().permute(0, 1), tr0);
            se_part<2, double> spa(bisa, m10, 2);
            index<2> i00, i10; i10[0] = 1;
            spa.add_map(i00, i10, tr0);

            block_tensor_ctrl<2, double> ca(bta), cb(btb);
            ca.req_symmetry().insert(spa);
            cb.req_symmetry().insert(sp10);
        }

        //  Load random data for input

        btod_random<2>().perform(bta);
        btod_random<2>().perform(btb);
        bta.set_immutable();
        btb.set_immutable();

        //  Convert block tensors to regular tensors

        dense_tensor<2, double, allocator_t> ta(bisa.get_dims());
        dense_tensor<2, double, allocator_t> tb(bisb.get_dims());
        dense_tensor<2, double, allocator_t> tc(bisa.get_dims()), tc_ref(bisa.get_dims());
        tod_btconv<2>(bta).perform(ta);
        tod_btconv<2>(btb).perform(tb);

        //  Run contraction and compute the reference

        contraction2<1, 1, 1> contr;
        contr.contract(1, 0);
        btod_contract2<1, 1, 1>(contr, bta, btb).perform(btc);
        tod_btconv<2>(btc).perform(tc);
        tod_contract2<1, 1, 1>(contr, ta, tb).perform(true, tc_ref);

        //  Compare against reference

        compare_ref<2>::compare(tn.c_str(), tc, tc_ref, 1e-14);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }

}

void btod_contract2_test::test_contr_20b()
throw(libtest::test_exception) {

    //
    //  c_iy = a_ix b_xy
    //  Dimensions [i] = 10 (four blocks), [xy] = 16 (six blocks),
    //  perm and part symmetry
    //

    std::ostringstream ss;
    ss << "btod_contract2_test::test_contr_20b()";
    std::string tn = ss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i1a, i2a, i1b, i2b;
        i2a[0] = 9; i2a[1] = 15;
        i2b[0] = 15; i2b[1] = 15;

        block_index_space<2> bisa(dimensions<2>(index_range<2>(i1a, i2a)));
        block_index_space<2> bisb(dimensions<2>(index_range<2>(i1b, i2b)));

        mask<2> m01, m10, m11;
        m10[0] = true; m01[1] = true;
        m11[0] = true; m11[1] = true;
        bisa.split(m10, 3);
        bisa.split(m10, 5);
        bisa.split(m10, 8);
        bisa.split(m01, 2);
        bisa.split(m01, 5);
        bisa.split(m01, 8);
        bisa.split(m01, 10);
        bisa.split(m01, 13);
        bisb.split(m11, 2);
        bisb.split(m11, 5);
        bisb.split(m11, 8);
        bisb.split(m11, 10);
        bisb.split(m11, 13);

        block_tensor<2, double, allocator_t> bta(bisa);
        block_tensor<2, double, allocator_t> btb(bisb);
        block_tensor<2, double, allocator_t> btc(bisa);

        symmetry<2, double> sym_ref(bisa);

        { // Set symmetry
            scalar_transf<double> tr0;
            se_perm<2, double> sp10(permutation<2>().permute(0, 1), tr0);
            se_part<2, double> spa(bisa, m11, 2), spb(bisb, m11, 2);
            index<2> i00, i01, i10, i11;
            i10[0] = 1; i01[1] = 1;
            i11[0] = 1; i11[1] = 1;
            spa.add_map(i00, i11, tr0);
            spa.mark_forbidden(i01);
            spa.mark_forbidden(i10);
            spb.add_map(i00, i11, tr0);
            spb.mark_forbidden(i01);
            spb.mark_forbidden(i10);

            block_tensor_ctrl<2, double> ca(bta), cb(btb);
            ca.req_symmetry().insert(spa);
            cb.req_symmetry().insert(spb);
            cb.req_symmetry().insert(sp10);
            sym_ref.insert(spa);
        }

        //  Load random data for input

        btod_random<2>().perform(bta);
        btod_random<2>().perform(btb);
        bta.set_immutable();
        btb.set_immutable();

        //  Convert block tensors to regular tensors

        dense_tensor<2, double, allocator_t> ta(bisa.get_dims());
        dense_tensor<2, double, allocator_t> tb(bisb.get_dims());
        dense_tensor<2, double, allocator_t> tc(bisa.get_dims()), tc_ref(bisa.get_dims());
        tod_btconv<2>(bta).perform(ta);
        tod_btconv<2>(btb).perform(tb);

        //  Run contraction and compute the reference

        contraction2<1, 1, 1> contr;
        contr.contract(1, 0);
        btod_contract2<1, 1, 1>(contr, bta, btb).perform(btc);
        tod_btconv<2>(btc).perform(tc);
        tod_contract2<1, 1, 1>(contr, ta, tb).perform(true, tc_ref);

        //  Compare against reference
        {
            block_tensor_ctrl<2, double> cc(btc);
            compare_ref<2>::compare(tn.c_str(), cc.req_const_symmetry(), sym_ref);
        }

        compare_ref<2>::compare(tn.c_str(), tc, tc_ref, 1e-14);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }

}

/** \test Tests contraction \f$ c_{ij} = a_{ip} b_{jp} \f$.
Dimensions [ij] = 10 (two blocks), [p] = 12 (two blocks).
No symmetry in A, partition symmetry in B.
Zero non-diagonal blocks.
 **/
void btod_contract2_test::test_contr_21() throw(libtest::test_exception) {

    std::ostringstream ss;
    ss << "btod_contract2_test::test_contr_21()";
    std::string tn = ss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i1a, i2a, i1c, i2c;
        i2a[0] = 9; i2a[1] = 11;
        i2c[0] = 9; i2c[1] = 9;
        index<2> i00, i01, i10, i11;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;

        block_index_space<2> bisa(dimensions<2>(index_range<2>(i1a, i2a)));
        block_index_space<2> bisc(dimensions<2>(index_range<2>(i1c, i2c)));

        mask<2> m01, m10, m11;
        m10[0] = true; m01[1] = true;
        m11[0] = true; m11[1] = true;
        bisa.split(m10, 5);
        bisa.split(m01, 6);
        bisc.split(m11, 5);

        block_tensor<2, double, allocator_t> bta(bisa);
        block_tensor<2, double, allocator_t> btb(bisa);
        block_tensor<2, double, allocator_t> btc(bisc);

        { // set symmetry
            scalar_transf<double> tr0;
            se_part<2, double> spa(bisa, m11, 2);
            spa.add_map(i00, i11, tr0);
            spa.mark_forbidden(i01);
            spa.mark_forbidden(i10);

            block_tensor_ctrl<2, double> ca(bta), cb(btb);
            //      ca.req_symmetry().insert(spa);
            cb.req_symmetry().insert(spa);
        }

        //  Load random data for input

        btod_random<2>().perform(bta);
        btod_random<2>().perform(btb);

        { // zero out non-diagonal blocks
            block_tensor_ctrl<2, double> ca(bta), cb(btb);
            ca.req_zero_block(i01);
            ca.req_zero_block(i10);
        }
        bta.set_immutable();
        btb.set_immutable();

        //  Convert block tensors to regular tensors

        dense_tensor<2, double, allocator_t> ta(bisa.get_dims());
        dense_tensor<2, double, allocator_t> tb(bisa.get_dims());
        dense_tensor<2, double, allocator_t> tc(bisc.get_dims()),
                tc_ref(bisc.get_dims());
        tod_btconv<2>(bta).perform(ta);
        tod_btconv<2>(btb).perform(tb);

        //  Run contraction and compute the reference

        contraction2<1, 1, 1> contr;
        contr.contract(1, 1);
        btod_contract2<1, 1, 1>(contr, bta, btb).perform(btc);
        tod_btconv<2>(btc).perform(tc);
        tod_contract2<1, 1, 1>(contr, ta, tb).perform(true, tc_ref);

        //  Compare against reference

        compare_ref<2>::compare(tn.c_str(), tc, tc_ref, 1e-14);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }

}


void btod_contract2_test::test_contr_22() {

    //
    //  c_{ijkl} = a_{kpr} a_{lqr} b_{ijpq}
    //  [k,l,p,q] = 9, [ij] = 5, [r] = 11
    //  Permutational antisymmetry between (i,j) and (p,q) in b_{ijpq}
    //  Permutation symmetry between (k,p) in a_{kpr}
    //  Contraction is done in two steps
    //

    static const char *testname = "btod_contract2_test::test_contr_22()";

    typedef std_allocator<double> allocator_t;

    try {

        size_t ni = 5, nj = ni, nk = 9, nl = nk, np = nk, nq = nk, nr = 11;

        index<3> ia;
        ia[0] = nk - 1; ia[1] = np - 1; ia[2] = nr - 1;
        dimensions<3> dimsa(index_range<3>(index<3>(), ia));
        index<4> ib;
        ib[0] = ni - 1; ib[1] = nj - 1; ib[2] = np - 1; ib[3] = nq - 1;
        dimensions<4> dimsb(index_range<4>(index<4>(), ib));
        index<4> ic;
        ic[0] = ni - 1; ic[1] = nj - 1; ic[2] = nk - 1; ic[3] = nl - 1;
        dimensions<4> dimsc(index_range<4>(index<4>(), ic));
        index<4> ii;
        ii[0] = nk - 1; ii[1] = nl - 1; ii[2] = np - 1; ii[3] = nq - 1;
        dimensions<4> dimsi(index_range<4>(index<4>(), ii));

        block_index_space<3> bisa(dimsa);
        mask<3> m110, m001;
        m110[0] = true; m110[1] = true; m001[2] = true;
        bisa.split(m110, 3);
        bisa.split(m110, 7);
        bisa.split(m001, 5);

        block_index_space<4> bisb(dimsb);
        mask<4> m1100, m0011;
        m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;
        bisb.split(m1100, 2);
        bisb.split(m1100, 3);
        bisb.split(m0011, 3);
        bisb.split(m0011, 7);

        block_index_space<4> bisc(dimsc);
        bisc.split(m1100, 2);
        bisc.split(m1100, 3);
        bisc.split(m0011, 3);
        bisc.split(m0011, 7);

        block_index_space<4> bisi(dimsi);
        mask<4> m1111;
        m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
        bisi.split(m1111, 3);
        bisi.split(m1111, 7);

        block_tensor<3, double, allocator_t> bta(bisa);
        block_tensor<4, double, allocator_t> btb(bisb), btc(bisc), bti(bisi);

        //  Set symmetry

        se_perm<3, double> seperma1(permutation<3>().permute(0, 1),
            scalar_transf<double>(1.0));
        se_perm<4, double> sepermb1(permutation<4>().permute(0, 1),
            scalar_transf<double>(-1.0));
        se_perm<4, double> sepermb2(permutation<4>().permute(2, 3),
            scalar_transf<double>(-1.0));
        se_perm<4, double> sepermi1(permutation<4>().permute(0, 2),
            scalar_transf<double>(1.0));
        se_perm<4, double> sepermi2(permutation<4>().permute(1, 3),
            scalar_transf<double>(1.0));
        se_perm<4, double> sepermi3(permutation<4>().permute(0, 1).
            permute(2, 3), scalar_transf<double>(1.0));
        symmetry<4, double> symi_ref(bisi);

        {
            block_tensor_ctrl<3, double> ca(bta);
            ca.req_symmetry().insert(seperma1);

            block_tensor_ctrl<4, double> cb(btb);
            cb.req_symmetry().insert(sepermb1);
            cb.req_symmetry().insert(sepermb2);

            symi_ref.insert(sepermi1);
            symi_ref.insert(sepermi2);
            symi_ref.insert(sepermi3);
        }

        //  Load random data for input

        btod_random<3>().perform(bta);
        btod_random<4>().perform(btb);
        bta.set_immutable();
        btb.set_immutable();

        //  Run contraction

        // a_{kpr} a_{lqr} -> I_{klpq}
        // kplq -> klpq
        contraction2<2, 2, 1> contr1(permutation<4>().permute(1, 2));
        contr1.contract(2, 2);
        // I_{klpq} b_{ijpq} -> c_{ijkl}
        // klij -> ijkl
        contraction2<2, 2, 2> contr2(permutation<4>().permute(0, 2).
            permute(1, 3));
        contr2.contract(2, 2);
        contr2.contract(3, 3);

        btod_contract2<2, 2, 1>(contr1, bta, bta).perform(bti);
        {
        block_tensor_ctrl<4, double> ctrli(bti);
        compare_ref<4>::compare(testname, ctrli.req_const_symmetry(), symi_ref);
        }
        btod_contract2<2, 2, 2>(contr2, bti, btb).perform(btc);

        //  Convert block tensors to regular tensors

        dense_tensor<3, double, allocator_t> ta(dimsa);
        dense_tensor<4, double, allocator_t> tb(dimsb), ti(dimsi),
            ti_ref(dimsi), tc(dimsc), tc_ref(dimsc);
        tod_btconv<3>(bta).perform(ta);
        tod_btconv<4>(btb).perform(tb);
        tod_btconv<4>(btc).perform(tc);
        tod_btconv<4>(bti).perform(ti);

        //  Compute reference tensor

        tod_contract2<2, 2, 1>(contr1, ta, ta).perform(true, ti_ref);
        tod_contract2<2, 2, 2>(contr2, ti_ref, tb).perform(true, tc_ref);

        //  Compare against reference

        compare_ref<4>::compare(testname, ti, ti_ref, 1e-13);
        compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract2_test::test_contr_23() {

    const char *testname = "btod_contract2_test::test_contr_23()";

    typedef std_allocator<double> allocator_t;

    try {

    index<4> i4a, i4b;
    i4b[0] = 9; i4b[1] = 9; i4b[2] = 10; i4b[3] = 19;
    dimensions<4> dims_ijka(index_range<4>(i4a, i4b));
    i4b[0] = 10; i4b[1] = 9; i4b[2] = 9; i4b[3] = 19;
    dimensions<4> dims_kija(index_range<4>(i4a, i4b));
    i4b[0] = 9; i4b[1] = 9; i4b[2] = 19; i4b[3] = 19;
    dimensions<4> dims_ijab(index_range<4>(i4a, i4b));

    block_index_space<4> bis_ijka(dims_ijka), bis_kija(dims_kija),
        bis_ijab(dims_ijab);
    mask<4> m0001, m0011, m0110, m1000, m1100;
    m1000[0] = true; m0001[3] = true;
    m0110[1] = true; m0110[2] = true;
    m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;
    bis_ijka.split(m1100, 3);
    bis_ijka.split(m1100, 5);
    bis_ijka.split(m0001, 6);
    bis_ijka.split(m0001, 13);
    bis_kija.split(m0110, 3);
    bis_kija.split(m0110, 5);
    bis_kija.split(m0001, 6);
    bis_kija.split(m0001, 13);
    bis_ijab.split(m1100, 3);
    bis_ijab.split(m1100, 5);
    bis_ijab.split(m0011, 6);
    bis_ijab.split(m0011, 13);

    block_tensor<4, double, allocator_t> bt1(bis_kija), bt2(bis_ijab),
        bt3(bis_ijka);

    btod_random<4>().perform(bt1);
    btod_random<4>().perform(bt2);
    bt1.set_immutable();
    bt2.set_immutable();

    contraction2<2, 2, 2> contr(permutation<4>().permute(0, 2));
    contr.contract(1, 1);
    contr.contract(3, 3);
    btod_contract2<2, 2, 2>(contr, bt1, bt2).perform(bt3);

    dense_tensor<4, double, allocator_t> t1(dims_kija), t2(dims_ijab),
        t3(dims_ijka), t3_ref(dims_ijka);
    tod_btconv<4>(bt1).perform(t1);
    tod_btconv<4>(bt2).perform(t2);
    tod_btconv<4>(bt3).perform(t3);
    tod_contract2<2, 2, 2>(contr, t1, t2).perform(true, t3_ref);

    compare_ref<4>::compare(testname, t3, t3_ref, 6e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Tests \f$ c_{ijab} = a_{ia} a_{jb} \f$, expected perm symmetry
\f$ c_{ijab} = c_{jiba} \f$.
 **/
void btod_contract2_test::test_self_1() throw(libtest::test_exception) {

    static const char *testname = "btod_contract2_test::test_self_1()";

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i2a, i2b;
        i2b[0] = 10; i2b[1] = 20;
        dimensions<2> dims_ia(index_range<2>(i2a, i2b));
        index<4> i4a, i4b;
        i4b[0] = 10; i4b[1] = 10; i4b[2] = 20; i4b[3] = 20;
        dimensions<4> dims_ijab(index_range<4>(i4a, i4b));
        block_index_space<2> bis_ia(dims_ia);
        block_index_space<4> bis_ijab(dims_ijab);
        mask<2> m2i, m2a;
        m2i[0] = true; m2a[1] = true;
        mask<4> m4i, m4a;
        m4i[0] = true; m4i[1] = true; m4a[2] = true; m4a[3] = true;
        bis_ia.split(m2i, 3);
        bis_ia.split(m2i, 5);
        bis_ia.split(m2a, 10);
        bis_ia.split(m2a, 14);
        bis_ijab.split(m4i, 3);
        bis_ijab.split(m4i, 5);
        bis_ijab.split(m4a, 10);
        bis_ijab.split(m4a, 14);

        block_tensor<2, double, allocator_t> bta(bis_ia);
        block_tensor<4, double, allocator_t> btc(bis_ijab);

        //  Load random data for input

        btod_random<2>().perform(bta);
        bta.set_immutable();

        //  Run contraction

        contraction2<2, 2, 0> contr(permutation<4>().permute(1, 2));
        btod_contract2<2, 2, 0>(contr, bta, bta).perform(btc);

        //  Convert block tensors to regular tensors

        dense_tensor<2, double, allocator_t> ta(dims_ia);
        dense_tensor<4, double, allocator_t> tc(dims_ijab), tc_ref(dims_ijab);
        tod_btconv<2>(bta).perform(ta);
        tod_btconv<4>(btc).perform(tc);

        //  Compute reference symmetry and tensor

        symmetry<4, double> symc(bis_ijab), symc_ref(bis_ijab);
        scalar_transf<double> tr0;
       symc_ref.insert(se_perm<4, double>(permutation<4>().
                permute(0, 1).permute(2, 3), tr0));
        {
            block_tensor_ctrl<4, double> cc(btc);
            so_copy<4, double>(cc.req_const_symmetry()).perform(symc);
        }
        tod_contract2<2, 2, 0>(contr, ta, ta).perform(true, tc_ref);

        //  Compare against reference

        compare_ref<4>::compare(testname, symc, symc_ref);
        compare_ref<4>::compare(testname, tc, tc_ref, 5e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Tests \f$ c_{ijab} = \sum_c a_{iac} a_{jbc} \f$,
expected perm symmetry \f$ c_{ijab} = c_{jiba} \f$.
 **/
void btod_contract2_test::test_self_2() throw(libtest::test_exception) {

    static const char *testname = "btod_contract2_test::test_self_2()";

    typedef std_allocator<double> allocator_t;

    try {

        index<3> i3a, i3b;
        i3b[0] = 10; i3b[1] = 20; i3b[2] = 20;
        dimensions<3> dims_iac(index_range<3>(i3a, i3b));
        index<4> i4a, i4b;
        i4b[0] = 10; i4b[1] = 10; i4b[2] = 20; i4b[3] = 20;
        dimensions<4> dims_ijab(index_range<4>(i4a, i4b));
        block_index_space<3> bis_iac(dims_iac);
        block_index_space<4> bis_ijab(dims_ijab);
        mask<3> m3i, m3a;
        m3i[0] = true; m3a[1] = true; m3a[2] = true;
        mask<4> m4i, m4a;
        m4i[0] = true; m4i[1] = true; m4a[2] = true; m4a[3] = true;
        bis_iac.split(m3i, 3);
        bis_iac.split(m3i, 5);
        bis_iac.split(m3a, 10);
        bis_iac.split(m3a, 14);
        bis_ijab.split(m4i, 3);
        bis_ijab.split(m4i, 5);
        bis_ijab.split(m4a, 10);
        bis_ijab.split(m4a, 14);

        block_tensor<3, double, allocator_t> bta(bis_iac);
        block_tensor<4, double, allocator_t> btc(bis_ijab);

        //  Load random data for input

        btod_random<3>().perform(bta);
        bta.set_immutable();

        //  Run contraction

        contraction2<2, 2, 1> contr(permutation<4>().permute(1, 2));
        contr.contract(2, 2);
        btod_contract2<2, 2, 1>(contr, bta, bta).perform(btc);

        //  Convert block tensors to regular tensors

        dense_tensor<3, double, allocator_t> ta(dims_iac);
        dense_tensor<4, double, allocator_t> tc(dims_ijab), tc_ref(dims_ijab);
        tod_btconv<3>(bta).perform(ta);
        tod_btconv<4>(btc).perform(tc);

        //  Compute reference symmetry and tensor

        symmetry<4, double> symc(bis_ijab), symc_ref(bis_ijab);
        scalar_transf<double> tr0;
        symc_ref.insert(se_perm<4, double>(permutation<4>().
                permute(0, 1).permute(2, 3), tr0));
        {
            block_tensor_ctrl<4, double> cc(btc);
            so_copy<4, double>(cc.req_const_symmetry()).perform(symc);
        }
        tod_contract2<2, 2, 1>(contr, ta, ta).perform(true, tc_ref);

        //  Compare against reference

        compare_ref<4>::compare(testname, symc, symc_ref);
        compare_ref<4>::compare(testname, tc, tc_ref, 5e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Tests \f$ c_{ijab} = \sum_c a_{iac} a_{jcb} \f$,
initial perm symmetry \f$ a_{iac} = a_{ica} \f$,
expected perm symmetry \f$ c_{ijab} = c_{jiba} \f$.
 **/
void btod_contract2_test::test_self_3() throw(libtest::test_exception) {

    static const char *testname = "btod_contract2_test::test_self_3()";

    typedef std_allocator<double> allocator_t;

    try {

        index<3> i3a, i3b;
        i3b[0] = 10; i3b[1] = 20; i3b[2] = 20;
        dimensions<3> dims_iac(index_range<3>(i3a, i3b));
        index<4> i4a, i4b;
        i4b[0] = 10; i4b[1] = 10; i4b[2] = 20; i4b[3] = 20;
        dimensions<4> dims_ijab(index_range<4>(i4a, i4b));
        block_index_space<3> bis_iac(dims_iac);
        block_index_space<4> bis_ijab(dims_ijab);
        mask<3> m3i, m3a;
        m3i[0] = true; m3a[1] = true; m3a[2] = true;
        mask<4> m4i, m4a;
        m4i[0] = true; m4i[1] = true; m4a[2] = true; m4a[3] = true;
        bis_iac.split(m3i, 3);
        bis_iac.split(m3i, 5);
        bis_iac.split(m3a, 10);
        bis_iac.split(m3a, 14);
        bis_ijab.split(m4i, 3);
        bis_ijab.split(m4i, 5);
        bis_ijab.split(m4a, 10);
        bis_ijab.split(m4a, 14);

        block_tensor<3, double, allocator_t> bta(bis_iac);
        block_tensor<4, double, allocator_t> btc(bis_ijab);

        //  Install initial symmetry

        {
            block_tensor_ctrl<3, double> ca(bta);
            scalar_transf<double> tr0;
            ca.req_symmetry().insert(se_perm<3, double>(
                    permutation<3>().permute(1, 2), tr0));
        }

        //  Load random data for input

        btod_random<3>().perform(bta);
        bta.set_immutable();

        //  Run contraction

        contraction2<2, 2, 1> contr(permutation<4>().permute(1, 2));
        contr.contract(2, 1);
        btod_contract2<2, 2, 1>(contr, bta, bta).perform(btc);

        //  Convert block tensors to regular tensors

        dense_tensor<3, double, allocator_t> ta(dims_iac);
        dense_tensor<4, double, allocator_t> tc(dims_ijab), tc_ref(dims_ijab);
        tod_btconv<3>(bta).perform(ta);
        tod_btconv<4>(btc).perform(tc);

        //  Compute reference symmetry and tensor

        symmetry<4, double> symc(bis_ijab), symc_ref(bis_ijab);
        scalar_transf<double> tr0;
        symc_ref.insert(se_perm<4, double>(permutation<4>().
                permute(0, 1).permute(2, 3), tr0));
        {
            block_tensor_ctrl<4, double> cc(btc);
            so_copy<4, double>(cc.req_const_symmetry()).perform(symc);
        }
        tod_contract2<2, 2, 1>(contr, ta, ta).perform(true, tc_ref);

        //  Compare against reference

        compare_ref<4>::compare(testname, symc, symc_ref);
        compare_ref<4>::compare(testname, tc, tc_ref, 5e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

