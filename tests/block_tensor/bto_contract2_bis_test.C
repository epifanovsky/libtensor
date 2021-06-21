#include <sstream>
#include <libtensor/gen_block_tensor/impl/gen_bto_contract2_bis.h>
#include "bto_contract2_bis_test.h"

namespace libtensor {


void bto_contract2_bis_test::perform() {

    test_1();
    test_2();
    test_3();
    test_4();
    test_5();
    test_6();
}


void bto_contract2_bis_test::test_1() {

    //
    //  c_ijkl = a_ijkp b_lp
    //  [ij] = 5  (no splits)
    //  [kl] = 10 (no splits)
    //  [p]  = 4  (no splits)
    //
    //  Expected block index space:
    //  [ijkl] have correct dimensions, no splits
    //

    static const char *testname = "bto_contract2_bis_test::test_1()";

    try {

        libtensor::index<4> ia1, ia2;
        ia2[0] = 4; ia2[1] = 4; ia2[2] = 9; ia2[3] = 3;
        dimensions<4> dimsa(index_range<4>(ia1, ia2));
        block_index_space<4> bisa(dimsa);

        libtensor::index<2> ib1, ib2;
        ib2[0] = 9; ib2[1] = 3;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));
        block_index_space<2> bisb(dimsb);

        libtensor::index<4> ic1, ic2;
        ic2[0] = 4; ic2[1] = 4; ic2[2] = 9; ic2[3] = 9;
        dimensions<4> dimsc(index_range<4>(ic1, ic2));
        block_index_space<4> bisc_ref(dimsc);

        contraction2<3, 1, 1> contr;
        contr.contract(3, 1);

        gen_bto_contract2_bis<3, 1, 1> op(contr, bisa, bisb);

        if(!op.get_bis().equals(bisc_ref)) {
            fail_test(testname, __FILE__, __LINE__,
                "Incorrect output block index space.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bto_contract2_bis_test::test_2() {

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

    static const char *testname = "bto_contract2_bis_test::test_2()";

    try {

        libtensor::index<4> ia1, ia2;
        ia2[0] = 4; ia2[1] = 4; ia2[2] = 9; ia2[3] = 3;
        dimensions<4> dimsa(index_range<4>(ia1, ia2));
        block_index_space<4> bisa(dimsa);
        mask<4> ma1;
        ma1[2] = true;
        bisa.split(ma1, 4);

        libtensor::index<2> ib1, ib2;
        ib2[0] = 9; ib2[1] = 3;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));
        block_index_space<2> bisb(dimsb);
        mask<2> mb1;
        mb1[0] = true;
        bisb.split(mb1, 4);

        libtensor::index<4> ic1, ic2;
        ic2[0] = 4; ic2[1] = 4; ic2[2] = 9; ic2[3] = 9;
        dimensions<4> dimsc(index_range<4>(ic1, ic2));
        block_index_space<4> bisc_ref(dimsc);
        mask<4> mc1;
        mc1[2] = true; mc1[3] = true;
        bisc_ref.split(mc1, 4);

        contraction2<3, 1, 1> contr;
        contr.contract(3, 1);

        gen_bto_contract2_bis<3, 1, 1> op(contr, bisa, bisb);

        if(!op.get_bis().equals(bisc_ref)) {
            fail_test(testname, __FILE__, __LINE__,
                "Incorrect output block index space.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bto_contract2_bis_test::test_3() {

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

    static const char *testname = "bto_contract2_bis_test::test_3()";

    try {

        libtensor::index<4> ia1, ia2;
        ia2[0] = 4; ia2[1] = 4; ia2[2] = 9; ia2[3] = 3;
        dimensions<4> dimsa(index_range<4>(ia1, ia2));
        block_index_space<4> bisa(dimsa);
        mask<4> ma1, ma2;
        ma1[0] = true; ma1[1] = true;
        ma2[2] = true;
        bisa.split(ma1, 2);
        bisa.split(ma2, 4);

        libtensor::index<2> ib1, ib2;
        ib2[0] = 9; ib2[1] = 3;
        dimensions<2> dimsb(index_range<2>(ib1, ib2));
        block_index_space<2> bisb(dimsb);
        mask<2> mb1;
        mb1[0] = true;
        bisb.split(mb1, 4);

        libtensor::index<4> ic1, ic2;
        ic2[0] = 4; ic2[1] = 4; ic2[2] = 9; ic2[3] = 9;
        dimensions<4> dimsc(index_range<4>(ic1, ic2));
        block_index_space<4> bisc_ref(dimsc);
        mask<4> mc1, mc2;
        mc1[0] = true; mc1[1] = true;
        mc2[2] = true; mc2[3] = true;
        bisc_ref.split(mc1, 2);
        bisc_ref.split(mc2, 4);

        contraction2<3, 1, 1> contr;
        contr.contract(3, 1);

        gen_bto_contract2_bis<3, 1, 1> op(contr, bisa, bisb);

        if(!op.get_bis().equals(bisc_ref)) {
            fail_test(testname, __FILE__, __LINE__,
                "Incorrect output block index space.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bto_contract2_bis_test::test_4() {

    //
    //  c_ijkl = a_ijpq b_klpq
    //  [ijklpq] = 11 (3, 2, 5)
    //
    //  Expected block index space:
    //  [ijkl] have correct dimensions, one splitting pattern
    //

    static const char *testname = "bto_contract2_bis_test::test_4()";

    try {

        libtensor::index<4> ia1, ia2;
        ia2[0] = 10; ia2[1] = 10; ia2[2] = 10; ia2[3] = 10;
        dimensions<4> dimsa(index_range<4>(ia1, ia2));
        block_index_space<4> bisa(dimsa);
        mask<4> ma1;
        ma1[0] = true; ma1[1] = true; ma1[2] = true; ma1[3] = true;
        bisa.split(ma1, 3);
        bisa.split(ma1, 5);

        block_index_space<4> bisb(bisa), bisc_ref(bisa);

        contraction2<2, 2, 2> contr;
        contr.contract(0, 2);
        contr.contract(1, 3);

        gen_bto_contract2_bis<2, 2, 2> op(contr, bisa, bisb);

        if(!op.get_bis().equals(bisc_ref)) {
            fail_test(testname, __FILE__, __LINE__,
                "Incorrect output block index space.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bto_contract2_bis_test::test_5() {

    //
    //  c_ijk = a_ipqr b_jpqrk
    //  [ijpqr] = 11 (3, 2, 5)
    //  [k]     = 9  (4, 5)
    //
    //  Expected block index space:
    //  [ijk] have correct dimensions,
    //  [ij] and [k] preserve the splitting pattern
    //

    static const char *testname = "bto_contract2_bis_test::test_5()";

    try {

        libtensor::index<4> ia1, ia2;
        ia2[0] = 10; ia2[1] = 10; ia2[2] = 10; ia2[3] = 10;
        dimensions<4> dimsa(index_range<4>(ia1, ia2));
        block_index_space<4> bisa(dimsa);
        mask<4> ma1;
        ma1[0] = true; ma1[1] = true; ma1[2] = true; ma1[3] = true;
        bisa.split(ma1, 3);
        bisa.split(ma1, 5);

        libtensor::index<5> ib1, ib2;
        ib2[0] = 10; ib2[1] = 10; ib2[2] = 10; ib2[3] = 10; ib2[4] = 8;
        dimensions<5> dimsb(index_range<5>(ib1, ib2));
        block_index_space<5> bisb(dimsb);
        mask<5> mb1, mb2;
        mb1[0] = true; mb1[1] = true; mb1[2] = true; mb1[3] = true;
        mb2[4] = true;
        bisb.split(mb1, 3);
        bisb.split(mb1, 5);
        bisb.split(mb2, 4);

        libtensor::index<3> ic1, ic2;
        ic2[0] = 10; ic2[1] = 10; ic2[2] = 8;
        dimensions<3> dimsc(index_range<3>(ic1, ic2));
        block_index_space<3> bisc_ref(dimsc);
        mask<3> mc1, mc2;
        mc1[0] = true; mc1[1] = true;
        mc2[2] = true;
        bisc_ref.split(mc1, 3);
        bisc_ref.split(mc1, 5);
        bisc_ref.split(mc2, 4);

        contraction2<1, 2, 3> contr;
        contr.contract(1, 1);
        contr.contract(2, 2);
        contr.contract(3, 3);

        gen_bto_contract2_bis<1, 2, 3> op(contr, bisa, bisb);

        if(!op.get_bis().equals(bisc_ref)) {
            fail_test(testname, __FILE__, __LINE__,
                "Invalid output block index space.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bto_contract2_bis_test::test_6() {

    //
    //  c_{ijab} = a_{ia} a_{jb}
    //  [ij] = 11 (3, 5)
    //  [ab] = 21 (10, 14)
    //

    static const char *testname = "bto_contract2_bis_test::test_6()";

    try {

        libtensor::index<2> ia1, ia2;
        ia2[0] = 10; ia2[1] = 20;
        dimensions<2> dimsa(index_range<2>(ia1, ia2));
        block_index_space<2> bisa(dimsa);
        mask<2> ma10, ma01;
        ma10[0] = true; ma01[1] = true;
        bisa.split(ma10, 3);
        bisa.split(ma10, 5);
        bisa.split(ma01, 10);
        bisa.split(ma01, 14);

        block_index_space<2> bisb(bisa);

        libtensor::index<4> ic1, ic2;
        ic2[0] = 10; ic2[1] = 10; ic2[2] = 20; ic2[3] = 20;
        dimensions<4> dimsc(index_range<4>(ic1, ic2));
        block_index_space<4> bisc_ref(dimsc);
        mask<4> mc1100, mc0011;
        mc1100[0] = true; mc1100[1] = true; mc0011[2] = true; mc0011[3] = true;
        bisc_ref.split(mc1100, 3);
        bisc_ref.split(mc1100, 5);
        bisc_ref.split(mc0011, 10);
        bisc_ref.split(mc0011, 14);

        contraction2<2, 2, 0> contr(permutation<4>().permute(1, 2));

        gen_bto_contract2_bis<2, 2, 0> op(contr, bisa, bisb);

        if(!op.get_bis().equals(bisc_ref)) {
            fail_test(testname, __FILE__, __LINE__,
                "Invalid output block index space.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

