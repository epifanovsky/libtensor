#include <sstream>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/cuda_block_tensor/cuda_block_tensor.h>
#include <libtensor/block_tensor/btod_contract2.h>
#include <libtensor/cuda_block_tensor/cuda_btod_contract2.h>
#include <libtensor/cuda_block_tensor/cuda_btod_copy_d2h.h>
#include <libtensor/cuda_block_tensor/cuda_btod_copy_h2d.h>
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
#include <libtensor/cuda_dense_tensor/cuda_tod_contract2.h>
#include <libtensor/cuda_dense_tensor/cuda_tod_set.h>
#include "../compare_ref.h"
#include "cuda_btod_contract2_test.h"

namespace libtensor {

void cuda_btod_contract2_test::perform() throw(libtest::test_exception) {

    allocator<double>::init(16, 16, 16777216, 16777216);

    try {

    test_bis_1();
//    test_bis_2();
//    test_bis_3();
//    test_bis_4();
//    test_bis_5();
//
//    //  Tests for zero block structure
//    test_zeroblk_1();
//    test_zeroblk_2();
//    test_zeroblk_3();
//    test_zeroblk_4();
//    test_zeroblk_5();
//    test_zeroblk_6();
//
    //  Tests for contractions

    test_contr_1();
//    test_contr_2();
//    test_contr_3();
//    test_contr_4();
//    test_contr_5();
//    test_contr_6();
//    test_contr_7();
//    test_contr_8();
//    test_contr_9();
//    test_contr_10();
//    test_contr_11();
//    test_contr_12();
//    test_contr_13();
    test_contr_14(0.0);
    test_contr_14(1.0);
    test_contr_14(-2.2);

    test_contr_18(0.0);
    test_contr_18(-1.5);
//    test_contr_19();
//    test_contr_20a();
//    test_contr_20b();
//    test_contr_21();
//    test_contr_22();
//    test_contr_23();
//    test_contr_24();
//    test_contr_25();
//    test_contr_26();
//
//    //  Tests for the contraction of a block tensor with itself
//
    test_self_1();
//    test_self_2();
//    test_self_3();

    //  Tests for the batching mechanism

    test_batch_1();
//    test_batch_2(); // These two tests take
//    test_batch_3(); // a long time to run

    } catch(...) {
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();
}


void cuda_btod_contract2_test::test_bis_1() throw(libtest::test_exception) {

    //
    //  c_ijkl = a_ijkp b_lp
    //  [ij] = 5  (no splits)
    //  [kl] = 10 (no splits)
    //  [p]  = 4  (no splits)
    //
    //  Expected block index space:
    //  [ijkl] have correct dimensions, no splits
    //

    static const char *testname = "cuda_btod_contract2_test::test_bis_1()";



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

        cuda_block_tensor<4, double, cuda_allocator_t> bta(bisa);
        cuda_block_tensor<2, double, cuda_allocator_t> btb(bisb);
        contraction2<3, 1, 1> contr;
        contr.contract(3, 1);

        cuda_btod_contract2<3, 1, 1> op(contr, bta, btb);

        if(!op.get_bis().equals(bisc_ref)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Incorrect output block index space.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}



void cuda_btod_contract2_test::test_contr_1() throw(libtest::test_exception) {

    //
    //  c_ijkl = a_ijpq b_klpq
    //  All dimensions are identical, no symmetry
    //

    static const char *testname = "cuda_btod_contract2_test::test_contr_1()";

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

        block_tensor<4, double, allocator_t> bta(bisa), btb(bisb), btc(bisc), btc_ref(bisc);
        cuda_block_tensor<4, double, cuda_allocator_t> bta_d(bisa), btb_d(bisb), btc_d(bisc);

        //  Load random data for input

        btod_random<4> rand;
        rand.perform(bta);
        rand.perform(btb);
        bta.set_immutable();
        btb.set_immutable();

        //  Run reference contraction

        contraction2<2, 2, 2> contr;
        contr.contract(2, 2);
        contr.contract(3, 3);

        btod_contract2<2, 2, 2> op(contr, bta, btb);
        op.perform(btc_ref);
        btc_ref.set_immutable();

        //  Copy from host to device memory

        cuda_btod_copy_h2d<4>(bta).perform(bta_d);
        cuda_btod_copy_h2d<4>(btb).perform(btb_d);

        //Run contraction on GPU
        cuda_btod_contract2<2, 2, 2> op_d(contr, bta_d, btb_d);
        op_d.perform(btc_d);

        //  Copy back from device to host memory

        cuda_btod_copy_d2h<4>(btc_d).perform(btc);

        //  Compare against reference

        compare_ref<4>::compare(testname, btc, btc_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void cuda_btod_contract2_test::test_contr_14(double c)
throw(libtest::test_exception) {

    //
    //  c_ijkl = a_ijmn b_klmn
    //  Dimensions [ijlkmn] = 15 (three blocks), no symmetry
    //  bis of the operation and the output tensor are not equal
    //

    std::ostringstream ss;
    ss << "cuda_btod_contract2_test::test_contr_14(" << c << ")";
    std::string tn = ss.str();

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
        block_tensor<4, double, allocator_t> btc_ref(bisc);

        cuda_block_tensor<4, double, cuda_allocator_t> bta_d(bisa), btb_d(bisb), btc_d(bisc);

        //  Load random data for input

        btod_random<4>().perform(bta);
        btod_random<4>().perform(btb);
        btod_random<4>().perform(btc_ref);
        bta.set_immutable();
        btb.set_immutable();

        //  Copy from host to device memory

		cuda_btod_copy_h2d<4>(bta).perform(bta_d);
		cuda_btod_copy_h2d<4>(btb).perform(btb_d);
		cuda_btod_copy_h2d<4>(btc_ref).perform(btc_d);

		//  Run reference contraction

		contraction2<2, 2, 2> contr;
		contr.contract(2, 2);
		contr.contract(3, 3);

		if(c == 0.0) btod_contract2<2, 2, 2>(contr, bta, btb).perform(btc_ref);
		else btod_contract2<2, 2, 2>(contr, bta, btb).perform(btc_ref, c);
		btc_ref.set_immutable();

		//Run contraction on GPU

		if(c == 0.0) cuda_btod_contract2<2, 2, 2>(contr, bta_d, btb_d).perform(btc_d);
		else cuda_btod_contract2<2, 2, 2>(contr, bta_d, btb_d).perform(btc_d, c);

		//  Copy back from device to host memory

		cuda_btod_copy_d2h<4>(btc_d).perform(btc);

		//  Compare against reference

		compare_ref<4>::compare(tn.c_str(), btc, btc_ref, 1e-13);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}

void cuda_btod_contract2_test::test_contr_18(double c)
throw(libtest::test_exception) {

    //
    //  c_ij = a_jkab b_iakb
    //  Dimensions [ijk] = 13 (two blocks), [ab] = 7 (three blocks),
    //  no symmetry,
    //  all blocks non-zero
    //

    std::ostringstream ss;
    ss << "cuda_btod_contract2_test::test_contr_18(" << c << ")";
    std::string tn = ss.str();


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
        block_tensor<2, double, allocator_t> btc_ref(bis_ii);

        cuda_block_tensor<4, double, cuda_allocator_t> bta_d(bis_iiaa), btb_d(bis_iaia);

        cuda_block_tensor<2, double, cuda_allocator_t> btc_d(bis_ii);

        //  Load random data for input

        btod_random<4>().perform(bta);
        btod_random<4>().perform(btb);
        if(c != 0.0) btod_random<2>().perform(btc);
        bta.set_immutable();
        btb.set_immutable();

        //  Copy from host to device memory

		cuda_btod_copy_h2d<4>(bta).perform(bta_d);
		cuda_btod_copy_h2d<4>(btb).perform(btb_d);


        //  Compute the reference

        contraction2<1, 1, 3> contr(permutation<2>().permute(0, 1));
        contr.contract(1, 2);
        contr.contract(2, 1);
        contr.contract(3, 3);

        if(c == 0.0) btod_contract2<1, 1, 3>(contr, bta, btb).perform(btc_ref);
        else btod_contract2<1, 1, 3>(contr, bta, btb).perform(btc_ref, c);
        btc_ref.set_immutable();

        //Run contraction on GPU

        if(c == 0.0) cuda_btod_contract2<1, 1, 3>(contr, bta_d, btb_d).perform(btc_d);
        else cuda_btod_contract2<1, 1, 3>(contr, bta_d, btb_d).perform(btc_d, c);

        //  Copy back from device to host memory

		cuda_btod_copy_d2h<2>(btc_d).perform(btc);

		//  Compare against reference

		compare_ref<2>::compare(tn.c_str(), btc, btc_ref, 1e-13);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}



/** \test Tests \f$ c_{ijab} = a_{ia} a_{jb} \f$, expected perm symmetry
\f$ c_{ijab} = c_{jiba} \f$.
 **/

void cuda_btod_contract2_test::test_self_1() {

    static const char *testname = "cuda_btod_contract2_test::test_self_1()";


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
        block_tensor<4, double, allocator_t> btc(bis_ijab), btc_ref(bis_ijab);

        cuda_block_tensor<2, double, cuda_allocator_t> bta_d(bis_ia);
        cuda_block_tensor<4, double, cuda_allocator_t> btc_d(bis_ijab);

        //  Load random data for input

        btod_random<2>().perform(bta);
        bta.set_immutable();

        //  Copy from host to device memory

        cuda_btod_copy_h2d<2>(bta).perform(bta_d);


        //  Compute the reference

        contraction2<2, 2, 0> contr(permutation<4>().permute(1, 2));

        btod_contract2<2, 2, 0>(contr, bta, bta).perform(btc_ref);
        btc_ref.set_immutable();


        //  Run contraction

        cuda_btod_contract2<2, 2, 0>(contr, bta_d, bta_d).perform(btc_d);

        //  Copy back from device to host memory

		cuda_btod_copy_d2h<4>(btc_d).perform(btc);

		 //  Compute reference symmetry and tensor

		symmetry<4, double> symc(bis_ijab), symc_ref(bis_ijab);
		scalar_transf<double> tr0;
		symc_ref.insert(se_perm<4, double>(permutation<4>().
				permute(0, 1).permute(2, 3), tr0));
		{
			block_tensor_ctrl<4, double> cc(btc);
			so_copy<4, double>(cc.req_const_symmetry()).perform(symc);
		}

		//  Compare against reference

		compare_ref<4>::compare(testname, symc, symc_ref);
		compare_ref<4>::compare(testname, btc, btc_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void cuda_btod_contract2_test::test_batch_1() {

    //
    //  c_ijkl = a_ij b_kl
    //  All dimensions are identical, no symmetry
    //

    static const char *testname = "cuda_btod_contract2_test::test_batch_1()";


    try {

        index<2> i2a, i2b;
        i2b[0] = 19; i2b[1] = 19;
        dimensions<2> dims2(index_range<2>(i2a, i2b));
        block_index_space<2> bis2(dims2);
        index<4> i4a, i4b;
        i4b[0] = 19; i4b[1] = 19; i4b[2] = 19; i4b[3] = 19;
        dimensions<4> dims4(index_range<4>(i4a, i4b));
        block_index_space<4> bis4(dims4);
        mask<2> m11;
        m11[0] = true; m11[1] = true;
        mask<4> m1111;
        m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
        for(size_t i = 1; i < 10; i++) {
            bis2.split(m11, 2 * i);
            bis4.split(m1111, 2 * i);
        }

        block_tensor<2, double, allocator_t> bta(bis2), btb(bis2);
        block_tensor<4, double, allocator_t> btc(bis4), btc_ref(bis4);

        cuda_block_tensor<2, double, cuda_allocator_t> bta_d(bis2), btb_d(bis2);
        cuda_block_tensor<4, double, cuda_allocator_t> btc_d(bis4);

        //  Load random data for input

        btod_random<2>().perform(bta);
        btod_random<2>().perform(btb);
        bta.set_immutable();
        btb.set_immutable();

		//  Copy from host to device memory

		cuda_btod_copy_h2d<2>(bta).perform(bta_d);
		cuda_btod_copy_h2d<2>(btb).perform(btb_d);


		//  Compute the reference

        contraction2<2, 2, 0> contr;

        btod_contract2<2, 2, 0>(contr, bta, btb).perform(btc_ref);

        btc_ref.set_immutable();

        //  Run contraction

        cuda_btod_contract2<2, 2, 0>(contr, bta_d, btb_d).perform(btc_d);


        //  Copy back from device to host memory

		cuda_btod_copy_d2h<4>(btc_d).perform(btc);

        //  Compare against reference

        compare_ref<4>::compare(testname, btc, btc_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

} // namespace libtensor

