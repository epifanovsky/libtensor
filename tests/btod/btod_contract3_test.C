#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/btod_contract3.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/symmetry/permutation_group.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/dense_tensor/tod_btconv.h>
#include <libtensor/dense_tensor/tod_contract2.h>
#include "../compare_ref.h"
#include "btod_contract3_test.h"

namespace libtensor {


void btod_contract3_test::perform() throw(libtest::test_exception) {

    allocator<double>::init(16, 16, 16777216, 16777216);

    try {

        test_contr_1();
        //test_contr_2();
        test_contr_3a();
        test_contr_3b();
        test_contr_4();
        test_contr_5();
        test_contr_6();

        //  Tests for the batching mechanism

        test_batch_1();
        test_batch_2();

    } catch(...) {
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();
}


void btod_contract3_test::test_contr_1() {

    //
    //  d_{ij} = a_{ip} b_{pq} c_{jq}
    //  All dimensions are identical, no symmetry
    //

    static const char *testname = "btod_contract3_test::test_contr_1()";

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i1, i2;
        i2[0] = 10; i2[1] = 10;
        dimensions<2> dims(index_range<2>(i1, i2));
        block_index_space<2> bisa(dims);
        mask<2> m1;
        m1[0] = true; m1[1] = true;
        bisa.split(m1, 3);
        bisa.split(m1, 5);

        block_index_space<2> bisb(bisa), bisc(bisa), bisd(bisa);

        block_tensor<2, double, allocator_t> bta(bisa), btb(bisb), btc(bisc),
            btd(bisd);

        //  Load random data for input

        btod_random<2>().perform(bta);
        btod_random<2>().perform(btb);
        btod_random<2>().perform(btc);
        bta.set_immutable();
        btb.set_immutable();
        btc.set_immutable();

        //  Run contraction

        contraction2<1, 1, 1> contr1;
        contr1.contract(1, 0);
        contraction2<1, 1, 1> contr2;
        contr2.contract(1, 1);

        btod_contract3<1, 0, 1, 1, 1>(contr1, contr2, bta, btb, btc).
            perform(btd);

        //  Convert block tensors to regular tensors

        dense_tensor<2, double, allocator_t> ta(dims), tb(dims), tab(dims),
            tc(dims), td(dims), td_ref(dims);
        tod_btconv<2>(bta).perform(ta);
        tod_btconv<2>(btb).perform(tb);
        tod_btconv<2>(btc).perform(tc);
        tod_btconv<2>(btd).perform(td);

        //  Compute reference tensor

        tod_contract2<1, 1, 1>(contr1, ta, tb).perform(true, tab);
        tod_contract2<1, 1, 1>(contr2, tab, tc).perform(true, td_ref);

        //  Compare against reference

        compare_ref<2>::compare(testname, td, td_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract3_test::test_contr_2() {

    //
    //  d_{ijk} = a_{ip} b_{kqp} c_{qj}
    //  All dimensions are identical, no symmetry
    //

    static const char *testname = "btod_contract3_test::test_contr_2()";

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i2;
        i2[0] = 10; i2[1] = 10;
        dimensions<2> dims2(index_range<2>(index<2>(), i2));
        index<3> i3;
        i3[0] = 10; i3[1] = 10; i3[2] = 10;
        dimensions<3> dims3(index_range<3>(index<3>(), i3));

        block_index_space<2> bisa(dims2);
        mask<2> m11;
        m11[0] = true; m11[1] = true;
        bisa.split(m11, 3);
        bisa.split(m11, 5);

        block_index_space<3> bisb(dims3);
        mask<3> m111;
        m111[0] = true; m111[1] = true;
        bisb.split(m111, 3);
        bisb.split(m111, 5);

        block_index_space<2> bisc(bisa);
        block_index_space<3> bisd(bisb);

        block_tensor<2, double, allocator_t> bta(bisa), btc(bisc);
        block_tensor<3, double, allocator_t> btb(bisb), btd(bisd);

        //  Load random data for input

        btod_random<2>().perform(bta);
        btod_random<3>().perform(btb);
        btod_random<2>().perform(btc);
        bta.set_immutable();
        btb.set_immutable();
        btc.set_immutable();

        //  Run contraction

        contraction2<1, 2, 1> contr1;
        contr1.contract(1, 2);
        contraction2<2, 1, 1> contr2(permutation<3>().permute(1, 2));
        contr2.contract(2, 0);

        btod_contract3<1, 1, 1, 1, 1>(contr1, contr2, bta, btb, btc).
            perform(btd);

        //  Convert block tensors to regular tensors

        dense_tensor<2, double, allocator_t> ta(dims2), tc(dims2);
        dense_tensor<3, double, allocator_t> tb(dims3), tab(dims3), td(dims3),
            td_ref(dims3);
        tod_btconv<2>(bta).perform(ta);
        tod_btconv<3>(btb).perform(tb);
        tod_btconv<2>(btc).perform(tc);
        tod_btconv<3>(btd).perform(td);

        //  Compute reference tensor

        tod_contract2<1, 2, 1>(contr1, ta, tb).perform(true, tab);
        tod_contract2<2, 1, 1>(contr2, tab, tc).perform(true, td_ref);

        //  Compare against reference

        compare_ref<3>::compare(testname, td, td_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract3_test::test_contr_3a() {

    //
    //  c_{ijkl} = a_{kpr} a_{lqr} b_{ijpq}
    //  [k,l,p,q] = 9, [ij] = 5, [r] = 11
    //

    static const char *testname = "btod_contract3_test::test_contr_3a()";

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

        block_tensor<3, double, allocator_t> bta(bisa);
        block_tensor<4, double, allocator_t> btb(bisb), btc(bisc);

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

        btod_contract3<2, 0, 2, 1, 2>(contr1, contr2, bta, bta, btb).
            perform(btc);

        //  Convert block tensors to regular tensors

        dense_tensor<3, double, allocator_t> ta(dimsa);
        dense_tensor<4, double, allocator_t> tb(dimsb), ti(dimsi), tc(dimsc),
            tc_ref(dimsc);
        tod_btconv<3>(bta).perform(ta);
        tod_btconv<4>(btb).perform(tb);
        tod_btconv<4>(btc).perform(tc);

        //  Compute reference tensor

        tod_contract2<2, 2, 1>(contr1, ta, ta).perform(true, ti);
        tod_contract2<2, 2, 2>(contr2, ti, tb).perform(true, tc_ref);

        //  Compare against reference

        compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract3_test::test_contr_3b() {

    //
    //  c_{ijkl} = a_{kpr} a_{lqr} b_{ijpq}
    //  [k,l,p,q] = 9, [ij] = 5, [r] = 11
    //

    static const char *testname = "btod_contract3_test::test_contr_3b()";

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

        block_tensor<3, double, allocator_t> bta(bisa);
        block_tensor<4, double, allocator_t> btb(bisb), btc(bisc);

        dense_tensor<3, double, allocator_t> ta(dimsa);
        dense_tensor<4, double, allocator_t> tb(dimsb), ti(dimsi), tc(dimsc),
            tc_ref(dimsc);

        //  Load random data for input

        btod_random<3>().perform(bta);
        btod_random<4>().perform(btb);
        btod_random<4>().perform(btc);
        bta.set_immutable();
        btb.set_immutable();

        //  Convert block tensors to regular tensors

        tod_btconv<3>(bta).perform(ta);
        tod_btconv<4>(btb).perform(tb);
        tod_btconv<4>(btc).perform(tc_ref);

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

        btod_contract3<2, 0, 2, 1, 2>(contr1, contr2, bta, bta, btb).
            perform(btc, -0.7);

        tod_btconv<4>(btc).perform(tc);

        //  Compute reference tensor

        tod_contract2<2, 2, 1>(contr1, ta, ta).perform(true, ti);
        tod_contract2<2, 2, 2>(contr2, ti, tb, -0.7).perform(false, tc_ref);

        //  Compare against reference

        compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract3_test::test_contr_4() {

    //
    //  c_{ijkl} = a1_{kpr} a2_{lqr} b_{ijpq}
    //  [k,l,p,q] = 9, [ij] = 5, [r] = 11
    //

    static const char *testname = "btod_contract3_test::test_contr_4()";

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

        block_tensor<3, double, allocator_t> bta1(bisa), bta2(bisa);
        block_tensor<4, double, allocator_t> btb(bisb), btc(bisc);

        //  Load random data for input

        btod_random<3>().perform(bta1);
        btod_random<3>().perform(bta2);
        btod_random<4>().perform(btb);
        bta1.set_immutable();
        bta2.set_immutable();
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

        btod_contract3<2, 0, 2, 1, 2>(contr1, contr2, bta1, bta2, btb).
            perform(btc);

        //  Convert block tensors to regular tensors

        dense_tensor<3, double, allocator_t> ta1(dimsa), ta2(dimsa);
        dense_tensor<4, double, allocator_t> tb(dimsb), ti(dimsi), tc(dimsc),
            tc_ref(dimsc);
        tod_btconv<3>(bta1).perform(ta1);
        tod_btconv<3>(bta2).perform(ta2);
        tod_btconv<4>(btb).perform(tb);
        tod_btconv<4>(btc).perform(tc);

        //  Compute reference tensor

        tod_contract2<2, 2, 1>(contr1, ta1, ta2).perform(true, ti);
        tod_contract2<2, 2, 2>(contr2, ti, tb).perform(true, tc_ref);

        //  Compare against reference

        compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract3_test::test_contr_5() {

    //
    //  c_{ijkl} = a_{kpr} a_{lqr} b_{ijpq}
    //  [k,l,p,q] = 9, [ij] = 5, [r] = 11
    //  Permutational antisymmetry between (i,j) and (p,q) in b_{ijpq}
    //  Permutation symmetry between (k,p) in a_{kpr}
    //

    static const char *testname = "btod_contract3_test::test_contr_5()";

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

        block_tensor<3, double, allocator_t> bta(bisa);
        block_tensor<4, double, allocator_t> btb(bisb), btc(bisc);

        //  Set symmetry

        se_perm<3, double> seperma1(permutation<3>().permute(0, 1),
            scalar_transf<double>(1.0));
        se_perm<4, double> sepermb1(permutation<4>().permute(0, 1),
            scalar_transf<double>(-1.0));
        se_perm<4, double> sepermb2(permutation<4>().permute(2, 3),
            scalar_transf<double>(-1.0));

        {
            block_tensor_ctrl<3, double> ca(bta);
            ca.req_symmetry().insert(seperma1);

            block_tensor_ctrl<4, double> cb(btb);
            cb.req_symmetry().insert(sepermb1);
            cb.req_symmetry().insert(sepermb2);
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

        btod_contract3<2, 0, 2, 1, 2>(contr1, contr2, bta, bta, btb).
            perform(btc);

        //  Convert block tensors to regular tensors

        dense_tensor<3, double, allocator_t> ta(dimsa);
        dense_tensor<4, double, allocator_t> tb(dimsb), ti(dimsi), tc(dimsc),
            tc_ref(dimsc);
        tod_btconv<3>(bta).perform(ta);
        tod_btconv<4>(btb).perform(tb);
        tod_btconv<4>(btc).perform(tc);

        //  Compute reference tensor

        tod_contract2<2, 2, 1>(contr1, ta, ta).perform(true, ti);
        tod_contract2<2, 2, 2>(contr2, ti, tb).perform(true, tc_ref);

        //  Compare against reference

        compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract3_test::test_contr_6() {

    //
    //  c_{ijkl} = a_{kpr} a_{lqr} b_{ijpq}
    //  [klij] = 5, [pq] = 9, [r] = 11
    //  Permutational antisymmetry between (i,j) and (p,q) in b_{ijpq}
    //

    static const char *testname = "btod_contract3_test::test_contr_6()";

    typedef std_allocator<double> allocator_t;

    try {

        size_t ni = 5, nj = ni, nk = ni, nl = ni, np = 9, nq = np, nr = 11;

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
        mask<3> m100, m010, m001;
        m100[0] = true; m010[1] = true; m001[2] = true;
        bisa.split(m100, 2);
        bisa.split(m100, 3);
        bisa.split(m010, 3);
        bisa.split(m010, 7);
        bisa.split(m001, 5);

        block_index_space<4> bisb(dimsb);
        mask<4> m1100, m0011;
        m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;
        bisb.split(m1100, 2);
        bisb.split(m1100, 3);
        bisb.split(m0011, 3);
        bisb.split(m0011, 7);

        block_index_space<4> bisc(dimsc);
        mask<4> m1111;
        m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
        bisc.split(m1111, 2);
        bisc.split(m1111, 3);

        block_tensor<3, double, allocator_t> bta(bisa);
        block_tensor<4, double, allocator_t> btb(bisb), btc(bisc);

        //  Set symmetry

        se_perm<4, double> sepermb1(permutation<4>().permute(0, 1),
            scalar_transf<double>(-1.0));
        se_perm<4, double> sepermb2(permutation<4>().permute(2, 3),
            scalar_transf<double>(-1.0));

        {
            block_tensor_ctrl<4, double> cb(btb);
            cb.req_symmetry().insert(sepermb1);
            cb.req_symmetry().insert(sepermb2);
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

        btod_contract3<2, 0, 2, 1, 2>(contr1, contr2, bta, bta, btb).
            perform(btc);

        //  Convert block tensors to regular tensors

        dense_tensor<3, double, allocator_t> ta(dimsa);
        dense_tensor<4, double, allocator_t> tb(dimsb), ti(dimsi), tc(dimsc),
            tc_ref(dimsc);
        tod_btconv<3>(bta).perform(ta);
        tod_btconv<4>(btb).perform(tb);
        tod_btconv<4>(btc).perform(tc);

        //  Compute reference tensor

        tod_contract2<2, 2, 1>(contr1, ta, ta).perform(true, ti);
        tod_contract2<2, 2, 2>(contr2, ti, tb).perform(true, tc_ref);

        //  Compare against reference

        compare_ref<4>::compare(testname, tc, tc_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract3_test::test_batch_1() {

    //
    //  d_ijkl = a_i b_j c_kl
    //  All dimensions are identical, no symmetry
    //

    static const char *testname = "btod_contract3_test::test_batch_1()";

    typedef std_allocator<double> allocator_t;

    try {

        index<1> i1a, i1b;
        i1b[0] = 19;
        dimensions<1> dims1(index_range<1>(i1a, i1b));
        block_index_space<1> bis1(dims1);
        index<2> i2a, i2b;
        i2b[0] = 19; i2b[1] = 19;
        dimensions<2> dims2(index_range<2>(i2a, i2b));
        block_index_space<2> bis2(dims2);
        index<4> i4a, i4b;
        i4b[0] = 19; i4b[1] = 19; i4b[2] = 19; i4b[3] = 19;
        dimensions<4> dims4(index_range<4>(i4a, i4b));
        block_index_space<4> bis4(dims4);
        mask<1> m1;
        m1[0] = true;
        mask<2> m11;
        m11[0] = true; m11[1] = true;
        mask<4> m1111;
        m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
        for(size_t i = 1; i < 10; i++) {
            bis1.split(m1, 2 * i);
            bis2.split(m11, 2 * i);
            bis4.split(m1111, 2 * i);
        }

        block_tensor<1, double, allocator_t> bta(bis1), btb(bis1);
        block_tensor<2, double, allocator_t> btc(bis2);
        block_tensor<4, double, allocator_t> btd(bis4);

        //  Load random data for input

        btod_random<1>().perform(bta);
        btod_random<1>().perform(btb);
        btod_random<2>().perform(btc);
        bta.set_immutable();
        btb.set_immutable();
        btc.set_immutable();

        //  Run contraction

        contraction2<1, 1, 0> contr1;
        contraction2<2, 2, 0> contr2;

        btod_contract3<1, 1, 2, 0, 0>(contr1, contr2, bta, btb, btc).
            perform(btd);

        //  Convert block tensors to regular tensors

        dense_tensor<1, double, allocator_t> ta(dims1), tb(dims1);
        dense_tensor<2, double, allocator_t> ti(dims2), tc(dims2);
        dense_tensor<4, double, allocator_t> td(dims4), td_ref(dims4);
        tod_btconv<1>(bta).perform(ta);
        tod_btconv<1>(btb).perform(tb);
        tod_btconv<2>(btc).perform(tc);
        tod_btconv<4>(btd).perform(td);

        //  Compute reference tensor

        tod_contract2<1, 1, 0>(contr1, ta, tb).perform(true, ti);
        tod_contract2<2, 2, 0>(contr2, ti, tc).perform(true, td_ref);

        //  Compare against reference

        compare_ref<4>::compare(testname, td, td_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_contract3_test::test_batch_2() {

    //
    //  d_jl = a_ij b_kl c_ik
    //  All dimensions are identical, no symmetry
    //

    static const char *testname = "btod_contract3_test::test_batch_2()";

    typedef std_allocator<double> allocator_t;

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

        block_tensor<2, double, allocator_t> bta(bis2);
        block_tensor<2, double, allocator_t> btb(bis2);
        block_tensor<2, double, allocator_t> btc(bis2);
        block_tensor<2, double, allocator_t> btd(bis2);

        //  Load random data for input

        btod_random<2>().perform(bta);
        btod_random<2>().perform(btb);
        btod_random<2>().perform(btc);
        bta.set_immutable();
        btb.set_immutable();
        btc.set_immutable();

        //  Run contraction

        contraction2<2, 2, 0> contr1;
        contraction2<2, 0, 2> contr2;
        contr2.contract(0, 0);
        contr2.contract(2, 1);

        btod_contract3<2, 0, 0, 0, 2>(contr1, contr2, bta, btb, btc).
            perform(btd);

        //  Convert block tensors to regular tensors

        dense_tensor<2, double, allocator_t> ta(dims2), tb(dims2), tc(dims2);
        dense_tensor<4, double, allocator_t> ti(dims4);
        dense_tensor<2, double, allocator_t> td(dims2), td_ref(dims2);
        tod_btconv<2>(bta).perform(ta);
        tod_btconv<2>(btb).perform(tb);
        tod_btconv<2>(btc).perform(tc);
        tod_btconv<2>(btd).perform(td);

        //  Compute reference tensor

        tod_contract2<2, 2, 0>(contr1, ta, tb).perform(true, ti);
        tod_contract2<2, 0, 2>(contr2, ti, tc).perform(true, td_ref);

        //  Compare against reference

        compare_ref<2>::compare(testname, td, td_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

