#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/btod/scalar_transf_double.h>
#include <libtensor/block_tensor/btod/btod_contract3.h>
#include <libtensor/btod/btod_copy.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/symmetry/permutation_group.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/dense_tensor/tod_btconv.h>
#include <libtensor/dense_tensor/tod_contract2.h>
#include "../compare_ref.h"
#include "btod_contract3_test.h"

namespace libtensor {


void btod_contract3_test::perform() throw(libtest::test_exception) {

    allocator<double>::vmm().init(16, 16, 16777216, 16777216);

    try {

        test_contr_1();
        test_contr_2();

    } catch(...) {
        allocator<double>::vmm().shutdown();
        throw;
    }

    allocator<double>::vmm().shutdown();
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

        tod_contract2<1, 1, 1>(contr1, ta, tb).perform(true, 1.0, tab);
        tod_contract2<1, 1, 1>(contr2, tab, tc).perform(true, 1.0, td_ref);

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

        tod_contract2<1, 2, 1>(contr1, ta, tb).perform(true, 1.0, tab);
        tod_contract2<2, 1, 1>(contr2, tab, tc).perform(true, 1.0, td_ref);

        //  Compare against reference

        compare_ref<3>::compare(testname, td, td_ref, 1e-13);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

