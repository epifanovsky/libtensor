#include <mpi.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod_random.h>
#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/ctf_dense_tensor/ctf_dense_tensor.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_collect.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_contract2.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_distribute.h>
#include "../compare_ref.h"
#include "ctf_tod_contract2_test.h"

namespace libtensor {


void ctf_tod_contract2_test::perform() throw(libtest::test_exception) {

    int nproc, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ctf::get().init(MPI_COMM_WORLD, rank, nproc);

    try {

        test_1a();
        test_1b();
        test_2a();
        test_2b();

    } catch(...) {
        ctf::get().exit();
        throw;
    }

    ctf::get().exit();
}


void ctf_tod_contract2_test::test_1a() {

    static const char testname[] = "ctf_tod_contract2_test::test_1a()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa), dimsc(dimsa);
    dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb), tc(dimsc),
        tc_ref(dimsc);
    ctf_dense_tensor<2, double> dta(dimsa), dtb(dimsb), dtc(dimsc);

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);
    tod_random<2>().perform(tc);
    tod_copy<2>(tc).perform(true, tc_ref);

    contraction2<1, 1, 1> contr;
    contr.contract(1, 0);
    tod_contract2<1, 1, 1>(contr, ta, tb).perform(true, tc_ref);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);
    ctf_tod_distribute<2>(tc).perform(dtc);

    ctf_tod_contract2<1, 1, 1>(contr, dta, dtb).perform(true, dtc);
    ctf_tod_collect<2>(dtc).perform(tc);

    compare_ref<2>::compare(testname, tc, tc_ref, 5e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_contract2_test::test_1b() {

    static const char testname[] = "ctf_tod_contract2_test::test_1b()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa), dimsc(dimsa);
    dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb), tc(dimsc),
        tc_ref(dimsc);
    ctf_dense_tensor<2, double> dta(dimsa), dtb(dimsb), dtc(dimsc);

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);
    tod_random<2>().perform(tc);
    tod_copy<2>(tc).perform(true, tc_ref);

    contraction2<1, 1, 1> contr;
    contr.contract(1, 0);
    tod_contract2<1, 1, 1>(contr, ta, tb, -0.5).perform(false, tc_ref);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);
    ctf_tod_distribute<2>(tc).perform(dtc);

    ctf_tod_contract2<1, 1, 1>(contr, dta, dtb, -0.5).perform(false, dtc);
    ctf_tod_collect<2>(dtc).perform(tc);

    compare_ref<2>::compare(testname, tc, tc_ref, 5e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_contract2_test::test_2a() {

    static const char testname[] = "ctf_tod_contract2_test::test_2a()";

    typedef std_allocator<double> allocator_t;

    try {

    index<3> ia1, ia2;
    ia2[0] = 29; ia2[1] = 29; ia2[2] = 19;
    dimensions<3> dimsa(index_range<3>(ia1, ia2));
    index<3> ib1, ib2;
    ib2[0] = 19; ib2[1] = 29; ib2[2] = 29;
    dimensions<3> dimsb(index_range<3>(ib1, ib2));
    index<4> ic1, ic2;
    ic2[0] = 29; ic2[1] = 29; ic2[2] = 29; ic2[3] = 29;
    dimensions<4> dimsc(index_range<4>(ic1, ic2));
    dense_tensor<3, double, allocator_t> ta(dimsa);
    dense_tensor<3, double, allocator_t> tb(dimsb);
    dense_tensor<4, double, allocator_t> tc(dimsc), tc_ref(dimsc);
    ctf_dense_tensor<3, double> dta(dimsa);
    ctf_dense_tensor<3, double> dtb(dimsb);
    ctf_dense_tensor<4, double> dtc(dimsc);

    tod_random<3>().perform(ta);
    tod_random<3>().perform(tb);
    tod_random<4>().perform(tc);

    // c(ijkl) = a(jkp) b(pil)
    contraction2<2, 2, 1> contr(permutation<4>().permute(1, 2).permute(0, 1));
    contr.contract(2, 0);
    tod_contract2<2, 2, 1>(contr, ta, tb, 0.1).perform(true, tc_ref);

    ctf_tod_distribute<3>(ta).perform(dta);
    ctf_tod_distribute<3>(tb).perform(dtb);
    ctf_tod_distribute<4>(tc).perform(dtc);

    ctf_tod_contract2<2, 2, 1>(contr, dta, dtb, 0.1).perform(true, dtc);
    ctf_tod_collect<4>(dtc).perform(tc);

    compare_ref<4>::compare(testname, tc, tc_ref, 5e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_contract2_test::test_2b() {

    static const char testname[] = "ctf_tod_contract2_test::test_2b()";

    typedef std_allocator<double> allocator_t;

    try {

    index<3> ia1, ia2;
    ia2[0] = 29; ia2[1] = 29; ia2[2] = 19;
    dimensions<3> dimsa(index_range<3>(ia1, ia2));
    index<3> ib1, ib2;
    ib2[0] = 19; ib2[1] = 29; ib2[2] = 29;
    dimensions<3> dimsb(index_range<3>(ib1, ib2));
    index<4> ic1, ic2;
    ic2[0] = 29; ic2[1] = 29; ic2[2] = 29; ic2[3] = 29;
    dimensions<4> dimsc(index_range<4>(ic1, ic2));
    dense_tensor<3, double, allocator_t> ta(dimsa);
    dense_tensor<3, double, allocator_t> tb(dimsb);
    dense_tensor<4, double, allocator_t> tc(dimsc), tc_ref(dimsc);
    ctf_dense_tensor<3, double> dta(dimsa);
    ctf_dense_tensor<3, double> dtb(dimsb);
    ctf_dense_tensor<4, double> dtc(dimsc);

    tod_random<3>().perform(ta);
    tod_random<3>().perform(tb);
    tod_random<4>().perform(tc);
    tod_copy<4>(tc).perform(true, tc_ref);

    // c(ijkl) = a(jkp) b(pil)
    contraction2<2, 2, 1> contr(permutation<4>().permute(1, 2).permute(0, 1));
    contr.contract(2, 0);
    tod_contract2<2, 2, 1>(contr, ta, tb, -0.1).perform(false, tc_ref);

    ctf_tod_distribute<3>(ta).perform(dta);
    ctf_tod_distribute<3>(tb).perform(dtb);
    ctf_tod_distribute<4>(tc).perform(dtc);

    ctf_tod_contract2<2, 2, 1>(contr, dta, dtb, -0.1).perform(false, dtc);
    ctf_tod_collect<4>(dtc).perform(tc);

    compare_ref<4>::compare(testname, tc, tc_ref, 5e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

