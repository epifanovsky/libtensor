#include <mpi.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod_dotprod.h>
#include <libtensor/dense_tensor/tod_random.h>
#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/ctf_dense_tensor/ctf_dense_tensor.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_dotprod.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_distribute.h>
#include "../compare_ref.h"
#include "ctf_tod_dotprod_test.h"

namespace libtensor {


void ctf_tod_dotprod_test::perform() throw(libtest::test_exception) {

    int nproc, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ctf::get().init(MPI_COMM_WORLD, rank, nproc);

    try {

        test_1a();
        test_1b();
        test_2a();
        test_2b();
        test_3a();
        test_3b();

    } catch(...) {
        ctf::get().exit();
        throw;
    }

    ctf::get().exit();
}


void ctf_tod_dotprod_test::test_1a() {

    static const char testname[] = "ctf_tod_dotprod_test::test_1a()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa);
    dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb);
    ctf_dense_tensor<2, double> dta(dimsa), dtb(dimsb);

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);

    double d_ref = tod_dotprod<2>(ta, tb).calculate();
    double d = ctf_tod_dotprod<2>(dta, dtb).calculate();

    if(fabs(d - d_ref) > 1e-14 * fabs(d_ref)) {
        std::ostringstream ss;
        ss << "Result doesn't match reference: " << d << " (result), "
            << d_ref << " (reference), " << d - d_ref << " (diff)";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_dotprod_test::test_1b() {

    static const char testname[] = "ctf_tod_dotprod_test::test_1b()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa);
    dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb);
    ctf_dense_tensor<2, double> dta(dimsa), dtb(dimsb);

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);

    tensor_transf<2, double> tra(permutation<2>(), scalar_transf<double>(-0.5));
    tensor_transf<2, double> trb(permutation<2>(), scalar_transf<double>(-0.5));

    double d_ref = 0.25 * tod_dotprod<2>(ta, tb).calculate();
    double d = ctf_tod_dotprod<2>(dta, tra, dtb, trb).calculate();

    if(fabs(d - d_ref) > 1e-14 * fabs(d_ref)) {
        std::ostringstream ss;
        ss << "Result doesn't match reference: " << d << " (result), "
            << d_ref << " (reference), " << d - d_ref << " (diff)";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_dotprod_test::test_2a() {

    static const char testname[] = "ctf_tod_dotprod_test::test_2a()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 49;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa);
    permutation<2> perma, permb;
    perma.permute(0, 1);
    dimsb.permute(perma);
    dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb);
    ctf_dense_tensor<2, double> dta(dimsa), dtb(dimsb);

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);

    double d_ref = tod_dotprod<2>(ta, perma, tb, permb).calculate();
    double d = ctf_tod_dotprod<2>(dta, perma, dtb, permb).calculate();

    if(fabs(d - d_ref) > 1e-14 * fabs(d_ref)) {
        std::ostringstream ss;
        ss << "Result doesn't match reference: " << d << " (result), "
            << d_ref << " (reference), " << d - d_ref << " (diff)";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_dotprod_test::test_2b() {

    static const char testname[] = "ctf_tod_dotprod_test::test_2b()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa);
    permutation<2> perma, permb;
    perma.permute(0, 1);
    dimsb.permute(perma);
    dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb);
    ctf_dense_tensor<2, double> dta(dimsa), dtb(dimsb);

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);

    tensor_transf<2, double> tra(perma, scalar_transf<double>(-0.5));
    tensor_transf<2, double> trb(permb, scalar_transf<double>(1.0));

    double d_ref = tod_dotprod<2>(ta, tra, tb, trb).calculate();
    double d = ctf_tod_dotprod<2>(dta, tra, dtb, trb).calculate();

    if(fabs(d - d_ref) > 1e-14 * fabs(d_ref)) {
        std::ostringstream ss;
        ss << "Result doesn't match reference: " << d << " (result), "
            << d_ref << " (reference), " << d - d_ref << " (diff)";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_dotprod_test::test_3a() {

    static const char testname[] = "ctf_tod_dotprod_test::test_3a()";

    typedef std_allocator<double> allocator_t;

    try {

    index<3> i1, i2;
    i2[0] = 19; i2[1] = 9; i2[2] = 7;
    dimensions<3> dimsa(index_range<3>(i1, i2)), dimsb(dimsa);
    permutation<3> perma, permb;
    perma.permute(0, 1).permute(1, 2);
    dimsb.permute(perma);
    dense_tensor<3, double, allocator_t> ta(dimsa), tb(dimsb);
    ctf_dense_tensor<3, double> dta(dimsa), dtb(dimsb);

    tod_random<3>().perform(ta);
    tod_random<3>().perform(tb);

    ctf_tod_distribute<3>(ta).perform(dta);
    ctf_tod_distribute<3>(tb).perform(dtb);

    double d_ref = tod_dotprod<3>(ta, perma, tb, permb).calculate();
    double d = ctf_tod_dotprod<3>(dta, perma, dtb, permb).calculate();

    if(fabs(d - d_ref) > 1e-14 * fabs(d_ref)) {
        std::ostringstream ss;
        ss << "Result doesn't match reference: " << d << " (result), "
            << d_ref << " (reference), " << d - d_ref << " (diff)";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_dotprod_test::test_3b() {

    static const char testname[] = "ctf_tod_dotprod_test::test_3b()";

    typedef std_allocator<double> allocator_t;

    try {

    index<3> i1, i2;
    i2[0] = 19; i2[1] = 9; i2[2] = 7;
    dimensions<3> dimsa(index_range<3>(i1, i2)), dimsb(dimsa);
    permutation<3> perma, permb;
    perma.permute(0, 2).permute(1, 2);
    dimsb.permute(perma);
    dense_tensor<3, double, allocator_t> ta(dimsa), tb(dimsb), tb_ref(dimsb);
    ctf_dense_tensor<3, double> dta(dimsa), dtb(dimsb);

    tod_random<3>().perform(ta);
    tod_random<3>().perform(tb);

    ctf_tod_distribute<3>(ta).perform(dta);
    ctf_tod_distribute<3>(tb).perform(dtb);

    tensor_transf<3, double> tra(perma, scalar_transf<double>(-1.0));
    tensor_transf<3, double> trb(permb, scalar_transf<double>(1.0));

    double d_ref = tod_dotprod<3>(ta, tra, tb, trb).calculate();
    double d = ctf_tod_dotprod<3>(dta, tra, dtb, trb).calculate();

    if(fabs(d - d_ref) > 1e-14 * fabs(d_ref)) {
        std::ostringstream ss;
        ss << "Result doesn't match reference: " << d << " (result), "
            << d_ref << " (reference), " << d - d_ref << " (diff)";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

