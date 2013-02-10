#include <mpi.h>
#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/ctf_dense_tensor/ctf_dense_tensor.h>
#include <libtensor/ctf_dense_tensor/ctf_dense_tensor_ctrl.h>
#include "ctf_dense_tensor_test.h"

namespace libtensor {


void ctf_dense_tensor_test::perform() throw(libtest::test_exception) {

    int nproc, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ctf::get().init(MPI_COMM_WORLD, rank, nproc);

    test_1();

    ctf::get().exit();
}


void ctf_dense_tensor_test::test_1() {

    static const char testname[] = "ctf_dense_tensor_test::test_1()";

    try {

    index<1> i1, i2;
    i2[0] = 2;
    dimensions<1> dims(index_range<1>(i1, i2));
    ctf_dense_tensor<1, double> t1(dims);

    if(t1.is_immutable()) {
        fail_test(testname, __FILE__, __LINE__,
            "New tensor must be mutable (t1)");
    }

    if(t1.get_dims()[0] != 3) {
        fail_test(testname, __FILE__, __LINE__,
            "Incorrect tensor dimension 0 (t1)");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

