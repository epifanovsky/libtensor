#include <cstdlib>
#include <iostream>
#include <mkl.h>
#include <mpi.h>
#include <libutil/timings/timer.h>
#include <libtensor/libtensor.h>
#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/ctf_block_tensor/ctf_btod_set.h>
#include <libtensor/ctf_block_tensor/ctf_btod_contract2.h>
#include <libtensor/expr/ctf_btensor/ctf_btensor.h>

using namespace libtensor;

void warmup() {

    double a[128*128], b[128*128], c[128*128];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128, 128, 128,
        1.0, a, 128, b, 128, 1.0, c, 128);
}


int run_bench(size_t n) {

    std::cout << "run_bench(" << n << ")" << std::endl;

    bispace<1> si(n);
    bispace<4> sijkl(si&si&si&si);

    ctf::init();

    ctf_btensor<4> A(sijkl), C(sijkl);
    ctf_btod_set<4>(0.55).perform(A);

    contraction2<2, 2, 2> contr;
    contr.contract(1, 1);
    contr.contract(3, 3);

    libutil::timer tim;
    tim.start();
    ctf_btod_contract2<2, 2, 2>(contr, A, A).perform(C);
    tim.stop();
    std::cout << "contract_ctf_symm_bench: " << tim.duration() << std::endl;

    ctf::exit();

    std::cout << "SUCCESS" << std::endl;
    return 0;
}


int main(int argc, char **argv) {

    int nproc, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if(argc != 2) {
        if(myid == 0) std::cout << "Use: \"contract_ctf_symm_bench N\", "
                     "where N is matrix size" << std::endl;
        return -1;
    }

    warmup();

    int n = atoi(argv[1]);
    int err = run_bench(n);
    MPI_Finalize();
    return err;
}

