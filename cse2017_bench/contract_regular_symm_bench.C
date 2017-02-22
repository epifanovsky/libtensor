#include <cstdlib>
#include <iostream>
#include <mkl.h>
#include <libutil/timings/timer.h>
#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/libtensor.h>
#include <libtensor/core/batching_policy_base.h>
#include <libtensor/block_tensor/btod_set.h>
#include <libtensor/block_tensor/btod_contract2.h>

using namespace libtensor;

void warmup() {

    double a[128*128], b[128*128], c[128*128];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128, 128, 128,
        1.0, a, 128, b, 128, 1.0, c, 128);
}


int run_bench(size_t n, unsigned buf_mb, unsigned nthr) {

    std::cout << "run_bench(" << n << ", " << buf_mb << ", " << nthr << ")"
        << std::endl;

    bispace<1> si(n);
    for(size_t i = 16; i < n; i+=16) si.split(i);
    bispace<4> sijkl(si&si&si&si);

    btensor<4> A(sijkl), C(sijkl);
    btod_set<4>(0.55).perform(A);

    contraction2<2, 2, 2> contr;
    contr.contract(1, 1);
    contr.contract(3, 3);

    libutil::thread_pool tp(nthr, nthr);
    tp.associate();

    size_t buf_blk = size_t(buf_mb) * 1024*1024 / (256*256*sizeof(double));
    batching_policy_base::set_batch_size(buf_blk);
    std::cout << "batching_policy_base::set_batch_size(" << buf_blk << ")"
        << std::endl;

    libutil::timer tim;
    tim.start();
    btod_contract2<2, 2, 2>(contr, A, A).perform(C);
    tim.stop();
    std::cout << "contract_regular_symm_bench: " << tim.duration() << std::endl;

    tp.dissociate();

    std::cout << "SUCCESS" << std::endl;
    return 0;
}


int main(int argc, char **argv) {

    if(argc != 4) {
        std::cout << "Use: \"contract_regular_symm_bench N B T\", "
                     "where N is matrix size, "
                     "B is memory buffer size (MB), "
                     "T is number of threads" << std::endl;
        return -1;
    }

    warmup();

    int n = atoi(argv[1]);
    int b = atoi(argv[2]);
    int t = atoi(argv[3]);
    return run_bench(n, b, t);
}

