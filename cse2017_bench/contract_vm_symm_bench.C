#include <cstdlib>
#include <iostream>
#include <mkl.h>
#include <libutil/timings/timer.h>
#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/libtensor.h>
#include <libtensor/core/batching_policy_base.h>
#include <libtensor/core/impl/allocator_impl.h>
#include <libtensor/core/impl/vm_allocator.h>
#include <libtensor/block_tensor/btod_set.h>
#include <libtensor/block_tensor/btod_contract2.h>

using namespace libtensor;

void warmup() {

    double a[128*128], b[128*128], c[128*128];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128, 128, 128,
        1.0, a, 128, b, 128, 1.0, c, 128);
}


struct contract_timings {
    static const char k_clazz[];
};
const char contract_timings::k_clazz[] = "contract2";


int run_bench(size_t n, unsigned mem_mb, unsigned nthr, const char *pfprefix) {

    std::cout << "run_bench(" << n << ", " << mem_mb << ", " << nthr
        << ", " << pfprefix << ")" << std::endl;

    size_t mem_dbl = size_t(mem_mb) * 1024*1024 / sizeof(double);
    size_t blkmin = 16, blkmax = 4096*4096;
    libtensor::allocator<double>::init(libtensor::vm_allocator<double>(),
        blkmin, blkmin, blkmax, mem_dbl, pfprefix);
    //  Use 1/4 of memory for contraction buffer
    size_t buf_blk = mem_dbl / 4 / (256*256);
    batching_policy_base::set_batch_size(buf_blk);
    std::cout << "batching_policy_base::set_batch_size(" << buf_blk << ")"
        << std::endl;

    libutil::thread_pool tp(nthr, nthr);
    tp.associate();

    bispace<1> si(n);
    for(size_t i = 16; i < n; i+=16) si.split(i);
    bispace<4> sijkl(si&si&si&si);

    btensor<4> A(sijkl), C(sijkl);
    btod_set<4>(0.55).perform(A);

    contraction2<2, 2, 2> contr;
    contr.contract(1, 1);
    contr.contract(3, 3);

    libutil::timer tim;
    tim.start();
    btod_contract2<2, 2, 2>(contr, A, A).perform(C);
    tim.stop();
    std::cout << "contract_vm_symm_bench: " << tim.duration() << std::endl;

    tp.dissociate();
    libtensor::allocator<double>::shutdown();

    std::cout << "SUCCESS" << std::endl;
    return 0;
}


int main(int argc, char **argv) {

    if(argc != 5) {
        std::cout << "Use: \"contract_vm_bench N m T S\", "
                     "where N is matrix size, "
                     "M is memory size (MB), "
                     "T is number of threads, "
                     "S is scratch file prefix" << std::endl;
        return -1;
    }

    warmup();

    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int t = atoi(argv[3]);
    return run_bench(n, m, t, argv[4]);
}

