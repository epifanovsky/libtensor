#include <cstdlib>
#include <iostream>
#include <mkl.h>
#include <libutil/timings/timer.h>
#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/libtensor.h>
#include <libtensor/core/batching_policy_base.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/se_part.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/block_tensor/btod_set.h>
#include <libtensor/block_tensor/btod_contract2.h>

using namespace libtensor;

void warmup() {

    double a[128*128], b[128*128], c[128*128];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128, 128, 128,
        1.0, a, 128, b, 128, 1.0, c, 128);
}


void set_symmetry(btensor<4> &bt) {

    using libtensor::index;

    symmetry<4, double> sym(bt.get_bis());
    se_perm<4, double> se1(permutation<4>().permute(0,2).permute(1,3),
        scalar_transf<double>(1.0));
    mask<4> m1111;
    m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
    index<4> i0000, i0001, i0010, i0011, i0100, i0101, i0110, i0111;
    index<4> i1000, i1001, i1010, i1011, i1100, i1101, i1110, i1111;
    i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;
    i1110[0] = 1; i1110[1] = 1; i1110[2] = 1; i0001[3] = 1;
    i1101[0] = 1; i1101[1] = 1; i0010[2] = 1; i1101[3] = 1;
    i1011[0] = 1; i0100[1] = 1; i1011[2] = 1; i1011[3] = 1;
    i1000[0] = 1; i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
    i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
    i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
    i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
    se_part<4, double> se2(bt.get_bis(), m1111, 2);
    se2.add_map(i0000, i1111);
    se2.add_map(i0101, i1010);
    se2.add_map(i0110, i1001);
    se2.mark_forbidden(i0001);
    se2.mark_forbidden(i0010);
    se2.mark_forbidden(i0011);
    se2.mark_forbidden(i0100);
    se2.mark_forbidden(i0111);
    se2.mark_forbidden(i1000);
    se2.mark_forbidden(i1011);
    se2.mark_forbidden(i1100);
    se2.mark_forbidden(i1101);
    se2.mark_forbidden(i1110);
    sym.insert(se1);
    sym.insert(se2);
    block_tensor_wr_ctrl<4, double> ctrl(bt);
    so_copy<4, double>(sym).perform(ctrl.req_symmetry());
}


int run_bench(size_t n, unsigned buf_mb, unsigned nthr) {

    std::cout << "run_bench(" << n << ", " << buf_mb << ", " << nthr << ")"
        << std::endl;

    bispace<1> si(n);
    for(size_t i = 16; i < n; i+=16) si.split(i);
    bispace<4> sijkl(si&si&si&si);

    btensor<4> A(sijkl), C(sijkl);
    set_symmetry(A);
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
    std::cout << "contract_regular_part_bench: " << tim.duration() << std::endl;

    tp.dissociate();

    std::cout << "SUCCESS" << std::endl;
    return 0;
}


int main(int argc, char **argv) {

    if(argc != 4) {
        std::cout << "Use: \"contract_regular_part_bench N B T\", "
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

