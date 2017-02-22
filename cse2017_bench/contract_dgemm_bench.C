#include <cstdlib>
#include <iostream>
#include <mkl.h>
#include <libutil/timings/timer.h>
#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/libtensor.h>
#include <libtensor/block_tensor/btod_set.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_contract2_simple_impl.h>
#include <libtensor/block_tensor/btod_traits.h>

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


int run_bench(size_t n, unsigned nthr) {

    std::cout << "run_bench(" << n << ", " << nthr << ")" << std::endl;

    bispace<1> si(n);
    for(size_t i = 16; i < n; i+=16) si.split(i);
    bispace<4> sijkl(si&si&si&si);

    btensor<4> A(sijkl), B(sijkl), C(sijkl);
    btod_set<4>(0.55).perform(A);
    btod_set<4>(2.0).perform(B);

    contraction2<2, 2, 2> contr;
    contr.contract(2, 0);
    contr.contract(3, 1);

    scalar_transf<double> ka(1.0), kb(1.0), kc(1.0);

    libutil::thread_pool tp(nthr, nthr);
    tp.associate();

    libutil::timer tim;
    tim.start();
    gen_bto_contract2_simple<2, 2, 2, btod_traits, contract_timings> opcontr(
        contr, A, ka, B, kb, kc);
    gen_bto_aux_copy<4, btod_traits> opcopy(opcontr.get_symmetry(), C);
    opcopy.open();
    opcontr.perform(opcopy);
    opcopy.close();
    tim.stop();
    std::cout << "contract_dgemm_bench: " << tim.duration() << std::endl;

    tp.dissociate();

    std::cout << "SUCCESS" << std::endl;
    return 0;
}


int main(int argc, char **argv) {

    if(argc != 3) {
        std::cout << "Use: \"contract_dgemm_bench N T\", "
                     "where N is matrix size, "
                     "T is number of threads" << std::endl;
        return -1;
    }

    warmup();

    int n = atoi(argv[1]);
    int t = atoi(argv[2]);
    return run_bench(n, t);
}

