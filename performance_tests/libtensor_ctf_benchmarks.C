#include <iostream>
#include <mpi.h>
#include <libutil/timings/timings.h>
#include <libtensor/libtensor.h>
#include <libtensor/core/allocator.h>
#include <libtensor/linalg/linalg.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/gen_block_tensor/gen_block_tensor_ctrl.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/ctf_block_tensor/ctf_btod_random.h>
#include <libtensor/expr/ctf_btensor/ctf_btensor.h>

using namespace libtensor;

struct timings_tag { };
struct timings_id {
    static const char k_clazz[];
};
const char timings_id::k_clazz[] = "libtensor_ctf_benchmarks";

class bench_timings : public libutil::timings<timings_id, timings_tag, true> {
public:
    void start_timer(const char *t) {
        timings::start_timer(t);
    }

    void stop_timer(const char *t) {
        timings::stop_timer(t);
    }

    void print_timings(std::ostream &os) {
        libutil::timings_store<timings_tag> &t =
            libutil::timings_store<timings_tag>::get_instance();
        t.print(os);
    }

};


void set_symmetry(btensor<4, double> &bt) {

    symmetry<4, double> sym(bt.get_bis());
    sym.insert(se_perm<4, double>(permutation<4>().permute(0, 1).permute(2, 3),
        scalar_transf<double>(1.0)));
    sym.insert(se_perm<4, double>(permutation<4>().permute(0, 1),
        scalar_transf<double>(-1.0)));
    gen_block_tensor_ctrl<4, block_tensor_i_traits<double> > ctrl(bt);
    so_copy<4, double>(sym).perform(ctrl.req_symmetry());
}

void set_symmetry(ctf_btensor<4, double> &bt) {

    symmetry<4, double> sym(bt.get_bis());
    sym.insert(se_perm<4, double>(permutation<4>().permute(0, 1).permute(2, 3),
        scalar_transf<double>(1.0)));
//    sym.insert(se_perm<4, double>(permutation<4>().permute(0, 1),
//        scalar_transf<double>(-1.0)));
    gen_block_tensor_ctrl<4, ctf_block_tensor_i_traits<double> > ctrl(bt);
    so_copy<4, double>(sym).perform(ctrl.req_symmetry());
}


void expr_1(any_tensor<4, double> &ta, any_tensor<4, double> &tb,
    expr_lhs<4, double> &tc) {

    letter i, j, a, b, c, d;
    tc(i|j|a|b) = contract(c|d, ta(a|b|c|d), tb(i|j|c|d));
}


int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    libtensor::ctf::init();
    bool master = libtensor::ctf::is_master();

    size_t mem = 4096;
    libtensor::allocator<double>::init(16, 16, 16777216,
        mem * (1024 * 1024 / sizeof(double)));
    libtensor::linalg::rng_setup(0);

    bench_timings T;

    size_t n = 64;
    bispace<1> x(n), z(n);
    for(size_t i = 16; i < n; i+=16) z.split(i);
    bispace<4> xxxx(x&x&x&x), zzzz(z&z&z&z);

    btensor<4, double> bta(zzzz), btb(zzzz), btc(zzzz);
    ctf_btensor<4, double> dta(xxxx), dtb(xxxx), dtc(xxxx);

    set_symmetry(bta);
    set_symmetry(btb);
    btod_random<4>().perform(bta);
    btod_random<4>().perform(btb);

    T.start_timer("libtensor");
    expr_1(bta, btb, btc);
    T.stop_timer("libtensor");

    set_symmetry(dta);
    set_symmetry(dtb);
    ctf_btod_random<4>().perform(dta);
    ctf_btod_random<4>().perform(dtb);

    T.start_timer("ctf");
    expr_1(dta, dtb, dtc);
    T.stop_timer("ctf");

    if(master) T.print_timings(std::cout);

    libtensor::allocator<double>::shutdown();

    libtensor::ctf::exit();
    MPI_Finalize();

    return 0;
}

