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
#include <libtensor/block_tensor/btod_set.h>
#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/ctf_block_tensor/ctf_btod_random.h>
#include <libtensor/ctf_block_tensor/ctf_btod_set.h>
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


void set_symmetry(btensor<4, double> &bt, int z) {

    symmetry<4, double> sym(bt.get_bis());
    permutation<4> p1023, p1032;
    p1023.permute(0, 1);
    p1032.permute(0, 1).permute(2, 3);
    if(z > 0) {
        sym.insert(se_perm<4, double>(p1032, scalar_transf<double>(1.0)));
    }
    if(z > 1) {
        sym.insert(se_perm<4, double>(p1023, scalar_transf<double>(-1.0)));
    }
    gen_block_tensor_ctrl<4, block_tensor_i_traits<double> > ctrl(bt);
    ctrl.req_zero_all_blocks();
    so_copy<4, double>(sym).perform(ctrl.req_symmetry());
}

void set_random(btensor<4, double> &bt) {
    btod_set<4>(1.0).perform(bt);
    //btod_random<4>().perform(bt);
}

void set_symmetry(ctf_btensor<4, double> &bt, int z) {

    symmetry<4, double> sym(bt.get_bis());
    permutation<4> p1023, p1032;
    p1023.permute(0, 1);
    p1032.permute(0, 1).permute(2, 3);
    if(z > 0) {
        sym.insert(se_perm<4, double>(p1032, scalar_transf<double>(1.0)));
    }
    if(z > 1) {
        sym.insert(se_perm<4, double>(p1023, scalar_transf<double>(-1.0)));
    }
    gen_block_tensor_ctrl<4, ctf_block_tensor_i_traits<double> > ctrl(bt);
    ctrl.req_zero_all_blocks();
    so_copy<4, double>(sym).perform(ctrl.req_symmetry());
}


void set_random(ctf_btensor<4, double> &bt) {
    ctf_btod_set<4>(1.0).perform(bt);
    //ctf_btod_random<4>().perform(bt);
}

void expr_1(any_tensor<4, double> &ta, any_tensor<4, double> &tb,
    expr_lhs<4, double> &tc) {

    letter i, j, a, b, c, d;
    tc(i|j|a|b) = contract(c|d, ta(a|b|c|d), tb(i|j|c|d));
}

void expr_2(any_tensor<4, double> &ta, expr_lhs<4, double> &tb) {

    letter i, j, a, b, c, d;
    tb(i|j|a|b) = ta(i|j|a|b) + ta(j|i|a|b);
}

void expr_3(any_tensor<4, double> &ta, expr_lhs<4, double> &tb) {

    letter i, j, a, b, c, d;
    tb(i|j|a|b) = ta(i|j|a|b) + ta(i|j|b|a);
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

    size_t ni = 16, na = 128;
    bispace<1> si(ni), sj(ni), sa(na), sb(na);
    for(size_t i = 16; i < na; i+=16) sa.split(i);
    bispace<2> sii(si&si), sjj(sj&sj), saa(sa&sa), sbb(sb&sb);
    bispace<4> bbbb(sbb&sbb), aaaa(saa&saa);
    bispace<4> jjbb(sjj|sbb), iiaa(sii|saa);

    btensor<4, double> bta(aaaa), btb(iiaa), btc(iiaa), btd(aaaa);
    ctf_btensor<4, double> dta(bbbb), dtb(jjbb), dtc(jjbb), dtd(bbbb);

/*
    set_symmetry(bta, 0); set_symmetry(btb, 0);
    set_random(bta); set_random(btb);
    T.start_timer("libtensor0[nosym]");
    expr_1(bta, btb, btc);
    T.stop_timer("libtensor0[nosym]");
    set_symmetry(dta, 0); set_symmetry(dtb, 0);
    set_random(dta); set_random(dtb);
    T.start_timer("ctf0[nosym]");
    expr_1(dta, dtb, dtc);
    T.stop_timer("ctf0[nosym]");

    set_symmetry(bta, 1); set_symmetry(btb, 1);
    set_random(bta); set_random(btb);
    T.start_timer("libtensor1[jilk]");
    expr_1(bta, btb, btc);
    T.stop_timer("libtensor1[jilk]");
    set_symmetry(dta, 1); set_symmetry(dtb, 1);
    set_random(dta); set_random(dtb);
    T.start_timer("ctf1[jilk]");
    expr_1(dta, dtb, dtc);
    T.stop_timer("ctf1[jilk]");

    set_symmetry(bta, 2); set_symmetry(btb, 2);
    set_random(bta); set_random(btb);
    T.start_timer("libtensor2[ji,lk]");
    expr_1(bta, btb, btc);
    T.stop_timer("libtensor2[ji,lk]");
    set_symmetry(dta, 2); set_symmetry(dtb, 2);
    set_random(dta); set_random(dtb);
    T.start_timer("ctf2[ji,lk]");
    expr_1(dta, dtb, dtc);
    T.stop_timer("ctf2[ji,lk]");
 */

    set_symmetry(bta, 0);
    set_random(bta);
    T.start_timer("libtensor[ijab+jiab]");
    expr_2(bta, btd);
    T.stop_timer("libtensor[ijab+jiab]");
    set_symmetry(dta, 0);
    set_random(dta);
    T.start_timer("ctf[ijab+jiab]");
    expr_2(dta, dtd);
    T.stop_timer("ctf[ijab+jiab]");

    set_symmetry(bta, 0);
    set_random(bta);
    T.start_timer("libtensor[ijab+ijba]");
    expr_2(bta, btd);
    T.stop_timer("libtensor[ijab+ijba]");
    set_symmetry(dta, 0);
    set_random(dta);
    T.start_timer("ctf[ijab+ijba]");
    expr_2(dta, dtd);
    T.stop_timer("ctf[ijab+ijba]");

    if(master) T.print_timings(std::cout);

    libtensor::allocator<double>::shutdown();

    libtensor::ctf::exit();
    MPI_Finalize();

    return 0;
}

