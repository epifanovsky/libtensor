#include <set>
#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_contract2_clst_builder.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_unfold_block_list.h>
#include <libtensor/symmetry/se_part.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_copy.h>
#include "../compare_ref.h"
#include "gen_bto_contract2_clst_builder_test.h"

namespace libtensor {


void gen_bto_contract2_clst_builder_test::perform()
    throw(libtest::test_exception) {

    allocator<double>::init(16, 16, 65536, 65536);

    try {

    test_1();
    test_2();
    test_3();
//    test_4();

    } catch (...) {
        allocator<double>::shutdown();
        throw;
    }
    allocator<double>::shutdown();
}


void gen_bto_contract2_clst_builder_test::test_1() {

    static const char *testname =
        "gen_bto_contract2_clst_builder_test::test_1()";

    typedef std_allocator<double> allocator_type;
    typedef gen_bto_contract2_clst<2, 2, 2, double>::list_type clst_type;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> m1111;
    m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
    bis.split(m1111, 5);
    dimensions<4> bidims = bis.get_block_index_dims();

    index<4> i0000, i1111, i0001, i1110, i0010, i1101, i0011, i1100,
        i0100, i1011, i0101, i1010, i0110, i1001, i0111, i1000;
    i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;
    i1110[0] = 1; i1110[1] = 1; i1110[2] = 1; i0001[3] = 1;
    i1101[0] = 1; i1101[1] = 1; i0010[2] = 1; i1101[3] = 1;
    i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
    i1011[0] = 1; i0100[1] = 1; i1011[2] = 1; i1011[3] = 1;
    i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
    i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
    i1000[0] = 1; i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;

    symmetry<4, double> syma(bis), symb(bis);
    se_part<4, double> se1(bis, m1111, 2);
    se1.add_map(i0000, i1111);
    se1.add_map(i0001, i1110);
    se1.add_map(i0010, i1101);
    se1.add_map(i0011, i1100);
    se1.add_map(i0100, i1011);
    se1.add_map(i0101, i1010);
    se1.add_map(i0110, i1001);
    se1.add_map(i0111, i1000);
    se_perm<4, double> se2(permutation<4>().permute(0, 1),
        scalar_transf<double>(-1.0));
    se_perm<4, double> se3(permutation<4>().permute(2, 3),
        scalar_transf<double>(-1.0));
    syma.insert(se1);
    syma.insert(se2);
    syma.insert(se3);
    symb.insert(se1);
    symb.insert(se2);
    symb.insert(se3);

    block_list<4> bla(bidims), blb(bidims), blax(bidims), blbx(bidims);
    bla.add(i0000);
    bla.add(i0101);
    blb.add(i0000);
    blb.add(i0101);

    contraction2<2, 2, 2> contr;
    contr.contract(2, 2);
    contr.contract(3, 3);
    gen_bto_unfold_block_list<4, btod_traits>(syma, bla).build(blax);
    gen_bto_unfold_block_list<4, btod_traits>(symb, blb).build(blbx);
    gen_bto_contract2_block_list<2, 2, 2> blst(contr, bidims, blax,
        bidims, blbx);
    gen_bto_contract2_clst_builder<2, 2, 2, btod_traits> op(contr, syma, symb,
        bla, blb, bidims, i0101);
    op.build_list(false, blst);

    const clst_type &contr_lst = op.get_clst();
    std::set<size_t> s;
    for(clst_type::const_iterator i = contr_lst.begin(); i != contr_lst.end();
        ++i) {
        abs_index<4> aidxa(i->get_acindex_a(), bidims),
            aidxb(i->get_acindex_b(), bidims);
        if(!aidxa.get_index().equals(i0101) ||
            !aidxb.get_index().equals(i0101)) {
            fail_test(testname, __FILE__, __LINE__, "Wrong blocks contracted.");
        }
        if(s.find(aidxa.get_abs_index()) != s.end()) {
            fail_test(testname, __FILE__, __LINE__,
                "Duplication of contractions found.");
        }
        s.insert(aidxa.get_abs_index());
    }

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void gen_bto_contract2_clst_builder_test::test_2() {

    static const char *testname =
        "gen_bto_contract2_clst_builder_test::test_2()";

    typedef std_allocator<double> allocator_type;
    typedef gen_bto_contract2_clst<2, 2, 2, double>::list_type clst_type;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> m1111;
    m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
    bis.split(m1111, 2);
    bis.split(m1111, 5);
    bis.split(m1111, 7);
    dimensions<4> bidims = bis.get_block_index_dims();

    index<4> i0000, i1111, i0001, i1110, i0010, i1101, i0011, i1100,
        i0100, i1011, i0101, i1010, i0110, i1001, i0111, i1000;
    i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;
    i1110[0] = 1; i1110[1] = 1; i1110[2] = 1; i0001[3] = 1;
    i1101[0] = 1; i1101[1] = 1; i0010[2] = 1; i1101[3] = 1;
    i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
    i1011[0] = 1; i0100[1] = 1; i1011[2] = 1; i1011[3] = 1;
    i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
    i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
    i1000[0] = 1; i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;

    symmetry<4, double> syma(bis), symb(bis);
    se_part<4, double> se1(bis, m1111, 2);
    se1.add_map(i0000, i1111);
    se1.add_map(i0001, i1110);
    se1.add_map(i0010, i1101);
    se1.add_map(i0011, i1100);
    se1.add_map(i0100, i1011);
    se1.add_map(i0101, i1010);
    se1.add_map(i0110, i1001);
    se1.add_map(i0111, i1000);
    se_perm<4, double> se2(permutation<4>().permute(0, 1),
        scalar_transf<double>(-1.0));
    se_perm<4, double> se3(permutation<4>().permute(2, 3),
        scalar_transf<double>(-1.0));
    syma.insert(se1);
    syma.insert(se2);
    syma.insert(se3);
    symb.insert(se1);
    symb.insert(se2);
    symb.insert(se3);

    block_list<4> bla(bidims), blb(bidims), blax(bidims), blbx(bidims);
    for(size_t ii = 0; ii < 2; ii++)
    for(size_t jj = ii; jj < 2; jj++)
    for(size_t kk = 0; kk < 2; kk++)
    for(size_t ll = kk; ll < 2; ll++) {
        index<4> idx1, idx2;
        idx1[0] = ii; idx1[1] = jj; idx1[2] = kk; idx1[3] = ll;
        idx2[0] = ii; idx2[1] = 2 + jj; idx2[2] = kk; idx2[3] = 2 + ll;
        bla.add(idx1);
        bla.add(idx2);
        blb.add(idx1);
        blb.add(idx2);
    }

    index<4> i0302, i0303, i0313, i1212;
    i0302[0] = 0; i0302[1] = 3; i0302[2] = 0; i0302[3] = 2;
    i0303[0] = 0; i0303[1] = 3; i0303[2] = 0; i0303[3] = 3;
    i0313[0] = 0; i0313[1] = 3; i0313[2] = 1; i0313[3] = 3;
    i1212[0] = 1; i1212[1] = 2; i1212[2] = 1; i1212[3] = 2;

    contraction2<2, 2, 2> contr;
    contr.contract(2, 2);
    contr.contract(3, 3);
    gen_bto_unfold_block_list<4, btod_traits>(syma, bla).build(blax);
    gen_bto_unfold_block_list<4, btod_traits>(symb, blb).build(blbx);
    gen_bto_contract2_block_list<2, 2, 2> blst(contr, bidims, blax,
        bidims, blbx);
    gen_bto_contract2_clst_builder<2, 2, 2, btod_traits> op(contr, syma, symb,
        bla, blb, bidims, i1212);
    op.build_list(false, blst);

    const clst_type &contr_lst = op.get_clst();
    std::set<size_t> s;
    for(clst_type::const_iterator i = contr_lst.begin(); i != contr_lst.end();
        ++i) {
        abs_index<4> aidxa(i->get_acindex_a(), bidims),
            aidxb(i->get_acindex_b(), bidims);
        if(!aidxa.get_index().equals(aidxb.get_index()) ||
            (!aidxa.get_index().equals(i0302) &&
            !aidxa.get_index().equals(i0303) &&
            !aidxa.get_index().equals(i0313))) {
            fail_test(testname, __FILE__, __LINE__, "Wrong blocks contracted.");
        }
        if(s.find(aidxa.get_abs_index()) != s.end()) {
            fail_test(testname, __FILE__, __LINE__,
                "Duplication of contractions found.");
        }
        s.insert(aidxa.get_abs_index());
    }

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void gen_bto_contract2_clst_builder_test::test_3() {

    static const char *testname =
        "gen_bto_contract2_clst_builder_test::test_3()";

    typedef std_allocator<double> allocator_type;
    typedef gen_bto_contract2_clst<2, 2, 2, double>::list_type clst_type;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 5; i2[3] = 5;
    dimensions<4> dimsa(index_range<4>(i1, i2));
    i2[0] = 5; i2[1] = 5; i2[2] = 11; i2[3] = 11;
    dimensions<4> dimsb(index_range<4>(i1, i2));
    i2[0] = 9; i2[1] = 9; i2[2] = 11; i2[3] = 11;
    dimensions<4> dimsc(index_range<4>(i1, i2));
    block_index_space<4> bisa(dimsa), bisb(dimsb), bisc(dimsc);
    mask<4> m0011, m1100, m1111;
    m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;
    m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
    bisa.split(m1100, 3);
    bisa.split(m1100, 5);
    bisa.split(m0011, 4);
    bisb.split(m1100, 4);
    bisb.split(m0011, 6);
    bisc.split(m1100, 3);
    bisc.split(m1100, 5);
    bisc.split(m0011, 6);
    dimensions<4> bidimsa = bisa.get_block_index_dims();
    dimensions<4> bidimsb = bisb.get_block_index_dims();
    dimensions<4> bidimsc = bisc.get_block_index_dims();

    symmetry<4, double> syma(bisa), symb(bisb);
    se_perm<4, double> se1(permutation<4>().permute(0, 1),
        scalar_transf<double>());
    se_perm<4, double> se2(permutation<4>().permute(2, 3),
        scalar_transf<double>());
    syma.insert(se1);
    syma.insert(se2);
    symb.insert(se1);
    symb.insert(se2);

    block_list<4> bla(bidimsa), blb(bidimsb), blax(bidimsa), blbx(bidimsb);
    orbit_list<4, double> ola(syma), olb(symb);
    for(orbit_list<4, double>::iterator i = ola.begin(); i != ola.end(); ++i) {
        bla.add(ola.get_abs_index(i));
    }
    for(orbit_list<4, double>::iterator i = olb.begin(); i != olb.end(); ++i) {
        blb.add(olb.get_abs_index(i));
    }

    index<4> i0000;

    contraction2<2, 2, 2> contr;
    contr.contract(2, 0);
    contr.contract(3, 1);
    gen_bto_unfold_block_list<4, btod_traits>(syma, bla).build(blax);
    gen_bto_unfold_block_list<4, btod_traits>(symb, blb).build(blbx);
    gen_bto_contract2_block_list<2, 2, 2> blst(contr, bidimsa, blax,
        bidimsb, blbx);
    gen_bto_contract2_clst_builder<2, 2, 2, btod_traits> op(contr, syma, symb,
        bla, blb, bidimsc, i0000);
    op.build_list(false, blst);

    const clst_type &contr_lst = op.get_clst();
    size_t n = 0;
    for(clst_type::const_iterator i = contr_lst.begin(); i != contr_lst.end();
        ++i) {
        n++;
    }
    if(n != 3) {
        fail_test(testname, __FILE__, __LINE__,
            "Wrong number of contractions.");
    }

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


} // namespace libtensor
