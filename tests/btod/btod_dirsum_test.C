#include <cmath>
#include <ctime>
#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/btod/scalar_transf_double.h>
#include <libtensor/btod/btod_dirsum.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/symmetry/permutation_group.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/se_label.h>
#include <libtensor/symmetry/se_part.h>
#include <libtensor/dense_tensor/tod_btconv.h>
#include <libtensor/dense_tensor/tod_dirsum.h>
#include "../compare_ref.h"
#include "btod_dirsum_test.h"
#include <libtensor/btod/btod_print.h>

namespace libtensor {


void btod_dirsum_test::perform() throw(libtest::test_exception) {

    srand48(time(0));

    test_ij_i_j_1(true);
    test_ij_i_j_1(true, -0.5);
    test_ij_i_j_1(false);
    test_ij_i_j_1(false, 0.5);

    test_ij_i_j_2(true);
    test_ij_i_j_2(true, -0.8);
    test_ij_i_j_2(false);
    test_ij_i_j_2(false, 0.8);

    test_ij_i_j_3(true);
    test_ij_i_j_3(true, -0.8);
    test_ij_i_j_3(false);
    test_ij_i_j_3(false, 0.8);

    test_ijk_ij_k_1(true);
    test_ijk_ij_k_1(true, 1.2);
    test_ijk_ij_k_1(false);
    test_ijk_ij_k_1(false, -1.2);

    test_ikjl_ij_kl_1(true);
    test_ikjl_ij_kl_1(true, 2.0);
    test_ikjl_ij_kl_1(false);
    test_ikjl_ij_kl_1(false, -2.0);

    test_ikjl_ij_kl_2(true);
    test_ikjl_ij_kl_2(true, 2.0);
    test_ikjl_ij_kl_2(false);
    test_ikjl_ij_kl_2(false, -2.0);

    test_ikjl_ij_kl_3a(true, true, true);
    test_ikjl_ij_kl_3a(true, false, true);
    test_ikjl_ij_kl_3a(false, true, true);
    test_ikjl_ij_kl_3a(false, false, true);
    test_ikjl_ij_kl_3a(true, true, true, 1.2);
    test_ikjl_ij_kl_3a(true, false, true, -1.2);
    test_ikjl_ij_kl_3a(false, true, true, 1.2);
    test_ikjl_ij_kl_3a(false, false, true, -1.2);
    test_ikjl_ij_kl_3a(true, true, false);
    test_ikjl_ij_kl_3a(true, false, false);
    test_ikjl_ij_kl_3a(false, true, false);
    test_ikjl_ij_kl_3a(false, false, false);
    test_ikjl_ij_kl_3a(true, true, false, 1.2);
    test_ikjl_ij_kl_3a(true, false, false, -1.2);
    test_ikjl_ij_kl_3a(false, true, false, 1.2);
    test_ikjl_ij_kl_3a(false, false, false, -1.2);

    test_ikjl_ij_kl_3b(true);
    test_ikjl_ij_kl_3b(true, 0.2);
    test_ikjl_ij_kl_3b(false);
    test_ikjl_ij_kl_3b(false, -0.2);

    test_ikjl_ij_kl_3c(true);
    test_ikjl_ij_kl_3c(true, -1.5);
    test_ikjl_ij_kl_3c(false);
    test_ikjl_ij_kl_3c(false, 1.5);

    test_iklj_ij_kl_1(true);
    test_iklj_ij_kl_1(true, 1.25);
    test_iklj_ij_kl_1(false);
    test_iklj_ij_kl_1(false, -1.25);

}

void btod_dirsum_test::test_ij_i_j_1(bool rnd, double d)
    throw(libtest::test_exception) {

    //  c_{ij} = a_i + b_j

    std::stringstream tnss;
    tnss << "btod_dirsum_test::test_ij_i_j_1(" << rnd << ", " << d << ")";
    std::string tns = tnss.str();

    typedef std_allocator<double> allocator;

    cpu_pool cpus(1);

    try {

    size_t ni = 9, nj = 7;

    index<1> ia1, ia2; ia2[0] = ni - 1;
    index<1> ib1, ib2; ib2[0] = nj - 1;
    index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<1> dima(index_range<1>(ia1, ia2));
    dimensions<1> dimb(index_range<1>(ib1, ib2));
    dimensions<2> dimc(index_range<2>(ic1, ic2));
    block_index_space<1> bisa(dima), bisb(dimb);
    block_index_space<2> bisc(dimc);

    block_tensor<1, double, allocator> bta(bisa);
    block_tensor<1, double, allocator> btb(bisb);
    block_tensor<2, double, allocator> btc(bisc);

    dense_tensor<1, double, allocator> ta(dima);
    dense_tensor<1, double, allocator> tb(dimb);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    //  Fill in random input

    btod_random<1>().perform(bta);
    btod_random<1>().perform(btb);
    if(rnd) btod_random<2>().perform(btc);
    tod_btconv<1>(bta).perform(ta);
    tod_btconv<1>(btb).perform(tb);
    if(d != 0.0) tod_btconv<2>(btc).perform(tc_ref);

    //  Generate reference data

	if(d == 0.0) {
	    tod_dirsum<1, 1>(ta, 1.0, tb, 1.0).perform(cpus, true, 1.0, tc_ref);
	} else {
	    tod_dirsum<1, 1>(ta, 1.0, tb, 1.0).perform(cpus, false, d, tc_ref);
	}

    //  Invoke the direct sum routine

    if(d == 0.0) btod_dirsum<1, 1>(bta, 1.0, btb, 1.0).perform(btc);
    else btod_dirsum<1, 1>(bta, 1.0, btb, 1.0).perform(btc, scalar_transf<double>(d));
    tod_btconv<2>(btc).perform(tc);

    //  Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}

void btod_dirsum_test::test_ij_i_j_2(bool rnd, double d)
    throw(libtest::test_exception) {

    //  c_{ij} = a_i + a_j

    std::stringstream tnss;
    tnss << "btod_dirsum_test::test_ij_i_j_2(" << rnd << ", " << d << ")";
    std::string tns = tnss.str();

    typedef std_allocator<double> allocator;

    cpu_pool cpus(1);

	try {

    size_t ni = 9;

    index<1> ia1, ia2; ia2[0] = ni - 1;
    index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = ni - 1;
    dimensions<1> dima(index_range<1>(ia1, ia2));
    dimensions<2> dimc(index_range<2>(ic1, ic2));
    block_index_space<1> bisa(dima);
    block_index_space<2> bisc(dimc);

    mask<1> ma; ma[0] = true;
    bisa.split(ma, 3);
    mask<2> mc; mc[0] = true; mc[1] = true;
    bisc.split(mc, 3);

    block_tensor<1, double, allocator> bta(bisa);
    block_tensor<2, double, allocator> btc(bisc);

    dense_tensor<1, double, allocator> ta(dima);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    //  Fill in random input

    btod_random<1>().perform(bta);
    if(rnd) btod_random<2>().perform(btc);

    bta.set_immutable();

    tod_btconv<1>(bta).perform(ta);
    if(d != 0.0) tod_btconv<2>(btc).perform(tc_ref);

    //  Generate reference data

	if(d == 0.0) {
	    tod_dirsum<1, 1>(ta, 1.0, ta, 1.0).perform(cpus, true, 1.0, tc_ref);
	} else {
	    tod_dirsum<1, 1>(ta, 1.0, ta, 1.0).perform(cpus, false, d, tc_ref);
	}

    // Check the symmetry of the result

    btod_dirsum<1, 1> op(bta, 1.0, bta, 1.0);
    {
        permutation<2> p01;
        p01.permute(0, 1);
        scalar_transf<double> tr0, tr1(-1.);
        const symmetry<2, double> &sym = op.get_symmetry();
        symmetry<2, double>::iterator is = sym.begin();
        const symmetry_element_set<2, double> &set = sym.get_subset(is);
        symmetry_element_set_adapter<2, double, se_perm<2, double> > adapter(set);
        permutation_group<2, double> grp(set);
        if (! grp.is_member(tr0, p01)) {
            fail_test(tns.c_str(), __FILE__, __LINE__,
                    "Permutational symmetry (0-1) missing.");
        }
    }

    //  Invoke the direct sum routine

    if(d == 0.0) op.perform(btc);
    else op.perform(btc, scalar_transf<double>(d));
    tod_btconv<2>(btc).perform(tc);

    //  Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, 1e-15);


    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}

void btod_dirsum_test::test_ij_i_j_3(bool rnd, double d)
    throw(libtest::test_exception) {

    //  c_{ij} = a_i - a_j

    std::stringstream tnss;
    tnss << "btod_dirsum_test::test_ij_i_j_3(" << rnd << ", " << d << ")";
    std::string tns = tnss.str();

    typedef std_allocator<double> allocator;

    cpu_pool cpus(1);

	try {

    size_t ni = 9;

    index<1> ia1, ia2; ia2[0] = ni - 1;
    index<2> ic1, ic2; ic2[0] = ni - 1; ic2[1] = ni - 1;
    dimensions<1> dima(index_range<1>(ia1, ia2));
    dimensions<2> dimc(index_range<2>(ic1, ic2));
    block_index_space<1> bisa(dima);
    block_index_space<2> bisc(dimc);

    mask<1> ma; ma[0] = true;
    bisa.split(ma, 3);
    mask<2> mc; mc[0] = true; mc[1] = true;
    bisc.split(mc, 3);

    block_tensor<1, double, allocator> bta(bisa);
    block_tensor<2, double, allocator> btc(bisc);

    dense_tensor<1, double, allocator> ta(dima);
    dense_tensor<2, double, allocator> tc(dimc);
    dense_tensor<2, double, allocator> tc_ref(dimc);

    //  Fill in random input

    btod_random<1>().perform(bta);
    if(rnd) btod_random<2>().perform(btc);

    bta.set_immutable();

    tod_btconv<1>(bta).perform(ta);
    if(d != 0.0) tod_btconv<2>(btc).perform(tc_ref);

    //  Generate reference data

	if(d == 0.0) {
	    tod_dirsum<1, 1>(ta, 1.0, ta, -1.0).perform(cpus, true, 1.0, tc_ref);
	} else {
	    tod_dirsum<1, 1>(ta, 1.0, ta, -1.0).perform(cpus, false, d, tc_ref);
	}

    // Check the symmetry of the result

    btod_dirsum<1, 1> op(bta, 1.0, bta, -1.0);
    {
        permutation<2> p01;
        p01.permute(0, 1);
        scalar_transf<double> tr0, tr1(-1.);
        const symmetry<2, double> &sym = op.get_symmetry();
        symmetry<2, double>::iterator is = sym.begin();
        const symmetry_element_set<2, double> &set = sym.get_subset(is);
        symmetry_element_set_adapter<2, double, se_perm<2, double> > adapter(set);
        permutation_group<2, double> grp(set);
        if (! grp.is_member(tr1, p01)) {
            fail_test(tns.c_str(), __FILE__, __LINE__,
                    "Permutational symmetry (0-1) missing.");
        }
    }

    //  Invoke the direct sum routine

    if(d == 0.0) op.perform(btc);
    else op.perform(btc, scalar_transf<double>(d));
    tod_btconv<2>(btc).perform(tc);

    //  Compare against the reference

    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, 1e-15);


    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}

void btod_dirsum_test::test_ijk_ij_k_1(bool rnd, double d)
    throw(libtest::test_exception) {

    // c_{ijk} = a_{ij} + b_k

    std::stringstream tnss;
    tnss << "btod_dirsum_test::test_ijk_ij_k_1(" << rnd << ", " << d
        << ")";
    std::string tns = tnss.str();

    typedef std_allocator<double> allocator;

    cpu_pool cpus(1);

	try {

    size_t ni = 9, nj = 9, nk = 7;

    index<2> ia1, ia2;
    ia2[0] = ni - 1; ia2[1] = nj - 1;
    index<1> ib1, ib2;
    ib2[0] = nk - 1;
    index<3> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1;
    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<1> dimb(index_range<1>(ib1, ib2));
    dimensions<3> dimc(index_range<3>(ic1, ic2));
    block_index_space<2> bisa(dima);
    block_index_space<1> bisb(dimb);
    block_index_space<3> bisc(dimc);

    block_tensor<2, double, allocator> bta(bisa);
    block_tensor<1, double, allocator> btb(bisb);
    block_tensor<3, double, allocator> btc(bisc);

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<1, double, allocator> tb(dimb);
    dense_tensor<3, double, allocator> tc(dimc);
    dense_tensor<3, double, allocator> tc_ref(dimc);

    //  Fill in random input

    btod_random<2>().perform(bta);
    btod_random<1>().perform(btb);
    if(rnd) btod_random<3>().perform(btc);
    tod_btconv<2>(bta).perform(ta);
    tod_btconv<1>(btb).perform(tb);
    if(d != 0.0) tod_btconv<3>(btc).perform(tc_ref);

    //  Generate reference data

	if(d == 0.0) {
		tod_dirsum<2, 1>(ta, 1.5, tb, 1.0).perform(cpus, true, 1.0, tc_ref);
	} else {
		tod_dirsum<2, 1>(ta, 1.5, tb, 1.0).perform(cpus, false, d, tc_ref);
	}

    //  Invoke the direct sum routine

    if(d == 0.0) {
        btod_dirsum<2, 1>(bta, 1.5, btb, 1.0).perform(btc);
    } else {
        btod_dirsum<2, 1>(bta, 1.5, btb, 1.0).perform(btc, scalar_transf<double>(d));
    }
    tod_btconv<3>(btc).perform(tc);

    //  Compare against the reference

    compare_ref<3>::compare(tns.c_str(), tc, tc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}

void btod_dirsum_test::test_ikjl_ij_kl_1(bool rnd, double d)
    throw(libtest::test_exception) {

    //  c_{ikjl} = a_{ij} + b_{kl}

    std::stringstream tnss;
    tnss << "btod_dirsum_test::test_ikjl_ij_kl_1(" << rnd << ", " << d
        << ")";
    std::string tns = tnss.str();

    typedef std_allocator<double> allocator;

    cpu_pool cpus(1);

	try {

    size_t ni = 9, nj = 9, nk = 7, nl = 7;

    index<2> ia1, ia2;
    ia2[0] = ni - 1; ia2[1] = nj - 1;
    index<2> ib1, ib2;
    ib2[0] = nk - 1; ib2[1] = nl - 1;
    index<4> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nk - 1; ic2[2] = nj - 1; ic2[3] = nl - 1;
    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<2> dimb(index_range<2>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    block_index_space<2> bisa(dima), bisb(dimb);
    block_index_space<4> bisc(dimc);

    block_tensor<2, double, allocator> bta(bisa);
    block_tensor<2, double, allocator> btb(bisb);
    block_tensor<4, double, allocator> btc(bisc);

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<2, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    //  Fill in random input

    btod_random<2>().perform(bta);
    btod_random<2>().perform(btb);
    if(rnd) btod_random<4>().perform(btc);
    tod_btconv<2>(bta).perform(ta);
    tod_btconv<2>(btb).perform(tb);
    if(d != 0.0) tod_btconv<4>(btc).perform(tc_ref);

    //  Generate reference data

    permutation<4> permc;
	permc.permute(1, 2);
	if(d == 0.0) {
		tod_dirsum<2, 2>(ta, 1.5, tb, -1.0, permc).
		    perform(cpus, true, 1.0, tc_ref);
	} else {
		tod_dirsum<2, 2>(ta, 1.5, tb, -1.0, permc).
		    perform(cpus, false, d, tc_ref);
	}

    //  Invoke the direct sum routine

    if(d == 0.0) {
        btod_dirsum<2, 2>(bta, 1.5, btb, -1.0, permc).perform(btc);
    } else {
        btod_dirsum<2, 2>(bta, 1.5, btb, -1.0, permc).perform(btc, scalar_transf<double>(d));
    }
    tod_btconv<4>(btc).perform(tc);

    //  Compare against the reference

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}

void btod_dirsum_test::test_ikjl_ij_kl_2(bool rnd, double d)
    throw(libtest::test_exception) {

    //  c_{ikjl} = a_{ij} + b_{kl}
    // with splits

    std::stringstream tnss;
    tnss << "btod_dirsum_test::test_ikjl_ij_kl_2(" << rnd << ", " << d
        << ")";
    std::string tns = tnss.str();

    typedef std_allocator<double> allocator;

    cpu_pool cpus(1);

	try {

    size_t ni = 7, nj = 11, nk = 7, nl = 5;

    index<2> ia1, ia2;
    ia2[0] = ni - 1; ia2[1] = nj - 1;
    index<2> ib1, ib2;
    ib2[0] = nk - 1; ib2[1] = nl - 1;
    index<4> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nk - 1; ic2[2] = nj - 1; ic2[3] = nl - 1;
    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<2> dimb(index_range<2>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    block_index_space<2> bisa(dima), bisb(dimb);
    block_index_space<4> bisc(dimc);
    // split dimensions
    mask<2> msk1, msk2;
    mask<4> mskc;

    msk1[0]=true; msk2[1]=true;
    bisa.split(msk1,ni/2-1); bisa.split(msk2,nj/2);
    bisb.split(msk1,nk/2-1); bisb.split(msk2,nl/2);
    mskc[0]=true;
    bisc.split(mskc,ni/2-1);
    mskc[0]=false; mskc[1]=true;
    bisc.split(mskc,nk/2-1);
    mskc[1]=false; mskc[2]=true;
    bisc.split(mskc,nj/2);
    mskc[2]=false; mskc[3]=true;
    bisc.split(mskc,nl/2);
    bisc.match_splits();

    block_tensor<2, double, allocator> bta(bisa);
    block_tensor<2, double, allocator> btb(bisb);
    block_tensor<4, double, allocator> btc(bisc);

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<2, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    //  Fill in random input

    btod_random<2>().perform(bta);
    btod_random<2>().perform(btb);

    // set zero blocks
    {
        index<2> idxa, idxb;
        idxa[1]=1;
        idxb[0]=1;
        block_tensor_ctrl<2,double> bctrla(bta), bctrlb(btb);
        bctrla.req_zero_block(idxa);
        bctrlb.req_zero_block(idxb);
    }
    if(rnd) btod_random<4>().perform(btc);
    tod_btconv<2>(bta).perform(ta);
    tod_btconv<2>(btb).perform(tb);
    if(d != 0.0) tod_btconv<4>(btc).perform(tc_ref);

    //  Generate reference data

	permutation<4> permc;
	permc.permute(1, 2);
	if(d == 0.0) {
		tod_dirsum<2, 2>(ta, 1.5, tb, -1.0, permc).
		    perform(cpus, true, 1.0, tc_ref);
	} else {
		tod_dirsum<2, 2>(ta, 1.5, tb, -1.0, permc).
		    perform(cpus, false, d, tc_ref);
	}

    //  Invoke the direct sum routine

    if(d == 0.0) {
        btod_dirsum<2, 2>(bta, 1.5, btb, -1.0, permc).perform(btc);
    } else {
        btod_dirsum<2, 2>(bta, 1.5, btb, -1.0, permc).perform(btc, scalar_transf<double>(d));
    }
    tod_btconv<4>(btc).perform(tc);

    //  Compare against the reference

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}

void btod_dirsum_test::test_ikjl_ij_kl_3a(bool s1, bool s2,
        bool rnd, double d) throw(libtest::test_exception) {

    //  c_{ikjl} = a_{ij} + b_{kl}
    // with splits and se_perm i<->j, k<->l

    std::stringstream tnss;
    tnss << "btod_dirsum_test::test_ikjl_ij_kl_3a(" << s1 << ", " << s2
            << ", " << rnd << ", " << d << ")";
    std::string tns = tnss.str();

    typedef std_allocator<double> allocator;

    cpu_pool cpus(1);

    try {

    size_t ni = 8, nj = 8, nk = 8, nl = 8;

    index<2> ia1, ia2;
    ia2[0] = ni - 1; ia2[1] = nj - 1;
    index<2> ib1, ib2;
    ib2[0] = nk - 1; ib2[1] = nl - 1;
    index<4> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nk - 1; ic2[2] = nj - 1; ic2[3] = nl - 1;
    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<2> dimb(index_range<2>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    block_index_space<2> bisa(dima), bisb(dimb);
    block_index_space<4> bisc(dimc);
    // split dimensions
    mask<2> mska;
    mask<4> mskc;

    mska[0] = true; mska[1] = true;
    bisa.split(mska, 4); 
    bisb.split(mska, 4); 
    mskc[0] = true; mskc[1] = true; mskc[2] = true; mskc[3] = true;
    bisc.split(mskc, 4); 

    block_tensor<2, double, allocator> bta(bisa);
    block_tensor<2, double, allocator> btb(bisb);
    block_tensor<4, double, allocator> btc(bisc);
    symmetry<4, double> sym_ref(bisc);

    // add symmetries
    {
        block_tensor_ctrl<2, double> ctrla(bta), ctrlb(btb);
        block_tensor_ctrl<4, double> ctrlc(btc);

        scalar_transf<double> tr0, tr1(-1.);
        se_perm<2, double> se1(permutation<2>().permute(0, 1), s1 ? tr0 : tr1);
        se_perm<2, double> se2(permutation<2>().permute(0, 1), s2 ? tr0 : tr1);

        ctrla.req_symmetry().insert(se1);
        ctrlb.req_symmetry().insert(se2);

        if (s1) {
            se_perm<4, double> se_ref(permutation<4>().permute(0, 2),
                    s1 ? tr0 : tr1);
            sym_ref.insert(se_ref);
            ctrlc.req_symmetry().insert(se_ref);
        }
        if (s2) {
            se_perm<4, double> se_ref(permutation<4>().permute(1, 3),
                    s2 ? tr0 : tr1);
            sym_ref.insert(se_ref);
            ctrlc.req_symmetry().insert(se_ref);
        }
        if (!s1 && !s2) {
            permutation<4> p;
            p.permute(0, 2).permute(1, 3);
            se_perm<4, double> se_ref(p, s1 ? tr0 : tr1);
            sym_ref.insert(se_ref);
            ctrlc.req_symmetry().insert(se_ref);
        }
    }

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<2, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    //  Fill in random input

    btod_random<2>().perform(bta);
    btod_random<2>().perform(btb);

    if(rnd) btod_random<4>().perform(btc);
    tod_btconv<2>(bta).perform(ta);
    tod_btconv<2>(btb).perform(tb);
    if(d != 0.0) tod_btconv<4>(btc).perform(tc_ref);

    //  Generate reference data

    permutation<4> permc;
    permc.permute(1, 2);
    if(d == 0.0) {
        tod_dirsum<2, 2>(ta, 1.5, tb, -1.0, permc).
            perform(cpus, true, 1.0, tc_ref);
    } else {
        tod_dirsum<2, 2>(ta, 1.5, tb, -1.0, permc).
            perform(cpus, false, d, tc_ref);
    }

    //  Invoke the direct sum routine

    if(d == 0.0) {
        btod_dirsum<2, 2>(bta, 1.5, btb, -1.0, permc).perform(btc);
    } else {
        btod_dirsum<2, 2>(bta, 1.5, btb, -1.0, permc).perform(btc, scalar_transf<double>(d));
    }

    // Compare symmetry
    {
        block_tensor_ctrl<4, double> ctrlc(btc);
        compare_ref<4>::compare(tns.c_str(),
                sym_ref, ctrlc.req_const_symmetry());
    }

    //  Compare against the reference
    tod_btconv<4>(btc).perform(tc);
    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}

void btod_dirsum_test::test_ikjl_ij_kl_3b(bool rnd,
        double d) throw(libtest::test_exception) {

    //  c_{ikjl} = a_{ij} + b_{kl}
    // with splits and se_part

    std::stringstream tnss;
    tnss << "btod_dirsum_test::test_ikjl_ij_kl_3b("
            << rnd << ", " << d << ")";
    std::string tns = tnss.str();

    typedef std_allocator<double> allocator;

    cpu_pool cpus(1);

	try {

    size_t ni = 8, nj = 16, nk = 8, nl = 10;

    index<2> ia1, ia2;
    ia2[0] = ni - 1; ia2[1] = nj - 1;
    index<2> ib1, ib2;
    ib2[0] = nk - 1; ib2[1] = nl - 1;
    index<4> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nk - 1; ic2[2] = nj - 1; ic2[3] = nl - 1;
    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<2> dimb(index_range<2>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    block_index_space<2> bisa(dima), bisb(dimb);
    block_index_space<4> bisc(dimc);
    // split dimensions
    mask<2> msk1, msk2;
    mask<4> mskc;

    msk1[0] = true; msk2[1] = true;
    bisa.split(msk1, 2); bisa.split(msk1, 4); bisa.split(msk1, 6);
    bisa.split(msk2, 3); bisa.split(msk2, 8); bisa.split(msk2, 11);
    bisb.split(msk1, 2); bisb.split(msk1, 4); bisb.split(msk1, 6);
    bisb.split(msk2, 2); bisb.split(msk2, 5); bisb.split(msk2, 7);
    mskc[0] = true; mskc[1] = true;
    bisc.split(mskc, 2); bisc.split(mskc, 4); bisc.split(mskc, 6);
    mskc[0] = false; mskc[1] = false; mskc[2] = true;
    bisc.split(mskc, 3); bisc.split(mskc, 8); bisc.split(mskc, 11);
    mskc[2] = false; mskc[3] = true;
    bisc.split(mskc, 2); bisc.split(mskc, 5); bisc.split(mskc, 7);

    block_tensor<2, double, allocator> bta(bisa);
    block_tensor<2, double, allocator> btb(bisb);
    block_tensor<4, double, allocator> btc(bisc);
    symmetry<4, double> sym_ref(bisc);

    // add symmetries
    {
        block_tensor_ctrl<2, double> ctrla(bta), ctrlb(btb);
        block_tensor_ctrl<4, double> ctrlc(btc);
        msk1[1] = true;
        mskc[0] = true; mskc[1] = true; mskc[2] = true; mskc[3] = true;

        se_part<2, double> spa(bisa, msk1, 2), spb(bisb, msk1, 2);
        index<2> i00, i01, i10, i11;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;
        spa.add_map(i00, i11);
        spa.add_map(i01, i10);
        spb.add_map(i00, i11);
        spb.add_map(i01, i10);
        ctrla.req_symmetry().insert(spa);
        ctrlb.req_symmetry().insert(spb);

        se_part<4, double> spc(bisc, mskc, 2);
        index<4> i0000, i1111, i0001, i1110, i0010, i1101, i0011, i1100,
            i0100, i1011, i0101, i1010, i0110, i1001, i0111, i1000;
        i1110[0] = 1; i1110[1] = 1; i1110[2] = 1; i0001[3] = 1;
        i1101[0] = 1; i1101[1] = 1; i0010[2] = 1; i1101[3] = 1;
        i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
        i1011[0] = 1; i0100[1] = 1; i1011[2] = 1; i1011[3] = 1;
        i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
        i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
        i1000[0] = 1; i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
        i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;

        spc.add_map(i0000, i0101);
        spc.add_map(i0101, i1010);
        spc.add_map(i1010, i1111);
        spc.add_map(i0001, i0100);
        spc.add_map(i0100, i1011);
        spc.add_map(i1011, i1110);
        spc.add_map(i0010, i0111);
        spc.add_map(i0111, i1000);
        spc.add_map(i1000, i1101);
        spc.add_map(i0011, i0110);
        spc.add_map(i0110, i1001);
        spc.add_map(i1001, i1100);

        sym_ref.insert(spc);
        ctrlc.req_symmetry().insert(spc);
    }

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<2, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    //  Fill in random input

    btod_random<2>().perform(bta);
    btod_random<2>().perform(btb);

    if(rnd) btod_random<4>().perform(btc);
    tod_btconv<2>(bta).perform(ta);
    tod_btconv<2>(btb).perform(tb);
    if(d != 0.0) tod_btconv<4>(btc).perform(tc_ref);

    //  Generate reference data

	permutation<4> permc;
	permc.permute(1, 2);
	if(d == 0.0) {
		tod_dirsum<2, 2>(ta, 1.5, tb, -1.0, permc).
		    perform(cpus, true, 1.0, tc_ref);
	} else {
		tod_dirsum<2, 2>(ta, 1.5, tb, -1.0, permc).
		    perform(cpus, false, d, tc_ref);
	}

    //  Invoke the direct sum routine

    if(d == 0.0) {
        btod_dirsum<2, 2>(bta, 1.5, btb, -1.0, permc).perform(btc);
    } else {
        btod_dirsum<2, 2>(bta, 1.5, btb, -1.0, permc).perform(btc, scalar_transf<double>(d));
    }

    // Compare symmetry
    {
        block_tensor_ctrl<4, double> ctrlc(btc);
        compare_ref<4>::compare(tns.c_str(),
                ctrlc.req_const_symmetry(), sym_ref);
    }

    //  Compare against the reference
    tod_btconv<4>(btc).perform(tc);
    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}

void btod_dirsum_test::test_ikjl_ij_kl_3c(
        bool rnd, double d) throw(libtest::test_exception) {

    //  c_{ikjl} = a_{ij} + b_{kl}
    // with splits and se_label

    std::stringstream tnss;
    tnss << "btod_dirsum_test::test_ikjl_ij_kl_3c("
            << rnd << ", " << d << ")";
    std::string tns = tnss.str();

    typedef std_allocator<double> allocator;

    {
        std::vector<std::string> irnames(2);
        irnames[0] = "g"; irnames[1] = "u";
        point_group_table pg(tns, irnames, irnames[0]);
        pg.add_product(0, 0, 0);
        pg.add_product(0, 1, 1);
        pg.add_product(1, 1, 0);

        product_table_container::get_instance().add(pg);
    }

    cpu_pool cpus(1);

    try {

    size_t ni = 8, nj = 16, nk = 8, nl = 10;

    index<2> ia1, ia2;
    ia2[0] = ni - 1; ia2[1] = nj - 1;
    index<2> ib1, ib2;
    ib2[0] = nk - 1; ib2[1] = nl - 1;
    index<4> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nk - 1; ic2[2] = nj - 1; ic2[3] = nl - 1;
    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<2> dimb(index_range<2>(ib1, ib2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    block_index_space<2> bisa(dima), bisb(dimb);
    block_index_space<4> bisc(dimc);
    // split dimensions
    mask<2> msk1, msk2;
    mask<4> mskc;

    msk1[0] = true; msk2[1] = true;
    bisa.split(msk1, 2); bisa.split(msk1, 4); bisa.split(msk1, 6);
    bisa.split(msk2, 3); bisa.split(msk2, 8); bisa.split(msk2, 11);
    bisb.split(msk1, 2); bisb.split(msk1, 4); bisb.split(msk1, 6);
    bisb.split(msk2, 2); bisb.split(msk2, 5); bisb.split(msk2, 7);
    mskc[0] = true; mskc[1] = true;
    bisc.split(mskc, 2); bisc.split(mskc, 4); bisc.split(mskc, 6);
    mskc[0] = false; mskc[1] = false; mskc[2] = true;
    bisc.split(mskc, 3); bisc.split(mskc, 8); bisc.split(mskc, 11);
    mskc[2] = false; mskc[3] = true;
    bisc.split(mskc, 2); bisc.split(mskc, 5); bisc.split(mskc, 7);

    block_tensor<2, double, allocator> bta(bisa);
    block_tensor<2, double, allocator> btb(bisb);
    block_tensor<4, double, allocator> btc(bisc);
    symmetry<4, double> sym_ref(bisc);

    // add symmetries
    {
        block_tensor_ctrl<2, double> ctrla(bta), ctrlb(btb);
        block_tensor_ctrl<4, double> ctrlc(btc);
        msk1[0] = msk1[1] = true;
        mskc[0] = true; mskc[1] = true; mskc[2] = true; mskc[3] = true;

        se_label<2, double> sl(bisa.get_block_index_dims(), tns);
        block_labeling<2> &bl = sl.get_labeling();
        bl.assign(msk1, 0, 0);
        bl.assign(msk1, 1, 1);
        bl.assign(msk1, 2, 0);
        bl.assign(msk1, 3, 1);
        sl.set_rule(0);

        ctrla.req_symmetry().insert(sl);
        ctrlb.req_symmetry().insert(sl);

        se_label<4, double> slc(bisc.get_block_index_dims(), tns);
        block_labeling<4> &blc = slc.get_labeling();
        blc.assign(mskc, 0, 0);
        blc.assign(mskc, 1, 1);
        blc.assign(mskc, 2, 0);
        blc.assign(mskc, 3, 1);
        evaluation_rule<4> rc;
        sequence<4, size_t> sc1(0), sc2(0);
        sc1[0] = sc1[2] = 1;
        sc2[1] = sc2[3] = 1;
        rc.add_sequence(sc1);
        rc.add_sequence(sc2);
        rc.add_product(0, 0, 0);
        rc.add_product(1, 0, 0);
        slc.set_rule(rc);

        sym_ref.insert(slc);
        ctrlc.req_symmetry().insert(slc);

    }

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<2, double, allocator> tb(dimb);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    //  Fill in random input

    btod_random<2>().perform(bta);
    btod_random<2>().perform(btb);

    if(rnd) btod_random<4>().perform(btc);
    tod_btconv<2>(bta).perform(ta);
    tod_btconv<2>(btb).perform(tb);
    if(d != 0.0) tod_btconv<4>(btc).perform(tc_ref);

    //  Generate reference data

    permutation<4> permc;
    permc.permute(1, 2);
    if(d == 0.0) {
        tod_dirsum<2, 2>(ta, 1.5, tb, -1.0, permc).
            perform(cpus, true, 1.0, tc_ref);
    } else {
        tod_dirsum<2, 2>(ta, 1.5, tb, -1.0, permc).
            perform(cpus, false, d, tc_ref);
    }

    //  Invoke the direct sum routine

    if(d == 0.0) {
        btod_dirsum<2, 2>(bta, 1.5, btb, -1.0, permc).perform(btc);
    } else {
        btod_dirsum<2, 2>(bta, 1.5, btb, -1.0, permc).perform(btc, scalar_transf<double>(d));
    }

    // Compare symmetry
    {
        block_tensor_ctrl<4, double> ctrlc(btc);
        compare_ref<4>::compare(tns.c_str(),
                sym_ref, ctrlc.req_const_symmetry());
    }

    //  Compare against the reference
    tod_btconv<4>(btc).perform(tc);
    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, 1e-15);

    } catch(exception &e) {
        product_table_container::get_instance().erase(tns);

        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    } catch(...) {
        product_table_container::get_instance().erase(tns);

        throw;
    }

    product_table_container::get_instance().erase(tns);
}

void btod_dirsum_test::test_iklj_ij_kl_1(bool rnd, double d)
    throw(libtest::test_exception) {

    //  c_{iklj} = a_{ij} + a_{kl}
    // with splits and symmetry

    std::stringstream tnss;
    tnss << "btod_dirsum_test::test_iklj_ij_kl_1(" << rnd << ", " << d
        << ")";
    std::string tns = tnss.str();

    typedef std_allocator<double> allocator;

    cpu_pool cpus(1);

	try {

    size_t ni = 10, nj = 12;

    index<2> ia1, ia2;
    ia2[0] = ni - 1; ia2[1] = nj - 1;
    index<4> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = ni - 1; ic2[2] = nj - 1; ic2[3] = nj - 1;
    dimensions<2> dima(index_range<2>(ia1, ia2));
    dimensions<4> dimc(index_range<4>(ic1, ic2));
    block_index_space<2> bisa(dima);
    block_index_space<4> bisc(dimc);
    // split dimensions
    mask<2> mska;
    mask<4> mskc;

    mska[0] = true;
    bisa.split(mska, ni / 2);
    mska[0] = false; mska[1] = true;
    bisa.split(mska, nj / 2);
    mskc[0] = true;  mskc[1] = true;
    bisc.split(mskc, ni / 2);
    mskc[0] = false; mskc[1] = false;
    mskc[2] = true;  mskc[3] = true;
    bisc.split(mskc, nj / 2);
    bisc.match_splits();

    block_tensor<2, double, allocator> bta(bisa);
    block_tensor<4, double, allocator> btc(bisc);

    dense_tensor<2, double, allocator> ta(dima);
    dense_tensor<4, double, allocator> tc(dimc);
    dense_tensor<4, double, allocator> tc_ref(dimc);

    // Set symmetry
    {
        mska[0] = true; mska[1] = true;
        se_part<2, double> sp(bisa, mska, 2);
        index<2> i00, i01, i10, i11;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;
        sp.add_map(i00, i11);
        sp.add_map(i01, i10);

        block_tensor_ctrl<2, double> cbta(bta);
        cbta.req_symmetry().insert(sp);
    }
    //  Fill in random input

    btod_random<2>().perform(bta);

    // set zero blocks
    {
        index<2> idxa;
        idxa[1] = 1;
        block_tensor_ctrl<2, double> bctrla(bta);
        bctrla.req_zero_block(idxa);
    }

    if(rnd) btod_random<4>().perform(btc);
    tod_btconv<2>(bta).perform(ta);
    if(d != 0.0) tod_btconv<4>(btc).perform(tc_ref);

    //  Generate reference data

	permutation<4> permc;
	permc.permute(1, 2);
	if(d == 0.0) {
		tod_dirsum<2, 2>(ta, -1.0, ta, -1.0, permc).
		    perform(cpus, true, 1.0, tc_ref);
	} else {
		tod_dirsum<2, 2>(ta, -1.0, ta, -1.0, permc).
		    perform(cpus, false, d, tc_ref);
	}

    // Check symmetry

    btod_dirsum<2, 2> op(bta, -1.0, bta, -1.0, permc);

    {
        permutation<4> p1032;
        p1032.permute(0, 1).permute(2, 3);
        scalar_transf<double> tr0, tr1(-1.);
        const symmetry<4, double> &sym = op.get_symmetry();
        symmetry<4, double>::iterator is = sym.begin();
        for (; is != sym.end(); is++)
            if (sym.get_subset(is).get_id()
                    .compare(se_perm<4, double>::k_sym_type) == 0)
                break;

        if (is == sym.end())
            fail_test(tns.c_str(), __FILE__, __LINE__,
                    "Permutational symmetry missing.");

        const symmetry_element_set<4, double> &set = sym.get_subset(is);
        symmetry_element_set_adapter<4, double, se_perm<4, double> > adapter(set);
        permutation_group<4, double> grp(set);
        if (! grp.is_member(tr0, p1032)) {
            fail_test(tns.c_str(), __FILE__, __LINE__,
                    "Permutational symmetry () missing.");
        }
    }

    //  Invoke the direct sum routine

    if(d == 0.0) {
        op.perform(btc);
    } else {
        op.perform(btc, scalar_transf<double>(d));
    }
    tod_btconv<4>(btc).perform(tc);

    //  Compare against the reference

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
