#include <cmath>
#include <ctime>
#include <sstream>
#include <libvmm/std_allocator.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/core/tensor.h>
#include <libtensor/btod/btod_dirsum.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/tod/tod_btconv.h>
#include "compare_ref.h"
#include "btod_dirsum_test.h"

namespace libtensor {

typedef libvmm::std_allocator<double> allocator;

void btod_dirsum_test::perform() throw(libtest::test_exception) {

	srand48(time(0));

	test_ij_i_j_1(true);
	test_ij_i_j_1(true, -0.5);
	test_ij_i_j_1(false);
	test_ij_i_j_1(false, 0.5);

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
}


void btod_dirsum_test::test_ij_i_j_1(bool rnd, double d)
	throw(libtest::test_exception) {

	//	c_{ij} = a_i + b_j

	std::stringstream tnss;
	tnss << "btod_dirsum_test::test_ij_i_j_1(" << rnd << ", " << d << ")";
	std::string tns = tnss.str();

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

	tensor<1, double, allocator> ta(dima);
	tensor<1, double, allocator> tb(dimb);
	tensor<2, double, allocator> tc(dimc);
	tensor<2, double, allocator> tc_ref(dimc);

	//	Fill in random input

	btod_random<1>().perform(bta);
	btod_random<1>().perform(btb);
	if(rnd) btod_random<2>().perform(btc);
	tod_btconv<1>(bta).perform(ta);
	tod_btconv<1>(btb).perform(tb);
	if(d != 0.0) tod_btconv<2>(btc).perform(tc_ref);

	//	Generate reference data

	if(d == 0.0) tod_dirsum<1, 1>(ta, 1.0, tb, 1.0).perform(tc_ref);
	else tod_dirsum<1, 1>(ta, 1.0, tb, 1.0).perform(tc_ref, d);

	//	Invoke the direct sum routine

	if(d == 0.0) btod_dirsum<1, 1>(bta, 1.0, btb, 1.0).perform(btc);
	else btod_dirsum<1, 1>(bta, 1.0, btb, 1.0).perform(btc, d);
	tod_btconv<2>(btc).perform(tc);

	//	Compare against the reference

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

	tensor<2, double, allocator> ta(dima);
	tensor<1, double, allocator> tb(dimb);
	tensor<3, double, allocator> tc(dimc);
	tensor<3, double, allocator> tc_ref(dimc);

	//	Fill in random input

	btod_random<2>().perform(bta);
	btod_random<1>().perform(btb);
	if(rnd) btod_random<3>().perform(btc);
	tod_btconv<2>(bta).perform(ta);
	tod_btconv<1>(btb).perform(tb);
	if(d != 0.0) tod_btconv<3>(btc).perform(tc_ref);

	//	Generate reference data

	if(d == 0.0) {
		tod_dirsum<2, 1>(ta, 1.5, tb, 1.0).perform(tc_ref);
	} else {
		tod_dirsum<2, 1>(ta, 1.5, tb, 1.0).perform(tc_ref, d);
	}

	//	Invoke the direct sum routine

	if(d == 0.0) {
		btod_dirsum<2, 1>(bta, 1.5, btb, 1.0).perform(btc);
	} else {
		btod_dirsum<2, 1>(bta, 1.5, btb, 1.0).perform(btc, d);
	}
	tod_btconv<3>(btc).perform(tc);

	//	Compare against the reference

	compare_ref<3>::compare(tns.c_str(), tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void btod_dirsum_test::test_ikjl_ij_kl_1(bool rnd, double d)
	throw(libtest::test_exception) {

	//	c_{ikjl} = a_{ij} + b_{kl}

	std::stringstream tnss;
	tnss << "btod_dirsum_test::test_ikjl_ij_kl_1(" << rnd << ", " << d
		<< ")";
	std::string tns = tnss.str();

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

	tensor<2, double, allocator> ta(dima);
	tensor<2, double, allocator> tb(dimb);
	tensor<4, double, allocator> tc(dimc);
	tensor<4, double, allocator> tc_ref(dimc);

	//	Fill in random input

	btod_random<2>().perform(bta);
	btod_random<2>().perform(btb);
	if(rnd) btod_random<4>().perform(btc);
	tod_btconv<2>(bta).perform(ta);
	tod_btconv<2>(btb).perform(tb);
	if(d != 0.0) tod_btconv<4>(btc).perform(tc_ref);

	//	Generate reference data

	permutation<4> permc;
	permc.permute(1, 2);
	if(d == 0.0) {
		tod_dirsum<2, 2>(ta, 1.5, tb, -1.0, permc).perform(tc_ref);
	} else {
		tod_dirsum<2, 2>(ta, 1.5, tb, -1.0, permc).perform(tc_ref, d);
	}

	//	Invoke the direct sum routine

	if(d == 0.0) {
		btod_dirsum<2, 2>(bta, 1.5, btb, -1.0, permc).perform(btc);
	} else {
		btod_dirsum<2, 2>(bta, 1.5, btb, -1.0, permc).perform(btc, d);
	}
	tod_btconv<4>(btc).perform(tc);

	//	Compare against the reference

	compare_ref<4>::compare(tns.c_str(), tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void btod_dirsum_test::test_ikjl_ij_kl_2(bool rnd, double d)
	throw(libtest::test_exception) {

	//	c_{ikjl} = a_{ij} + b_{kl}
	// with splits

	std::stringstream tnss;
	tnss << "btod_dirsum_test::test_ikjl_ij_kl_2(" << rnd << ", " << d
		<< ")";
	std::string tns = tnss.str();

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

	tensor<2, double, allocator> ta(dima);
	tensor<2, double, allocator> tb(dimb);
	tensor<4, double, allocator> tc(dimc);
	tensor<4, double, allocator> tc_ref(dimc);

	//	Fill in random input

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

	//	Generate reference data

	permutation<4> permc;
	permc.permute(1, 2);
	if(d == 0.0) {
		tod_dirsum<2, 2>(ta, 1.5, tb, -1.0, permc).perform(tc_ref);
	} else {
		tod_dirsum<2, 2>(ta, 1.5, tb, -1.0, permc).perform(tc_ref, d);
	}

	//	Invoke the direct sum routine

	if(d == 0.0) {
		btod_dirsum<2, 2>(bta, 1.5, btb, -1.0, permc).perform(btc);
	} else {
		btod_dirsum<2, 2>(bta, 1.5, btb, -1.0, permc).perform(btc, d);
	}
	tod_btconv<4>(btc).perform(tc);

	//	Compare against the reference

	compare_ref<4>::compare(tns.c_str(), tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}

} // namespace libtensor
