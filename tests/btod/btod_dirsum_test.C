#include <cmath>
#include <ctime>
#include <sstream>
#include <libvmm/std_allocator.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/core/tensor.h>
#include <libtensor/btod/btod_dirsum.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/se_label.h>
#include <libtensor/symmetry/se_part.h>
#include <libtensor/tod/tod_btconv.h>
#include "../compare_ref.h"
#include "btod_dirsum_test.h"
#include <libtensor/btod/btod_print.h>

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

	test_ikjl_ij_kl_3(false, false, true);
	test_ikjl_ij_kl_3(false, true, true);
	test_ikjl_ij_kl_3(true, false, true);
	test_ikjl_ij_kl_3(true, true, true);
	test_ikjl_ij_kl_3(false, false, true, 2.0);
	test_ikjl_ij_kl_3(false, true, true, 2.0);
	test_ikjl_ij_kl_3(true, false, true, 2.0);
	test_ikjl_ij_kl_3(true, true, true, 2.0);
	test_ikjl_ij_kl_3(false, false, false);
	test_ikjl_ij_kl_3(false, true, false);
	test_ikjl_ij_kl_3(true, false, false);
	test_ikjl_ij_kl_3(true, true, false);
	test_ikjl_ij_kl_3(false, false, false, -2.0);
	test_ikjl_ij_kl_3(false, true, false, -2.0);
	test_ikjl_ij_kl_3(true, false, false, -2.0);
	test_ikjl_ij_kl_3(true, true, false, -2.0);
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

void btod_dirsum_test::test_ikjl_ij_kl_3(bool part, bool label,
		bool rnd, double d) throw(libtest::test_exception) {

	//	c_{ikjl} = a_{ij} + b_{kl}
	// with splits, se_label, and se_part

	std::stringstream tnss;
	tnss << "btod_dirsum_test::test_ikjl_ij_kl_3(" << part << ", " << label
			<< ", " << rnd << ", " << d << ")";
	std::string tns = tnss.str();

	if (label) {
		point_group_table pg(tns, 2);
		pg.add_product(0, 0, 0);
		pg.add_product(0, 1, 1);
		pg.add_product(1, 1, 0);

		product_table_container::get_instance().add(pg);
	}

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
	if (label || part) {
		block_tensor_ctrl<2, double> ctrla(bta), ctrlb(btb);
		block_tensor_ctrl<4, double> ctrlc(btc);
		msk1[1] = true;
		mskc[0] = true; mskc[1] = true; mskc[2] = true; mskc[3] = true;

		if (label) {
			se_label<2, double> sl(bisa.get_block_index_dims(), tns);
			sl.assign(msk1, 0, 0);
			sl.assign(msk1, 1, 1);
			sl.assign(msk1, 2, 0);
			sl.assign(msk1, 3, 1);
			sl.add_target(0);

			ctrla.req_symmetry().insert(sl);
			ctrlb.req_symmetry().insert(sl);

			se_label<4, double> slc(bisc.get_block_index_dims(), tns);
			slc.assign(mskc, 0, 0);
			slc.assign(mskc, 1, 1);
			slc.assign(mskc, 2, 0);
			slc.assign(mskc, 3, 1);
			slc.add_target(0);
			slc.add_target(1);

			sym_ref.insert(slc);
			ctrlc.req_symmetry().insert(slc);
		}
		if (part) {
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

			spc.add_map(i0000, i1010); spc.add_map(i0001, i1011);
			spc.add_map(i0100, i1110); spc.add_map(i0101, i1111);
			spc.add_map(i0010, i1000); spc.add_map(i0011, i1001);
			spc.add_map(i0110, i1100); spc.add_map(i0111, i1101);
			spc.add_map(i0000, i0101); spc.add_map(i0010, i0111);
			spc.add_map(i1000, i1101); spc.add_map(i1010, i1111);
			spc.add_map(i0001, i0100); spc.add_map(i0011, i0110);
			spc.add_map(i1001, i1100); spc.add_map(i1011, i1110);

			sym_ref.insert(spc);
			ctrlc.req_symmetry().insert(spc);
		}
	}

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

	// Compare symmetry
	if (label || part) {
		block_tensor_ctrl<4, double> ctrlc(btc);
		compare_ref<4>::compare(tns.c_str(), sym_ref, ctrlc.req_const_symmetry());
	}

	//	Compare against the reference
	tod_btconv<4>(btc).perform(tc);
	compare_ref<4>::compare(tns.c_str(), tc, tc_ref, 1e-15);

	} catch(exception &e) {
		if (label) product_table_container::get_instance().erase(tns);

		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		if (label) product_table_container::get_instance().erase(tns);

		throw;
	}

	if (label) product_table_container::get_instance().erase(tns);
}

} // namespace libtensor
