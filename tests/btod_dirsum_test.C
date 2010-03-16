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

	//~ test_ij_i_j_1();
	//~ test_ij_i_j_1(-0.5);

	test_ikjl_ij_kl_1();
	//~ test_ikjl_ij_kl_1(2.0);

}


void btod_dirsum_test::test_ij_i_j_1(double d) throw(libtest::test_exception) {

	//	c_{ij} = a_i + b_j

	std::stringstream tnss;
	tnss << "btod_dirsum_test::test_ij_i_j_1(" << d << ")";
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
	btod_random<2>().perform(btc);
	tod_btconv<1>(bta).perform(ta);
	tod_btconv<1>(btb).perform(tb);
	if(d != 0.0) tod_btconv<2>(btc).perform(tc_ref);

	//	Generate reference data

	if(d == 0.0) tod_dirsum<1, 1>(ta, 1.0, tb, 1.0).perform(tc_ref);
	else tod_dirsum<1, 1>(ta, 1.0, tb, 1.0).perform(tc_ref, d);

	//	Invoke the contraction routine

	if(d == 0.0) btod_dirsum<1, 1>(bta, 1.0, btb, 1.0).perform(btc);
	else btod_dirsum<1, 1>(bta, 1.0, btb, 1.0).perform(btc, d);
	tod_btconv<2>(btc).perform(tc);

	//	Compare against the reference

	compare_ref<2>::compare(tns.c_str(), tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


void btod_dirsum_test::test_ikjl_ij_kl_1(double d)
	throw(libtest::test_exception) {

	//	c_{ikjl} = a_{ij} + b_{kl}

	std::stringstream tnss;
	tnss << "btod_dirsum_test::test_ikjl_ij_kl_1(" << d << ")";
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
	btod_random<4>().perform(btc);
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

	//	Invoke the contraction routine

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
