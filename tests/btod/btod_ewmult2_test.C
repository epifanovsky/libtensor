#include <iomanip>
#include <sstream>
#include <libvmm/std_allocator.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/btod/btod_ewmult2.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/tod/tod_btconv.h>
#include <libtensor/tod/tod_ewmult2.h>
#include "btod_ewmult2_test.h"
#include "../compare_ref.h"

namespace libtensor {


void btod_ewmult2_test::perform() throw(libtest::test_exception) {

	test_1(false);
	test_1(true);
	test_2(false);
	test_2(true);
	test_3(false);
	test_3(true);
	test_4(false);
	test_4(true);
	test_5(false);
	test_5(true);
	test_6(false);
	test_6(true);
}


/**	\test $c_{i} = a_{i} b_{i}$
 **/
void btod_ewmult2_test::test_1(bool doadd) throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "btod_ewmult2_test::test_1(" << doadd << ")";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<1> ia1, ia2;
	index<1> ib1, ib2;
	index<1> ic1, ic2;
	ia2[0] = 9;
	ib2[0] = 9;
	ic2[0] = 9;
	dimensions<1> dimsa(index_range<1>(ia1, ia2));
	dimensions<1> dimsb(index_range<1>(ib1, ib2));
	dimensions<1> dimsc(index_range<1>(ic1, ic2));
	block_index_space<1> bisa(dimsa);
	block_index_space<1> bisb(dimsb);
	block_index_space<1> bisc(dimsc);
	mask<1> m;
	m[0] = true;
	bisa.split(m, 3);
	bisb.split(m, 3);
	bisc.split(m, 3);

	block_tensor<1, double, allocator_t> bta(bisa);
	block_tensor<1, double, allocator_t> btb(bisb);
	block_tensor<1, double, allocator_t> btc(bisc);
	tensor<1, double, allocator_t> ta(dimsa);
	tensor<1, double, allocator_t> tb(dimsb);
	tensor<1, double, allocator_t> tc(dimsc), tc_ref(dimsc);

	//	Fill in random data

	btod_random<1>().perform(bta);
	btod_random<1>().perform(btb);
	btod_random<1>().perform(btc);
	bta.set_immutable();
	btb.set_immutable();

	//	Prepare the reference

	tod_btconv<1>(bta).perform(ta);
	tod_btconv<1>(btb).perform(tb);
	tod_btconv<1>(btc).perform(tc_ref);

	//	Invoke the operation

	double d = drand48();
	btod_ewmult2<0, 0, 1> op(bta, btb);
	if(!op.get_bis().equals(bisc)) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, "Bad bis.");
	}
	if(doadd) {
		tod_ewmult2<0, 0, 1>(ta, tb).perform(tc_ref, d);
		op.perform(btc, d);
	} else {
		tod_ewmult2<0, 0, 1>(ta, tb).perform(tc_ref);
		op.perform(btc);
	}
	tod_btconv<1>(btc).perform(tc);

	//	Compare against the reference

	compare_ref<1>::compare(tnss.str().c_str(), tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


/**	\test $c_{ij} = a_{ij} b_{ij}$
 **/
void btod_ewmult2_test::test_2(bool doadd) throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "btod_ewmult2_test::test_2(" << doadd << ")";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> ia1, ia2;
	index<2> ib1, ib2;
	index<2> ic1, ic2;
	ia2[0] = 9; ia2[1] = 5;
	ib2[0] = 9; ib2[1] = 5;
	ic2[0] = 9; ic2[1] = 5;
	dimensions<2> dimsa(index_range<2>(ia1, ia2));
	dimensions<2> dimsb(index_range<2>(ib1, ib2));
	dimensions<2> dimsc(index_range<2>(ic1, ic2));
	block_index_space<2> bisa(dimsa);
	block_index_space<2> bisb(dimsb);
	block_index_space<2> bisc(dimsc);
	mask<2> m01, m10;
	m10[0] = true; m01[1] = true;
	bisa.split(m01, 3);
	bisb.split(m01, 3);
	bisc.split(m01, 3);
	bisa.split(m10, 3);
	bisb.split(m10, 3);
	bisc.split(m10, 3);

	block_tensor<2, double, allocator_t> bta(bisa);
	block_tensor<2, double, allocator_t> btb(bisb);
	block_tensor<2, double, allocator_t> btc(bisc);
	tensor<2, double, allocator_t> ta(dimsa);
	tensor<2, double, allocator_t> tb(dimsb);
	tensor<2, double, allocator_t> tc(dimsc), tc_ref(dimsc);

	//	Fill in random data

	btod_random<2>().perform(bta);
	btod_random<2>().perform(btb);
	btod_random<2>().perform(btc);
	bta.set_immutable();
	btb.set_immutable();

	//	Prepare the reference

	tod_btconv<2>(bta).perform(ta);
	tod_btconv<2>(btb).perform(tb);
	tod_btconv<2>(btc).perform(tc_ref);

	//	Invoke the operation

	double d = drand48();
	permutation<2> perma;
	permutation<2> permb;
	permutation<2> permc;
	btod_ewmult2<0, 0, 2> op(bta, perma, btb, permb, permc);
	if(!op.get_bis().equals(bisc)) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, "Bad bis.");
	}
	if(doadd) {
		tod_ewmult2<0, 0, 2>(ta, perma, tb, permb, permc).
			perform(tc_ref, d);
		op.perform(btc, d);
	} else {
		tod_ewmult2<0, 0, 2>(ta, perma, tb, permb, permc).
			perform(tc_ref);
		op.perform(btc);
	}
	tod_btconv<2>(btc).perform(tc);

	//	Compare against the reference

	compare_ref<2>::compare(tnss.str().c_str(), tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


/**	\test $c_{ij} = a_{ij} b_{ji}$
 **/
void btod_ewmult2_test::test_3(bool doadd) throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "btod_ewmult2_test::test_3(" << doadd << ")";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> ia1, ia2;
	index<2> ib1, ib2;
	index<2> ic1, ic2;
	ia2[0] = 9; ia2[1] = 5;
	ib2[0] = 5; ib2[1] = 9;
	ic2[0] = 9; ic2[1] = 5;
	dimensions<2> dimsa(index_range<2>(ia1, ia2));
	dimensions<2> dimsb(index_range<2>(ib1, ib2));
	dimensions<2> dimsc(index_range<2>(ic1, ic2));
	block_index_space<2> bisa(dimsa);
	block_index_space<2> bisb(dimsb);
	block_index_space<2> bisc(dimsc);
	mask<2> m01, m10;
	m10[0] = true; m01[1] = true;
	bisa.split(m01, 3);
	bisb.split(m01, 3);
	bisc.split(m01, 3);
	bisa.split(m10, 3);
	bisb.split(m10, 3);
	bisc.split(m10, 3);

	block_tensor<2, double, allocator_t> bta(bisa);
	block_tensor<2, double, allocator_t> btb(bisb);
	block_tensor<2, double, allocator_t> btc(bisc);
	tensor<2, double, allocator_t> ta(dimsa);
	tensor<2, double, allocator_t> tb(dimsb);
	tensor<2, double, allocator_t> tc(dimsc), tc_ref(dimsc);

	//	Fill in random data

	btod_random<2>().perform(bta);
	btod_random<2>().perform(btb);
	btod_random<2>().perform(btc);
	bta.set_immutable();
	btb.set_immutable();

	//	Prepare the reference

	tod_btconv<2>(bta).perform(ta);
	tod_btconv<2>(btb).perform(tb);
	tod_btconv<2>(btc).perform(tc_ref);

	//	Invoke the operation

	double d = drand48();
	permutation<2> perma;
	permutation<2> permb; permb.permute(0, 1);
	permutation<2> permc;
	btod_ewmult2<0, 0, 2> op(bta, perma, btb, permb, permc);
	if(!op.get_bis().equals(bisc)) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, "Bad bis.");
	}
	if(doadd) {
		tod_ewmult2<0, 0, 2>(ta, perma, tb, permb, permc).
			perform(tc_ref, d);
		op.perform(btc, d);
	} else {
		tod_ewmult2<0, 0, 2>(ta, perma, tb, permb, permc).
			perform(tc_ref);
		op.perform(btc);
	}
	tod_btconv<2>(btc).perform(tc);

	//	Compare against the reference

	compare_ref<2>::compare(tnss.str().c_str(), tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


/**	\test $c_{ijkl} = a_{kj} b_{ikl}$
 **/
void btod_ewmult2_test::test_4(bool doadd) throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "btod_ewmult2_test::test_4(" << doadd << ")";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> ia1, ia2;
	index<3> ib1, ib2;
	index<4> ic1, ic2;
	ia2[0] = 9; ia2[1] = 5;
	ib2[0] = 5; ib2[1] = 9; ib2[2] = 9;
	ic2[0] = 5; ic2[1] = 5; ic2[2] = 9; ic2[3] = 9;
	dimensions<2> dimsa(index_range<2>(ia1, ia2));
	dimensions<3> dimsb(index_range<3>(ib1, ib2));
	dimensions<4> dimsc(index_range<4>(ic1, ic2));
	block_index_space<2> bisa(dimsa);
	block_index_space<3> bisb(dimsb);
	block_index_space<4> bisc(dimsc);
	mask<2> m01, m10;
	mask<3> m011, m100;
	mask<4> m0011, m0100, m1000;
	m10[0] = true; m01[1] = true;
	m100[0] = true; m011[1] = true; m011[2] = true;
	m1000[0] = true; m0100[1] = true; m0011[2] = true; m0011[3] = true;
	bisa.split(m01, 3);
	bisa.split(m10, 3);
	bisa.split(m10, 5);
	bisb.split(m011, 3);
	bisb.split(m011, 5);
	bisb.split(m100, 3);
	bisc.split(m0011, 3);
	bisc.split(m0011, 5);
	bisc.split(m0100, 3);
	bisc.split(m1000, 3);

	block_tensor<2, double, allocator_t> bta(bisa);
	block_tensor<3, double, allocator_t> btb(bisb);
	block_tensor<4, double, allocator_t> btc(bisc);
	tensor<2, double, allocator_t> ta(dimsa);
	tensor<3, double, allocator_t> tb(dimsb);
	tensor<4, double, allocator_t> tc(dimsc), tc_ref(dimsc);

	//	Fill in random data

	btod_random<2>().perform(bta);
	btod_random<3>().perform(btb);
	btod_random<4>().perform(btc);
	bta.set_immutable();
	btb.set_immutable();

	//	Prepare the reference

	tod_btconv<2>(bta).perform(ta);
	tod_btconv<3>(btb).perform(tb);
	tod_btconv<4>(btc).perform(tc_ref);

	//	Invoke the operation

	double d = drand48();
	permutation<2> perma; perma.permute(0, 1); // kj->jk
	permutation<3> permb; permb.permute(1, 2); // ikl->ilk
	permutation<4> permc; permc.permute(0, 1).permute(2, 3); // jilk->ijkl
	btod_ewmult2<1, 2, 1> op(bta, perma, btb, permb, permc);
	if(!op.get_bis().equals(bisc)) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, "Bad bis.");
	}
	if(doadd) {
		tod_ewmult2<1, 2, 1>(ta, perma, tb, permb, permc).
			perform(tc_ref, d);
		op.perform(btc, d);
	} else {
		tod_ewmult2<1, 2, 1>(ta, perma, tb, permb, permc).
			perform(tc_ref);
		op.perform(btc);
	}
	tod_btconv<4>(btc).perform(tc);

	//	Compare against the reference

	compare_ref<4>::compare(tnss.str().c_str(), tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


/**	\test $c_{ijkl} = a_{ljk} b_{jil}$
 **/
void btod_ewmult2_test::test_5(bool doadd) throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "btod_ewmult2_test::test_5(" << doadd << ")";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<3> ia1, ia2;
	index<3> ib1, ib2;
	index<4> ic1, ic2;
	ia2[0] = 9; ia2[1] = 9; ia2[2] = 5;
	ib2[0] = 9; ib2[1] = 5; ib2[2] = 9;
	ic2[0] = 5; ic2[1] = 9; ic2[2] = 5; ic2[3] = 9;
	dimensions<3> dimsa(index_range<3>(ia1, ia2));
	dimensions<3> dimsb(index_range<3>(ib1, ib2));
	dimensions<4> dimsc(index_range<4>(ic1, ic2));
	block_index_space<3> bisa(dimsa);
	block_index_space<3> bisb(dimsb);
	block_index_space<4> bisc(dimsc), bisc_ref(dimsc);
	mask<3> m001, m010, m101, m110;
	mask<4> m0010, m0101, m1000, m1010;
	m110[0] = true; m110[1] = true; m001[2] = true;
	m101[0] = true; m010[1] = true; m101[2] = true;
	m1000[0] = true; m0010[2] = true;
	m1010[0] = true; m0101[1] = true; m1010[2] = true; m0101[3] = true;
	bisa.split(m001, 3);
	bisa.split(m110, 3);
	bisa.split(m110, 5);
	bisb.split(m010, 3);
	bisb.split(m101, 3);
	bisb.split(m101, 5);
	bisc.split(m0101, 3);
	bisc.split(m0101, 5);
	bisc.split(m1010, 3);
	bisc_ref.split(m0101, 3);
	bisc_ref.split(m0101, 5);
	bisc_ref.split(m1010, 3);
	bisc_ref.split(m1000, 3);

	block_tensor<3, double, allocator_t> bta(bisa);
	block_tensor<3, double, allocator_t> btb(bisb);
	block_tensor<4, double, allocator_t> btc(bisc);
	tensor<3, double, allocator_t> ta(dimsa);
	tensor<3, double, allocator_t> tb(dimsb);
	tensor<4, double, allocator_t> tc(dimsc), tc_ref(dimsc);

	//	Fill in random data

	btod_random<3>().perform(bta);
	btod_random<3>().perform(btb);
	btod_random<4>().perform(btc);
	bta.set_immutable();
	btb.set_immutable();

	//	Prepare the reference

	tod_btconv<3>(bta).perform(ta);
	tod_btconv<3>(btb).perform(tb);
	tod_btconv<4>(btc).perform(tc_ref);

	//	Invoke the operation

	double d = drand48();
	permutation<3> perma; perma.permute(1, 2).permute(0, 1); // ljk->klj
	permutation<3> permb; permb.permute(0, 1).permute(1, 2); // jil->ilj
	permutation<4> permc;
	permc.permute(2, 3).permute(1, 2).permute(0, 2); // kilj->ijkl
	btod_ewmult2<1, 1, 2> op(bta, perma, btb, permb, permc);
	if(!op.get_bis().equals(bisc_ref)) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, "Bad bis.");
	}
	if(doadd) {
		tod_ewmult2<1, 1, 2>(ta, perma, tb, permb, permc).
			perform(tc_ref, d);
		op.perform(btc, d);
	} else {
		tod_ewmult2<1, 1, 2>(ta, perma, tb, permb, permc).
			perform(tc_ref);
		op.perform(btc);
	}
	tod_btconv<4>(btc).perform(tc);

	//	Compare against the reference

	compare_ref<4>::compare(tnss.str().c_str(), tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


/**	\test $c_{ijkl} = a_{ikl} b_{jkl}$, perm symmetry
 **/
void btod_ewmult2_test::test_6(bool doadd) throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "btod_ewmult2_test::test_6(" << doadd << ")";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	size_t ni = 10, nj = 10, nk = 8, nl = 8;
	index<3> ia1, ia2;
	index<3> ib1, ib2;
	index<4> ic1, ic2;
	ia2[0] = ni - 1; ia2[1] = nk - 1; ia2[2] = nl - 1;
	ib2[0] = nj - 1; ib2[1] = nk - 1; ib2[2] = nl - 1;
	ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1; ic2[3] = nl - 1;
	dimensions<3> dimsa(index_range<3>(ia1, ia2));
	dimensions<3> dimsb(index_range<3>(ib1, ib2));
	dimensions<4> dimsc(index_range<4>(ic1, ic2));
	block_index_space<3> bisa(dimsa);
	block_index_space<3> bisb(dimsb);
	block_index_space<4> bisc(dimsc), bisc_ref(dimsc);
	mask<3> m011, m100;
	mask<4> m0011, m0100, m1000, m1100;
	m100[0] = true; m011[1] = true; m011[2] = true;
	m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;
	m1000[0] = true; m0100[1] = true;
	bisa.split(m011, 3);
	bisa.split(m100, 3);
	bisa.split(m100, 6);
	bisb.split(m011, 3);
	bisb.split(m100, 3);
	bisb.split(m100, 6);
	bisc.split(m0011, 3);
	bisc.split(m1100, 3);
	bisc.split(m1100, 6);
	bisc_ref.split(m0011, 3);
	bisc_ref.split(m1000, 3);
	bisc_ref.split(m1000, 6);
	bisc_ref.split(m0100, 3);
	bisc_ref.split(m0100, 6);

	block_tensor<3, double, allocator_t> bta(bisa);
	block_tensor<3, double, allocator_t> btb(bisb);
	block_tensor<4, double, allocator_t> btc(bisc);
	symmetry<4, double> symc(bisc), symc_ref(bisc);
	tensor<3, double, allocator_t> ta(dimsa);
	tensor<3, double, allocator_t> tb(dimsb);
	tensor<4, double, allocator_t> tc(dimsc), tc_ref(dimsc);

	//	Set up symmetry

	{
		block_tensor_ctrl<3, double> ca(bta);
		se_perm<3, double> seperm(permutation<3>().permute(1, 2), true);
		ca.req_symmetry().insert(seperm);
	}
	{
		block_tensor_ctrl<3, double> cb(btb);
		se_perm<3, double> seperm(permutation<3>().permute(1, 2), true);
		cb.req_symmetry().insert(seperm);
	}
	{
		se_perm<4, double> seperm(permutation<4>().permute(2, 3), true);
		symc_ref.insert(seperm);
	}

	//	Fill in random data

	btod_random<3>().perform(bta);
	btod_random<3>().perform(btb);
	btod_random<4>().perform(btc);
	bta.set_immutable();
	btb.set_immutable();

	//	Prepare the reference

	tod_btconv<3>(bta).perform(ta);
	tod_btconv<3>(btb).perform(tb);
	tod_btconv<4>(btc).perform(tc_ref);

	//	Invoke the operation

	double d = drand48();
	permutation<3> perma;
	permutation<3> permb;
	permutation<4> permc;
	btod_ewmult2<1, 1, 2> op(bta, perma, btb, permb, permc);
	if(!op.get_bis().equals(bisc_ref)) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, "Bad bis.");
	}
	if(doadd) {
		tod_ewmult2<1, 1, 2>(ta, perma, tb, permb, permc).
			perform(tc_ref, d);
		op.perform(btc, d);
	} else {
		tod_ewmult2<1, 1, 2>(ta, perma, tb, permb, permc).
			perform(tc_ref);
		op.perform(btc);
	}
	tod_btconv<4>(btc).perform(tc);

	//	Compare against the reference

	{
		block_tensor_ctrl<4, double> cc(btc);
		so_copy<4, double>(cc.req_const_symmetry()).perform(symc);
	}

	compare_ref<4>::compare(tnss.str().c_str(), symc, symc_ref);
	compare_ref<4>::compare(tnss.str().c_str(), tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
