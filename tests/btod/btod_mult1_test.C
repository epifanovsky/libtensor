#include <libvmm/std_allocator.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/btod/btod_mult1.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/tod/tod_btconv.h>
#include <libtensor/tod/tod_mult1.h>
#include "btod_mult1_test.h"
#include "../compare_ref.h"

namespace libtensor {


void btod_mult1_test::perform() throw(libtest::test_exception) {

	test_1(false, false); test_1(false, true);
	test_1(true, false);  test_1(true, true);
	test_2(false, false); test_2(false, true);
	test_2(true, false);  test_2(true, true);
	test_3(false, false); test_3(false, true);
	test_3(true, false);  test_3(true, true);
	test_4(false, false); test_4(false, true);
	test_4(true, false);  test_4(true, true);
	test_5(false, false); test_5(false, true);
	test_5(true, false);  test_5(true, true);
}


/**	\test Elementwise operation of two order-2 tensors with no symmetry
		and no zero blocks.
 **/
void btod_mult1_test::test_1(
		bool recip, bool doadd) throw(libtest::test_exception) {

	std::ostringstream oss;
	oss << "btod_mult1_test::test_1("
			<< (recip ? "true" : "false") << ","
			<< (doadd ? "true" : "false") << ")";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	dimensions<2> bidims(bis.get_block_index_dims());

	block_tensor<2, double, allocator_t> bta(bis), btb(bis);
	tensor<2, double, allocator_t> ta(dims), tb(dims), ta_ref(dims);

	//	Fill in random data

	btod_random<2>().perform(bta);
	btod_random<2>().perform(btb);
	btb.set_immutable();

	//	Prepare the reference

	tod_btconv<2>(bta).perform(ta_ref);
	tod_btconv<2>(btb).perform(tb);

	//	Invoke the operation

	if (doadd) {
		tod_mult1<2>(tb, recip).perform(ta_ref, 0.2);
		btod_mult1<2>(btb, recip).perform(bta, 0.2);
	}
	else {
		tod_mult1<2>(tb, recip).perform(ta_ref);
		btod_mult1<2>(btb, recip).perform(bta);
	}
	tod_btconv<2>(bta).perform(ta);

	//	Compare against the reference

	compare_ref<2>::compare(oss.str().c_str(), ta, ta_ref, 1e-15);

	} catch(exception &e) {
		fail_test(oss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


/**	\test Elementwise operation of two order-2 tensors
		with no symmetry and no zero blocks, second tensor permuted.
 **/
void btod_mult1_test::test_2(
		bool recip, bool doadd) throw(libtest::test_exception) {

	std::ostringstream oss;
	oss << "btod_mult1_test::test_2("
			<< (recip ? "true" : "false") << ","
			<< (doadd ? "true" : "false") << ")";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	dimensions<2> bidims(bis.get_block_index_dims());
	permutation<2> p10;
	p10.permute(0, 1);

	block_tensor<2, double, allocator_t> bta(bis), btb(bis);
	tensor<2, double, allocator_t> ta(dims), tb(dims), ta_ref(dims);

	//	Fill in random data

	btod_random<2>().perform(bta);
	btod_random<2>().perform(btb);
	btb.set_immutable();

	//	Prepare the reference

	tod_btconv<2>(bta).perform(ta_ref);
	tod_btconv<2>(btb).perform(tb);

	//	Invoke the operation

	if (doadd) {
		tod_mult1<2>(tb, p10, recip).perform(ta_ref, 0.5);
		btod_mult1<2>(btb, p10, recip).perform(bta, 0.5);
	}
	else {
		tod_mult1<2>(tb, p10, recip).perform(ta_ref);
		btod_mult1<2>(btb, p10, recip).perform(bta);
	}
	tod_btconv<2>(bta).perform(ta);

	//	Compare against the reference

	compare_ref<2>::compare(oss.str().c_str(), ta, ta_ref, 1e-15);

	} catch(exception &e) {
		fail_test(oss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


/**	\test Elementwise operation of two order-2 tensors with symmetry
		and zero blocks (scaled)
 **/
void btod_mult1_test::test_3(
		bool recip, bool doadd) throw(libtest::test_exception) {

	std::ostringstream oss;
	oss << "btod_mult1_test::test_3("
			<< (recip ? "true" : "false") << ","
			<< (doadd ? "true" : "false") << ")";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> msk; msk[0] = true; msk[1] = true;
	bis.split(msk, 3);
	bis.split(msk, 7);
	dimensions<2> bidims(bis.get_block_index_dims());
	permutation<2> p10; p10.permute(0, 1);
	se_perm<2, double> sp10(p10, true), ap10(p10, false);

	block_tensor<2, double, allocator_t> bta(bis), btb(bis);
	tensor<2, double, allocator_t> ta(dims), tb(dims), ta_ref(dims);

	//  Add symmetries

	block_tensor_ctrl<2, double> cbta(bta), cbtb(btb);
	cbta.req_symmetry().insert(ap10);
	cbtb.req_symmetry().insert(sp10);

	//	Fill in random data

	btod_random<2>().perform(bta);
	btod_random<2>().perform(btb);
	btb.set_immutable();

	//  Add zero blocks

	index<2> idx;
	idx[0] = 1; idx[1] = 2;
	cbta.req_zero_block(idx);

	//	Prepare the reference

	tod_btconv<2>(bta).perform(ta_ref);
	tod_btconv<2>(btb).perform(tb);

	//	Invoke the operation

	if (doadd) {
		tod_mult1<2>(tb, recip, 0.7).perform(ta_ref, 0.3);
		btod_mult1<2>(btb, recip, 0.7).perform(bta, 0.3);
	}
	else {
		tod_mult1<2>(tb, recip, 0.7).perform(ta_ref);
		btod_mult1<2>(btb,recip, 0.7).perform(bta);
	}
	tod_btconv<2>(bta).perform(ta);

	//	Compare against the reference

	compare_ref<2>::compare(oss.str().c_str(), ta, ta_ref, 1e-15);

	} catch(exception &e) {
		fail_test(oss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


/**	\test Elementwise operation of two order-4 tensors
		with symmetry and zero blocks, second tensor permuted
 **/
void btod_mult1_test::test_4(
		bool recip, bool doadd) throw(libtest::test_exception) {

	std::ostringstream oss;
	oss << "btod_mult1_test::test_4("
			<< (recip ? "true" : "false") << ","
			<< (doadd ? "true" : "false") << ")";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	mask<4> msk; msk[0] = true; msk[1] = true; msk[2] = true; msk[3] = true;
	bis.split(msk, 3);	bis.split(msk, 7);
	dimensions<4> bidims(bis.get_block_index_dims());
	permutation<4> p10, p32, p21;
	p10.permute(0, 1);
	p32.permute(2, 3);
	p21.permute(1, 2);
	se_perm<4, double> sp10(p10, true), ap32(p32, false);

	block_tensor<4, double, allocator_t> bta(bis), btb(bis);
	tensor<4, double, allocator_t> ta(dims), tb(dims), ta_ref(dims);

	//  Add symmetries

	block_tensor_ctrl<4, double> cbta(bta), cbtb(btb);
	cbta.req_symmetry().insert(ap32);
	cbtb.req_symmetry().insert(sp10);

	//	Fill in random data

	btod_random<4>().perform(bta);
	btod_random<4>().perform(btb);
	btb.set_immutable();

	//  Add zero blocks

	index<4> idx;
	idx[0] = 1; idx[1] = 2; idx[2] = 0; idx[3] = 0;
	cbta.req_zero_block(idx);
	idx[0] = 0; idx[1] = 0; idx[2] = 1; idx[3] = 2;
	cbta.req_zero_block(idx);

	//	Prepare the reference

	tod_btconv<4>(bta).perform(ta_ref);
	tod_btconv<4>(btb).perform(tb);

	//	Invoke the operation
	if (doadd) {
		tod_mult1<4>(tb, p21, recip).perform(ta_ref, 0.5);
		btod_mult1<4>(btb, p21, recip).perform(bta, 0.5);
	}
	else {
		tod_mult1<4>(tb, p21, recip).perform(ta_ref);
		btod_mult1<4>(btb, p21, recip).perform(bta);
	}
	tod_btconv<4>(bta).perform(ta);

	//	Compare against the reference

	compare_ref<4>::compare(oss.str().c_str(), ta, ta_ref, 1e-15);

	} catch(exception &e) {
		fail_test(oss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


/**	\test Elementwise operation of two order-2 tensors with partition
		symmetry no zero blocks.
 **/
void btod_mult1_test::test_5(bool recip, bool doadd)
	throw(libtest::test_exception) {

	std::ostringstream oss;
	oss << "btod_mult1_test::test_5("
			<< (recip ? "true" : "false") << ", "
			<< (doadd ? "true" : "false") << ")";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	mask<2> m11;
	m11[0] = true; m11[1] = true;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	bis.split(m11, 2);
	bis.split(m11, 5);
	bis.split(m11, 7);
	dimensions<2> bidims(bis.get_block_index_dims());

	block_tensor<2, double, allocator_t> bta(bis), btb(bis);
	symmetry<2, double> syma(bis), syma_ref(bis);
	tensor<2, double, allocator_t> ta(dims), tb(dims), ta_ref(dims);

	//	Install symmetry

	index<2> i00, i01, i10, i11;
	i10[0] = 1; i01[1] = 1; i11[0] = 1; i11[1] = 1;
	se_part<2, double> separta(bis, m11, 2), separtb(bis, m11, 2);
	separta.add_map(i00, i11, false);
	separta.add_map(i01, i10, true);
	separtb.add_map(i00, i01, true);
	separtb.add_map(i01, i10, true);
	separtb.add_map(i10, i11, true);
	syma_ref.insert(separta);
	{
		block_tensor_ctrl<2, double> ctrla(bta), ctrlb(btb);
		ctrla.req_symmetry().insert(separta);
		ctrlb.req_symmetry().insert(separtb);
	}

	//	Fill in random data

	btod_random<2>().perform(bta);
	btod_random<2>().perform(btb);
	btb.set_immutable();

	//	Prepare the reference

	tod_btconv<2>(bta).perform(ta_ref);
	tod_btconv<2>(btb).perform(tb);

	//	Invoke the operation

	if(doadd) {
		tod_mult1<2>(tb, recip).perform(ta_ref, -1.2);
		btod_mult1<2>(btb, recip).perform(bta, -1.2);
	} else {
		tod_mult1<2>(tb, recip).perform(ta_ref);
		btod_mult1<2>(btb, recip).perform(bta);
	}
	tod_btconv<2>(bta).perform(ta);

	//	Compare against the reference

	{
		block_tensor_ctrl<2, double> ctrla(bta);
		so_copy<2, double>(ctrla.req_const_symmetry()).perform(syma);
	}
	compare_ref<2>::compare(oss.str().c_str(), syma, syma_ref);
	compare_ref<2>::compare(oss.str().c_str(), ta, ta_ref, 1e-15);

	} catch(exception &e) {
		fail_test(oss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
