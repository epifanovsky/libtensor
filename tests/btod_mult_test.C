#include <libvmm/std_allocator.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/btod/btod_mult.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/tod/tod_btconv.h>
#include <libtensor/tod/tod_mult.h>
#include <sstream>
#include "btod_mult_test.h"
#include "compare_ref.h"

namespace libtensor {


void btod_mult_test::perform() throw(libtest::test_exception) {

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
	test_6(false, false); test_6(false, true);
	test_6(true, false);  test_6(true, true);
}


/**	\test Elementwise multiplication/division of two order-2 tensors with no symmetry
		and no zero blocks.
 **/
void btod_mult_test::test_1(
		bool recip, bool doadd) throw(libtest::test_exception) {

	std::ostringstream oss;
	oss << "btod_mult_test::test_1("
			<< (recip ? "true" : "false") << ","
			<< (doadd ? "true" : "false") << ")";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	dimensions<2> bidims(bis.get_block_index_dims());

	block_tensor<2, double, allocator_t> bta(bis), btb(bis), btc(bis);
	tensor<2, double, allocator_t> ta(dims), tb(dims), tc(dims),
		tc_ref(dims);

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
	if (doadd) {
		tod_mult<2>(ta, tb, recip).perform(tc_ref, 0.5);
		btod_mult<2>(bta, btb, recip).perform(btc, 0.5);
	}
	else {
		tod_mult<2>(ta, tb, recip).perform(tc_ref);
		btod_mult<2>(bta, btb, recip).perform(btc);
	}

	tod_btconv<2>(btc).perform(tc);

	//	Compare against the reference

	compare_ref<2>::compare(oss.str().c_str(), tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(oss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


/**	\test Elementwise multiplication/division of two order-2 tensors
		with no symmetry and no zero blocks, second tensor permuted
 **/
void btod_mult_test::test_2(
		bool recip, bool doadd) throw(libtest::test_exception) {

	static const char *testname = "btod_mult_test::test_2";
	std::ostringstream oss;
	oss << testname << "("
			<< (recip ? "true" : "false") << ","
			<< (doadd ? "true" : "false") << ")";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	dimensions<2> bidims(bis.get_block_index_dims());

	block_tensor<2, double, allocator_t> bta(bis), btb(bis), btc(bis);
	tensor<2, double, allocator_t> ta(dims), tb(dims), tc(dims),
		tc_ref(dims);

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

	permutation<2> pa, pb;
	pb.permute(0, 1);

	//	Invoke the operation
	if (doadd) {
		tod_mult<2>(ta, pa, tb, pb, recip).perform(tc_ref, 0.5);
		btod_mult<2>(bta, pa, btb, pb, recip).perform(btc, 0.5);
	}
	else {
		tod_mult<2>(ta, pa, tb, pb, recip).perform(tc_ref);
		btod_mult<2>(bta, pa, btb, pb, recip).perform(btc);
	}
	tod_btconv<2>(btc).perform(tc);

	//	Compare against the reference

	compare_ref<2>::compare(oss.str().c_str(), tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(oss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


/**	\test Elementwise multiplication/division of two order-2 tensors with
		 symmetry and zero blocks.
 **/
void btod_mult_test::test_3(
		bool recip, bool doadd) throw(libtest::test_exception) {

	static const char *testname = "btod_mult_test::test_3";
	std::ostringstream oss;
	oss << testname << "("
			<< (recip ? "true" : "false") << ","
			<< (doadd ? "true" : "false") << ")";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> msk;
	msk[0] =  true; msk[1] = true;
	bis.split(msk, 3);
	bis.split(msk, 7);
	dimensions<2> bidims(bis.get_block_index_dims());

	permutation<2> perm;
	perm.permute(0, 1);
	se_perm<2, double> sp(perm, true);
	block_tensor<2, double, allocator_t> bta(bis), btb(bis), btc(bis);

	tensor<2, double, allocator_t> ta(dims), tb(dims), tc(dims),
		tc_ref(dims);


	{ // add symmetry and set zero blocks
	block_tensor_ctrl<2, double> cbta(bta), cbtb(btb), cbtc(btc);
	cbta.req_symmetry().insert(sp);
	cbtb.req_symmetry().insert(sp);
	cbtc.req_symmetry().insert(sp);
	}

	//	Fill in random data

	btod_random<2>().perform(bta);
	btod_random<2>().perform(btb);
	btod_random<2>().perform(btc);

	{ // set zero blocks
	block_tensor_ctrl<2, double> cbta(bta);
	index<2> idxa;
	idxa[0] = 0; idxa[1] = 2;
	orbit<2, double> oa(cbta.req_const_symmetry(), idxa);
	abs_index<2> cidxa(oa.get_abs_canonical_index(), bidims);
	cbta.req_zero_block(cidxa.get_index());
	}

	bta.set_immutable();
	btb.set_immutable();

	//	Prepare the reference

	tod_btconv<2>(bta).perform(ta);
	tod_btconv<2>(btb).perform(tb);
	tod_btconv<2>(btc).perform(tc_ref);

	//	Invoke the operation

	if (doadd) {
		tod_mult<2>(ta, tb, recip).perform(tc_ref, -0.5);
		btod_mult<2>(bta, btb, recip).perform(btc, -0.5);
	}
	else {
		tod_mult<2>(ta, tb, recip).perform(tc_ref);
		btod_mult<2>(bta, btb, recip).perform(btc);
	}
	tod_btconv<2>(btc).perform(tc);

	//	Compare against the reference

	compare_ref<2>::compare(oss.str().c_str(), tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(oss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


/**	\test Elementwise multiplaction/division of two order-4 tensors
		with symmetry and zero blocks.
 **/
void btod_mult_test::test_4(
		bool recip, bool doadd) throw(libtest::test_exception) {

	static const char *testname = "btod_mult_test::test_4";
	std::ostringstream oss;
	oss << testname << "("
			<< (recip ? "true" : "false") << ","
			<< (doadd ? "true" : "false") << ")";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 7; i2[3] = 7;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	mask<4> msk1, msk2;
	msk1[0] = true; msk1[1] = true;
	msk2[2] = true; msk2[3] = true;
	bis.split(msk1, 3); bis.split(msk1, 6);
	bis.split(msk2, 4);

	dimensions<4> bidims(bis.get_block_index_dims());

	permutation<4> p10, p32;
	p10.permute(0, 1);
	p32.permute(2, 3);
	se_perm<4, double> spa(p10, false), spb(p32, true);

	block_tensor<4, double, allocator_t> bta(bis), btb(bis), btc(bis);
	tensor<4, double, allocator_t> ta(dims), tb(dims), tc(dims),
		tc_ref(dims);

	{
	block_tensor_ctrl<4, double> cbta(bta), cbtb(btb);
	cbta.req_symmetry().insert(spa);
	cbtb.req_symmetry().insert(spb);
	}

	//	Fill in random data

	btod_random<4>().perform(bta);
	btod_random<4>().perform(btb);
	btod_random<4>().perform(btc);

	{
	block_tensor_ctrl<4, double> cbta(bta);
	index<4> idxa;
	idxa[0] = 0; idxa[1] = 1; idxa[2] = 1; idxa[3] = 0;
	orbit<4, double> oa(cbta.req_const_symmetry(), idxa);
	abs_index<4> cidxa(oa.get_abs_canonical_index(), bidims);
	cbta.req_zero_block(cidxa.get_index());
	}

	bta.set_immutable();
	btb.set_immutable();

	//	Prepare the reference

	tod_btconv<4>(bta).perform(ta);
	tod_btconv<4>(btb).perform(tb);
	tod_btconv<4>(btc).perform(tc_ref);


	//	Invoke the operation

	if (doadd) {
		tod_mult<4>(ta, tb, recip).perform(tc_ref, 0.5);
		btod_mult<4>(bta, btb, recip).perform(btc, 0.5);
	}
	else {
		tod_mult<4>(ta, tb, recip).perform(tc_ref);
		btod_mult<4>(bta, btb, recip).perform(btc);
	}

	tod_btconv<4>(btc).perform(tc);

	//	Compare against the reference

	compare_ref<4>::compare(oss.str().c_str(), tc, tc_ref, 1e-15);

	} catch(exception &e) {
		fail_test(oss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}

/**	\test Elementwise multiplaction/division of two order-2 tensors
		with permutational symmetry and anti-symmetry.
		Test for the right result symmetry!
 **/
void btod_mult_test::test_5(bool symm1, bool symm2) throw(libtest::test_exception) {

	std::ostringstream testname;
	testname << "btod_mult_test::test_5("
			<< (symm1 ? "true" : "false") << ", "
			<< (symm2 ? "true" : "false") << ")";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> msk1;
	msk1[0] = true; msk1[1] = true;
	bis.split(msk1, 5);

	dimensions<2> bidims(bis.get_block_index_dims());

	permutation<2> p10;
	p10.permute(0, 1);
	se_perm<2, double> spa(p10, symm1), spb(p10, symm2);

	block_tensor<2, double, allocator_t> bta(bis), btb(bis);

	{
	block_tensor_ctrl<2, double> cbta(bta), cbtb(btb);
	cbta.req_symmetry().insert(spa);
	cbtb.req_symmetry().insert(spb);
	}

	//	Fill in random data

	btod_random<2>().perform(bta);
	btod_random<2>().perform(btb);

	bta.set_immutable();
	btb.set_immutable();

	//	Invoke the operation
	btod_mult<2> op(bta, btb);
	const symmetry<2, double> &sym = op.get_symmetry();

	bool found = false;
	for (symmetry<2, double>::iterator is = sym.begin();
			is != sym.end(); is++) {

		const symmetry_element_set<2, double> &set = sym.get_subset(is);
		if (set.get_id().compare(spa.get_type()) != 0)
			fail_test(testname.str().c_str(), __FILE__, __LINE__,
					"Unknown symmetry element type.");

		if (set.is_empty())
			fail_test(testname.str().c_str(), __FILE__, __LINE__,
					"Permutational symmetry missing.");

		index<2> idx;
		transf<2, double> tr;
		for (symmetry_element_set<2, double>::const_iterator iss =
				set.begin(); iss != set.end(); iss++) {

			const symmetry_element_i<2, double> &elem = set.get_elem(iss);
			elem.apply(idx, tr);

		}

		if (! tr.get_perm().equals(p10))
			fail_test(testname.str().c_str(), __FILE__, __LINE__,
					"Wrong permutational symmetry.");

		if (symm1 == symm2) {
			if (tr.get_coeff() != 1.0) {
				fail_test(testname.str().c_str(), __FILE__, __LINE__,
						"Wrong permutational symmetry.");
			}
		}
		else {
			if (tr.get_coeff() != -1.0) {
				fail_test(testname.str().c_str(), __FILE__, __LINE__,
						"Wrong permutational symmetry.");
			}
		}

		found = true;
	}

	if (! found)
		fail_test(testname.str().c_str(), __FILE__, __LINE__, "Symmetry missing.");

	} catch(exception &e) {
		fail_test(testname.str().c_str(), __FILE__, __LINE__, e.what());
	}
}

/**	\test Elementwise multiplaction/division of two order-4 tensors
		with permutational symmetry and anti-symmetry.
		Test for the right result symmetry!
 **/
void btod_mult_test::test_6(bool symm1, bool symm2) throw(libtest::test_exception) {

	std::ostringstream testname;
	testname << "btod_mult_test::test_6("
			<< (symm1 ? "true" : "false") << ", "
			<< (symm2 ? "true" : "false") << ")";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = 9; i2[1] = 9; i2[2] = 7; i2[3] = 7;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	mask<4> msk1, msk2;
	msk1[0] = true; msk1[1] = true;
	msk2[2] = true; msk2[3] = true;
	bis.split(msk1, 5);
	bis.split(msk2, 4);
	dimensions<4> bidims(bis.get_block_index_dims());

	permutation<4> p10, p32;
	p10.permute(0, 1); p32.permute(2, 3);

	se_perm<4, double> spa1(p10, symm1), spa2(p32, symm1);
	se_perm<4, double> spb(p10, symm2);

	block_tensor<4, double, allocator_t> bta(bis), btb(bis);

	{
	block_tensor_ctrl<4, double> cbta(bta), cbtb(btb);
	cbta.req_symmetry().insert(spa1);
	cbta.req_symmetry().insert(spa2);
	cbtb.req_symmetry().insert(spb);
	}

	//	Fill in random data

	btod_random<4>().perform(bta);
	btod_random<4>().perform(btb);

	bta.set_immutable();
	btb.set_immutable();

	//	Invoke the operation
	btod_mult<4> op(bta, btb);
	const symmetry<4, double> &sym = op.get_symmetry();

	bool found = false;
	for (symmetry<4, double>::iterator is = sym.begin();
			is != sym.end(); is++) {

		const symmetry_element_set<4, double> &set = sym.get_subset(is);
		if (set.get_id().compare(spb.get_type()) != 0)
			fail_test(testname.str().c_str(), __FILE__, __LINE__,
					"Unknown symmetry element type.");

		if (set.is_empty())
			fail_test(testname.str().c_str(), __FILE__, __LINE__,
					"Permutational symmetry missing.");

		index<4> idx;
		transf<4, double> tr;
		for (symmetry_element_set<4, double>::const_iterator iss =
				set.begin(); iss != set.end(); iss++) {

			const symmetry_element_i<4, double> &elem = set.get_elem(iss);
			elem.apply(idx, tr);

		}

		if (! tr.get_perm().equals(p10))
			fail_test(testname.str().c_str(), __FILE__, __LINE__,
					"Wrong permutational symmetry.");

		if (symm1 == symm2) {
			if (tr.get_coeff() != 1.0) {
				fail_test(testname.str().c_str(), __FILE__, __LINE__,
						"Wrong symm flag symmetry.");
			}
		}
		else {
			if (tr.get_coeff() != -1.0) {
				fail_test(testname.str().c_str(), __FILE__, __LINE__,
						"Wrong symm flag symmetry.");
			}
		}

		found = true;
	}

	if (! found)
		fail_test(testname.str().c_str(), __FILE__, __LINE__, "Symmetry missing.");

	} catch(exception &e) {
		fail_test(testname.str().c_str(), __FILE__, __LINE__, e.what());
	}
}



} // namespace libtensor
