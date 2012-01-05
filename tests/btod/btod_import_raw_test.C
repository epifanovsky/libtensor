#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/btod/btod_import_raw.h>
#include <libtensor/tod/tod_add.h>
#include <libtensor/tod/tod_btconv.h>
#include <libtensor/tod/tod_random.h>
#include <libtensor/symmetry/se_perm.h>
#include "btod_import_raw_test.h"
#include "../compare_ref.h"

namespace libtensor {


void btod_import_raw_test::perform() throw(libtest::test_exception) {

	index<2> i2a, i2b;
	i2b[0] = 9; i2b[1] = 19;
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	block_index_space<2> bis2a(dims2), bis2b(dims2), bis2c(dims2);
	mask<2> m2a, m2b;
	m2a[0] = true; m2b[1] = true;
	bis2b.split(m2a, 4);
	bis2b.split(m2b, 11);
	bis2c.split(m2a, 3);
	bis2c.split(m2a, 5);

	index<4> i4a, i4b;
	i4b[0] = 9; i4b[1] = 19; i4b[2] = 9; i4b[3] = 19;
	dimensions<4> dims4(index_range<4>(i4a, i4b));
	block_index_space<4> bis4a(dims4), bis4b(dims4), bis4c(dims4);
	mask<4> m4a, m4b;
	m4a[0] = true; m4a[2] = true; m4b[1] = true; m4b[3] = true;
	bis4b.split(m4a, 4);
	bis4b.split(m4b, 11);
	bis4c.split(m4a, 3);
	bis4c.split(m4a, 5);

	test_1(bis2a);
	test_1(bis2b);
	test_1(bis2c);
	test_1(bis4a);
	test_1(bis4b);
	test_1(bis4c);
}


template<size_t N>
void btod_import_raw_test::test_1(const block_index_space<N> &bis)
	throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "btod_import_raw_test::test_1(" << bis << ")";

	typedef std_allocator<double> allocator_t;
	typedef tensor<N, double, allocator_t> tensor_t;
	typedef dense_tensor_ctrl<N, double> tensor_ctrl_t;
	typedef block_tensor<N, double, allocator_t> block_tensor_t;

	cpu_pool cpus(1);

	try {

	//	Create tensors

	tensor_t ta(bis.get_dims()), tb(bis.get_dims()), tb_ref(bis.get_dims());
	block_tensor_t btb(bis);

	//	Fill in random data

	tod_random<N>().perform(cpus, ta);

	//	Create reference data

	tod_copy<N>(ta).perform(cpus, true, 1.0, tb_ref);

	//	Invoke the operation

	{
		tensor_ctrl_t tca(ta);
		const double *pa = tca.req_const_dataptr();
		btod_import_raw<N>(pa, bis.get_dims()).perform(btb);
		tca.ret_const_dataptr(pa); pa = 0;
	}

	//	Compare against the reference

	tod_btconv<N>(btb).perform(tb);
	compare_ref<N>::compare(tnss.str().c_str(), tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}

template<size_t N>
void btod_import_raw_test::test_2(const block_index_space<N> &bis)
	throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "btod_import_raw_test::test_2(" << bis << ")";

	typedef std_allocator<double> allocator_t;
	typedef tensor<N, double, allocator_t> tensor_t;
	typedef dense_tensor_ctrl<N, double> tensor_ctrl_t;
	typedef block_tensor<N, double, allocator_t> block_tensor_t;

	try {

	bool found = false;
	size_t i = 0, j = 0;
	while (i != N) {
		size_t type = bis.get_type(i);
		j = i + 1;
		while (j != N) {
			if (type == bis.get_type(j)) {
				found = true;
				break;
			}
			j++;
		}

		if (found) break;
		i++;
	}
	if (! found) return;

	permutation<N> p_ij;
	p_ij.permute(i,j);
	se_perm<N, double> se_ij(p_ij, true);

	//	Create tensors

	tensor_t tmp(bis.get_dims()), ta(bis.get_dims()),
			tb(bis.get_dims()), tb_ref(bis.get_dims());
	block_tensor_t btb(bis);
	btb.req_symmetry().insert(se_ij);


	//	Fill in random data
	{
		tod_random<N>().perform(tmp);
		tod_add<N> tadd(tmp, 1.0);
		tadd.add_op(tmp, p_ij, 1.0).perform(ta);
	}
	//	Create reference data

	tod_copy<N>(ta).perform(tb_ref);

	//	Invoke the operation

	{
		tensor_ctrl_t tca(ta);
		const double *pa = tca.req_const_dataptr();
		btod_import_raw<N>(pa, bis.get_dims()).perform(btb);
		tca.ret_const_dataptr(pa); pa = 0;
	}

	//	Compare against the reference

	tod_btconv<N>(btb).perform(tb);
	compare_ref<N>::compare(tnss.str().c_str(), tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
