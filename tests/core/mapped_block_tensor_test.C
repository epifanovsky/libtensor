#include <libtensor/core/allocator.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/core/block_tensor_ctrl.h>
#include <libtensor/core/mapped_block_tensor.h>
#include <libtensor/tod/tod_random.h>
#include "mapped_block_tensor_test.h"

namespace libtensor {


void mapped_block_tensor_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
}


namespace mapped_block_tensor_test_ns {


template<size_t N>
class block_index_map_self : public block_index_map_i<N> {
private:
	block_index_space<N> m_bis;
	dimensions<N> m_bidims;

public:
	block_index_map_self(const block_index_space<N> &bis) :
		m_bis(bis), m_bidims(bis.get_block_index_dims()) { }
	virtual ~block_index_map_self() { }

	virtual const block_index_space<N> &get_bis_from() const {
		return m_bis;
	}
	virtual const block_index_space<N> &get_bis_to() const {
		return m_bis;
	}
	virtual bool map_exists(const index<N> &from) const {
		return true;
	}
	virtual void get_map(const index<N> &from, index<N> &to) const {
		abs_index<N> ai_verify(from, m_bidims);
		to = from;
	}
};


template<size_t N>
class block_index_map_perm : public block_index_map_i<N> {
private:
	block_index_space<N> m_bis;
	dimensions<N> m_bidims;
	permutation<N> m_perm;

public:
	block_index_map_perm(const block_index_space<N> &bis,
		const permutation<N> &perm) :
		m_bis(bis), m_bidims(bis.get_block_index_dims()),
		m_perm(perm) { }
	virtual ~block_index_map_perm() { }

	virtual const block_index_space<N> &get_bis_from() const {
		return m_bis;
	}
	virtual const block_index_space<N> &get_bis_to() const {
		return m_bis;
	}
	virtual bool map_exists(const index<N> &from) const {
		return true;
	}
	virtual void get_map(const index<N> &from, index<N> &to) const {
		abs_index<N> ai_verify(from, m_bidims);
		to = from;
		to.permute(m_perm);
	}
};


} // namespace mapped_block_tensor_test_ns
using namespace mapped_block_tensor_test_ns;


/**	\test Tests the identity mapping from one block %tensor onto another
 **/
void mapped_block_tensor_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "mapped_block_tensor_test::test_1()";

	typedef std_allocator<double> allocator_t;

	cpu_pool cpus(1);
	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	block_index_space<2> bis(index_range<2>(i1, i2));
	mask<2> m; m[0] = true; m[1] = true;
	bis.split(m, 2);
	bis.split(m, 5);
	dimensions<2> bidims = bis.get_block_index_dims();

	block_tensor<2, double, allocator_t> bt(bis);
	symmetry<2, double> sym(bis);
	block_tensor_ctrl<2, double> ctrl(bt);

	abs_index<2> aidx1(bidims);
	do {
		dense_tensor_i<2, double> &blk = ctrl.req_block(aidx1.get_index());
		tod_random<2>().perform(cpus, blk);
		ctrl.ret_block(aidx1.get_index());
	} while(aidx1.inc());

	bt.set_immutable();

	mapped_block_tensor<2, double> mbt(bt,
		block_index_map_self<2>(bis), sym);
	block_tensor_ctrl<2, double> mctrl(mbt);

	abs_index<2> aidx2(bidims);
	do {
		dense_tensor_i<2, double> &blk1 = ctrl.req_block(aidx2.get_index());
		dense_tensor_i<2, double> &blk2 = mctrl.req_block(aidx2.get_index());
		if(&blk1 != &blk2) {
			fail_test(testname, __FILE__, __LINE__,
				"&blk1 != &blk2");
		}
		ctrl.ret_block(aidx2.get_index());
		mctrl.ret_block(aidx2.get_index());
	} while(aidx2.inc());

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the permutation mapping from one block %tensor onto another
 **/
void mapped_block_tensor_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "mapped_block_tensor_test::test_2()";

	typedef std_allocator<double> allocator_t;

	cpu_pool cpus(1);

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	block_index_space<2> bis(index_range<2>(i1, i2));
	mask<2> m; m[0] = true; m[1] = true;
	bis.split(m, 2);
	bis.split(m, 5);
	dimensions<2> bidims = bis.get_block_index_dims();

	block_tensor<2, double, allocator_t> bt(bis);
	symmetry<2, double> sym(bis);
	block_tensor_ctrl<2, double> ctrl(bt);

	abs_index<2> aidx1(bidims);
	do {
		dense_tensor_i<2, double> &blk = ctrl.req_block(aidx1.get_index());
		tod_random<2>().perform(cpus, blk);
		ctrl.ret_block(aidx1.get_index());
	} while(aidx1.inc());

	bt.set_immutable();

	permutation<2> perm; perm.permute(0, 1);
	mapped_block_tensor<2, double> mbt(bt,
		block_index_map_perm<2>(bis, perm), sym);
	block_tensor_ctrl<2, double> mctrl(mbt);

	abs_index<2> aidx2(bidims);
	do {
		index<2> idx2(aidx2.get_index()); idx2.permute(perm);
		dense_tensor_i<2, double> &blk1 = ctrl.req_block(idx2);
		dense_tensor_i<2, double> &blk2 = mctrl.req_block(aidx2.get_index());
		if(&blk1 != &blk2) {
			fail_test(testname, __FILE__, __LINE__,
				"&blk1 != &blk2");
		}
		ctrl.ret_block(idx2);
		mctrl.ret_block(aidx2.get_index());
	} while(aidx2.inc());

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
