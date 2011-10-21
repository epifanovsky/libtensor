#include <libtensor/btod/btod_random.h>
#include <libtensor/iface/iface.h>
#include "mapped_btensor_test.h"
#include "../compare_ref.h"

namespace libtensor {


void mapped_btensor_test::perform() throw(libtest::test_exception) {

	allocator<double>::vmm().init(16, 16, 16777216, 16777216);

	try {

		test_1();
		test_2();

	} catch(...) {
		allocator<double>::vmm().shutdown();
		throw;
	}

	allocator<double>::vmm().shutdown();
}


namespace mapped_btensor_test_ns {


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
	block_index_space<N> m_bis_from;
	block_index_space<N> m_bis_to;
	dimensions<N> m_bidims;
	permutation<N> m_perm;

public:
	block_index_map_perm(const block_index_space<N> &bis,
		const permutation<N> &perm) :
		m_bis_from(bis), m_bis_to(bis),
		m_bidims(bis.get_block_index_dims()), m_perm(perm) {
		m_bis_to.permute(m_perm);
	}
	virtual ~block_index_map_perm() { }

	virtual const block_index_space<N> &get_bis_from() const {
		return m_bis_from;
	}
	virtual const block_index_space<N> &get_bis_to() const {
		return m_bis_to;
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


} // namespace mapped_btensor_test_ns
using namespace mapped_btensor_test_ns;


void mapped_btensor_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "mapped_btensor_test::test_1()";

	try {

	bispace<1> si(5), sa(6);
	si.split(2); sa.split(3);
	bispace<2> sia(si|sa);

	btensor<2> bt1(sia);
	btod_random<2>().perform(bt1);

	block_index_map_self<2> bimap(sia.get_bis());
	symmetry<2, double> sym(sia.get_bis());
	mapped_btensor<2> mbt(bt1, bimap, sym);

	btensor<2> bt(sia), bt_ref(sia);
	btod_copy<2>(bt1, 1.5).perform(bt_ref);

	letter i, a;
	bt(i|a) = 1.5 * mbt(i|a);

	compare_ref<2>::compare(testname, bt, bt_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void mapped_btensor_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "mapped_btensor_test::test_2()";

	try {

	bispace<1> si(6);
	si.split(2).split(4);
	bispace<2> sij(si|si), sji(sij);

	permutation<2> perm;
	perm.permute(0, 1);

	btensor<2> bt1(sij), bt2(sji);
	btod_random<2>().perform(bt1);

	letter i, j;

	block_index_map_perm<2> bimap1(sji.get_bis(), perm);
	block_index_map_perm<2> bimap2(sij.get_bis(), perm);
	symmetry<2, double> sym1(sji.get_bis());
	symmetry<2, double> sym2(sij.get_bis());
	mapped_btensor<2> mbt1(bt1, bimap1, sym1);
	bt2(j|i) = mbt1(j|i);
	mapped_btensor<2> mbt2(bt2, bimap2, sym2);

	btensor<2> bt(sij), bt_ref(sij);
	btod_copy<2>(bt1).perform(bt_ref);

	bt(i|j) = mbt2(i|j);

	compare_ref<2>::compare(testname, bt, bt_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
