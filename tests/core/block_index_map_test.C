#include <libtensor/core/block_index_map.h>
#include "block_index_map_test.h"

namespace libtensor {


void block_index_map_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
}


namespace block_index_map_test_ns {


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
class block_index_map_shift : public block_index_map_i<N> {
private:
	block_index_space<N> m_bis;
	dimensions<N> m_bidims;

public:
	block_index_map_shift(const block_index_space<N> &bis) :
		m_bis(bis), m_bidims(bis.get_block_index_dims()) { }
	virtual ~block_index_map_shift() { }

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
		for(size_t i = 0; i < N; i++) {
			to[i] = (to[i] + 2) % m_bidims[i];
		}
	}
};


template<size_t N>
class block_index_map_window1 : public block_index_map_i<N> {
private:
	block_index_space<N> m_bis1, m_bis2;
	index_range<N> m_ir;

public:
	block_index_map_window1(const block_index_space<N> &bis1,
		const block_index_space<N> &bis2,
		const index_range<N> &ir) :
		m_bis1(bis1), m_bis2(bis2), m_ir(ir) { }
	virtual ~block_index_map_window1() { }

	virtual const block_index_space<N> &get_bis_from() const {
		return m_bis1;
	}
	virtual const block_index_space<N> &get_bis_to() const {
		return m_bis2;
	}
	virtual bool map_exists(const index<N> &from) const {
		const index<N> &begin = m_ir.get_begin();
		const index<N> &end = m_ir.get_end();
		for(size_t i = 0; i < N; i++) {
			if(from[i] < begin[i] || from[i] > end[i]) return false;
		}
		return true;
	}
	virtual void get_map(const index<N> &from, index<N> &to) const {
		index<N> shift(m_ir.get_begin());
		for(size_t i = 0; i < N; i++) to[i] = from[i] - shift[i];
	}
};


} // namespace block_index_map_test_ns
using namespace block_index_map_test_ns;


/**	\test Tests a mapping from a block index space onto itself
 **/
void block_index_map_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "block_index_map_test::test_1()";

	try {

	index<2> i1, i2; i2[0] = 9; i2[1] = 9;
	mask<2> m; m[0] = true; m[1] = true;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	bis.split(m, 3);
	bis.split(m, 5);
	dimensions<2> bidims = bis.get_block_index_dims();

	block_index_map_self<2> bimap_ref(bis);
	block_index_map<2> bimap(bimap_ref);

	abs_index<2> ai(bidims);
	do {
		index<2> idx1(ai.get_index()), idx2;
		bimap.get_map(idx1, idx2);
		if(!idx1.equals(idx2)) {
			fail_test(testname, __FILE__, __LINE__, "idx1 != idx2");
		}
	} while(ai.inc());

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests a mapping from a block index space onto itself,
		block indexes are shifted
 **/
void block_index_map_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "block_index_map_test::test_2()";

	try {

	index<2> i1, i2; i2[0] = 8; i2[1] = 8;
	mask<2> m; m[0] = true; m[1] = true;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	bis.split(m, 3);
	bis.split(m, 6);
	dimensions<2> bidims = bis.get_block_index_dims();

	block_index_map_shift<2> bimap_ref(bis);
	block_index_map<2> bimap(bimap_ref);

	abs_index<2> ai(bidims);
	do {
		index<2> idx1(ai.get_index()), idx2;
		bimap.get_map(idx1, idx2);
		idx1[0] = (idx1[0] + 2) % 3;
		idx1[1] = (idx1[1] + 2) % 3;
		if(!idx1.equals(idx2)) {
			fail_test(testname, __FILE__, __LINE__, "idx1 != idx2");
		}
	} while(ai.inc());

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests a mapping from a block index space onto a smaller
		space (window)
 **/
void block_index_map_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "block_index_map_test::test_3()";

	try {

	mask<2> m; m[0] = true; m[1] = true;
	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	block_index_space<2> bis1(dimensions<2>(index_range<2>(i1, i2)));
	i2[0] = 5; i2[1] = 5;
	block_index_space<2> bis2(dimensions<2>(index_range<2>(i1, i2)));
	bis1.split(m, 2);
	bis1.split(m, 4);
	bis1.split(m, 6);
	bis1.split(m, 8);
	bis2.split(m, 2);
	bis2.split(m, 4);
	dimensions<2> bidims1 = bis1.get_block_index_dims();
	dimensions<2> bidims2 = bis2.get_block_index_dims();
	i1[0] = 1; i1[1] = 1;
	i2[0] = 3; i2[1] = 3;
	index_range<2> ir(i1, i2);

	block_index_map_window1<2> bimap_ref(bis1, bis2, ir);
	block_index_map<2> bimap(bimap_ref);

	abs_index<2> ai(bidims2);
	do {
		index<2> idx1(ai.get_index()), idx2;
		idx1[0]++; idx1[1]++;
		bimap.get_map(idx1, idx2);
		if(!idx2.equals(ai.get_index())) {
			fail_test(testname, __FILE__, __LINE__, "bad mapping");
		}
	} while(ai.inc());

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
