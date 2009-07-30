#include <libtensor.h>
#include "orbit_iterator_test.h"

namespace libtensor {

namespace orbit_iterator_test_ns {

class handler1 : public orbit_iterator_handler_i<2, int> {
private:
	dimensions<2> m_dims;

public:
	handler1(const dimensions<2> &dims) : m_dims(dims) { };

	virtual bool on_begin(index<2> &idx) const {
		return true;
	}

	virtual bool on_next(index<2> &idx) const {
		return m_dims.inc_index(idx);
	}
};

class handler2 : public orbit_iterator_handler_i<2, int> {
private:
	dimensions<2> m_dims;

public:
	handler2(const dimensions<2> &dims) : m_dims(dims) { };

	virtual bool on_begin(index<2> &idx) const {
		return true;
	}

	virtual bool on_next(index<2> &idx) const {
		m_dims.inc_index(idx);
		return m_dims.inc_index(idx);
	}
};

class bihandler : public block_iterator_handler_i<2, int> {
public:
	virtual bool on_begin(index<2> &idx, block_symop<2, int> &symop,
		const index<2> &orbit) const {

		idx = orbit;
		return true;
	}

	virtual bool on_next(index<2> &idx, block_symop<2, int> &symop,
		const index<2> &orbit) const {

		return false;
	}
};

} // namespace orbit_iterator_test_ns

namespace ns = orbit_iterator_test_ns;

void orbit_iterator_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
}

void orbit_iterator_test::test_1() throw(libtest::test_exception) {

	const char *testname = "orbit_iterator_test::test_1()";

	index<2> i0, i1, i2;
	i2[0] = 10; i2[1] = 20;
	dimensions<2> dims(index_range<2>(i1, i2));
	ns::bihandler bihandler;
	ns::handler1 handler1(dims);
	orbit_iterator<2, int> i(handler1, bihandler);
	bool end1 = false, end2;
	while(!i.end()) {
		if(!i.get_index().equals(i0)) {
			fail_test(testname, __FILE__, __LINE__,
				"Inconsistent indexes.");
		}
		end2 = !dims.inc_index(i0);
		if(end1 && end2) {
			fail_test(testname, __FILE__, __LINE__,	"Iterator "
				"goes beyond the end of the sequence.");
		}
		end1 = end2;
		i.next();
	}
	if(dims.inc_index(i0)) {
		fail_test(testname, __FILE__, __LINE__,
			"Iterations are incomplete.");
	}

}

void orbit_iterator_test::test_2() throw(libtest::test_exception) {

	const char *testname = "orbit_iterator_test::test_2()";

	index<2> i0, i1, i2;
	i2[0] = 10; i2[1] = 20;
	dimensions<2> dims(index_range<2>(i1, i2));
	ns::bihandler bihandler;
	ns::handler2 handler2(dims);
	orbit_iterator<2, int> i(handler2, bihandler);
	bool end1 = false, end2;
	while(!i.end()) {
		if(!i.get_index().equals(i0)) {
			fail_test(testname, __FILE__, __LINE__,
				"Inconsistent indexes.");
		}
		end2 = !(dims.inc_index(i0) && dims.inc_index(i0));
		if(end1 && end2) {
			fail_test(testname, __FILE__, __LINE__,	"Iterator "
				"goes beyond the end of the sequence.");
		}
		end1 = end2;
		i.next();
	}
	if(dims.inc_index(i0)) {
		fail_test(testname, __FILE__, __LINE__,
			"Iterations are incomplete.");
	}

}

} // namespace libtensor
