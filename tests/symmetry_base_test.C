#include <libtensor.h>
#include "symmetry_base_test.h"

namespace libtensor {

namespace symmetry_base_test_ns {

class sym1;
class sym2;

class sym1 : public symmetry_base<1, int, sym1> {
public:
	virtual ~sym1() { }

	virtual void disable_symmetry() {
		throw 0;
	}

	virtual void enable_symmetry() {
		throw 0;
	}

	virtual bool is_canonical(const index<1> &idx) const {
		throw 0;
	}

	virtual orbit_iterator<1, int> get_orbits() const {
		throw 0;
	}

	virtual const orbit_iterator_handler_i<1, int> &get_oi_handler() const {
		throw 0;
	}

	virtual const block_iterator_handler_i<1, int> &get_bi_handler() const {
		throw 0;
	}
};

class sym2 : public symmetry_base<1, int, sym2> {
public:
	virtual ~sym2() { }

	virtual void disable_symmetry() {
		throw 0;
	}

	virtual void enable_symmetry() {
		throw 0;
	}

	virtual bool is_canonical(const index<1> &idx) const {
		throw 0;
	}

	virtual orbit_iterator<1, int> get_orbits() const {
		throw 0;
	}

	virtual const orbit_iterator_handler_i<1, int> &get_oi_handler() const {
		throw 0;
	}

	virtual const block_iterator_handler_i<1, int> &get_bi_handler() const {
		throw 0;
	}
};

class tgt1 : public symmetry_const_target<1, int, sym1> {
private:
	bool m_flag;

public:
	tgt1() : m_flag(false) { }

	virtual void accept(const sym1 &sym) throw(exception) {
		m_flag = true;
	}

	void reset_flag() { m_flag = false; }
	bool flag() const { return m_flag; }
};

} // namespace symmetry_base_test_ns

namespace ns = symmetry_base_test_ns;

void symmetry_base_test::perform() throw(libtest::test_exception) {
	ns::sym1 s1;
	ns::sym2 s2;
	ns::tgt1 t1;

	const symmetry_i<1, int> &pcs1 = s1, &pcs2 = s2;
	symmetry_i<1, int> &ps1 = s1, &ps2 = s2;
	symmetry_target_i<1, int> &pt1 = t1;

	pcs1.dispatch(pt1);
	if(!t1.flag()) {
		fail_test("symmetry_base_test::perform()", __FILE__, __LINE__,
			"Failed dispatch 1");
	}
	t1.reset_flag();
	ps1.dispatch(pt1);
	if(!t1.flag()) {
		fail_test("symmetry_base_test::perform()", __FILE__, __LINE__,
			"Failed dispatch 2");
	}
	t1.reset_flag();
	ps2.dispatch(pt1);
	if(t1.flag()) {
		fail_test("symmetry_base_test::perform()", __FILE__, __LINE__,
			"Failed dispatch 3");
	}
}

} // namespace libtensor
