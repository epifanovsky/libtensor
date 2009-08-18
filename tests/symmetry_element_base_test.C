#include <libtensor.h>
#include "symmetry_element_base_test.h"

namespace libtensor {

namespace symmetry_element_base_test_ns {

class symel1;
class symel2;

class symel1 : public symmetry_element_base<1, int, symel1> {
public:
	virtual ~symel1() { }

	virtual const mask<1> &get_mask() const { throw 0; }
	virtual void permute(const permutation<1> &perm) { throw 0; }
	virtual bool is_allowed(const index<1> &idx) const { throw 0; }
	virtual void apply(index<1> &idx) const { throw 0; }
	virtual void apply(index<1> &idx, transf<1, int> &tr) const { throw 0; }
	virtual bool equals(const symmetry_element_i<1, int> &se) const {
		throw 0;
	}
	virtual symmetry_element_i<1, int> *clone() const {
		return new symel1;
	}
};

class symel2 : public symmetry_element_base<1, int, symel2> {
public:
	virtual ~symel2() { }

	virtual const mask<1> &get_mask() const { throw 0; }
	virtual void permute(const permutation<1> &perm) { throw 0; }
	virtual bool is_allowed(const index<1> &idx) const { throw 0; }
	virtual void apply(index<1> &idx) const { throw 0; }
	virtual void apply(index<1> &idx, transf<1, int> &tr) const { throw 0; }
	virtual bool equals(const symmetry_element_i<1, int> &se) const {
		throw 0;
	}
	virtual symmetry_element_i<1, int> *clone() const {
		return new symel2;
	}
};

class tgt1 : public symmetry_element_target<1, int, symel1> {
private:
	bool m_flag;

public:
	tgt1() : m_flag(false) { }

	virtual void accept(const symel1 &sym) throw(exception) {
		m_flag = true;
	}

	void reset_flag() { m_flag = false; }
	bool flag() const { return m_flag; }
};

} // namespace symmetry_element_base_test_ns

namespace ns = symmetry_element_base_test_ns;

void symmetry_element_base_test::perform() throw(libtest::test_exception) {

	static const char *testname = "symmetry_base_test::perform()";
	ns::symel1 se1;
	ns::symel2 se2;
	ns::tgt1 t1;

	symmetry_element_i<1, int> &pse1 = se1, &pse2 = se2;
	symmetry_element_target_i<1, int> &pt1 = t1;

	pse1.dispatch(pt1);
	if(!t1.flag()) {
		fail_test(testname, __FILE__, __LINE__, "Failed dispatch 1");
	}
	t1.reset_flag();
	pse1.dispatch(pt1);
	if(!t1.flag()) {
		fail_test(testname, __FILE__, __LINE__, "Failed dispatch 2");
	}
	t1.reset_flag();
	pse2.dispatch(pt1);
	if(t1.flag()) {
		fail_test(testname, __FILE__, __LINE__, "Failed dispatch 3");
	}
}

} // namespace libtensor
