#include <symmetry/symmetry_element_set_adapter.h>
#include "symmetry_element_set_adapter_test.h"

namespace libtensor {


void symmetry_element_set_adapter_test::perform()
	throw(libtest::test_exception) {

	test_1();
	test_2();
}


namespace symmetry_element_set_adapter_test_ns {

template<size_t N>
class sym_elem_1 : public symmetry_element_i<N, double> {
private:
	size_t m_m, m_n;

public:
	sym_elem_1(size_t m) : m_m(m), m_n(0) { }
	virtual ~sym_elem_1() { }
	virtual const char *get_type() const { return "sym_elem_1"; }
	virtual const mask<N> &get_mask() const { throw 0; }
	virtual void permute(const permutation<N> &perm) { throw 0; }
	virtual bool is_valid_bis(const block_index_space<N> &bis) const {
		throw 0;
	}
	virtual bool is_allowed(const index<N> &idx) const { throw 0; }
	virtual void apply(index<N> &idx) const { throw 0; }
	virtual void apply(index<N> &idx, transf<N, double> &tr) const {
		throw 0;
	}
	virtual symmetry_element_i<N, double> *clone() const {
		return new sym_elem_1(m_m, m_n + 1);
	}
	size_t get_m() const { return m_m; }
	size_t get_n() const { return m_n; }

private:
	sym_elem_1(size_t m, size_t n) : m_m(m), m_n(n) { }
};

}
using namespace symmetry_element_set_adapter_test_ns;


/**	\test Tests the construction and iterators on the empty set
 **/
void symmetry_element_set_adapter_test::test_1()
	throw(libtest::test_exception) {

	static const char *testname =
		"symmetry_element_set_adapter_test::test_1()";

	typedef symmetry_element_set<2, double> symmetry_element_set_t;
	typedef symmetry_element_set_adapter< 2, double, sym_elem_1<2> >
		symmetry_element_set_adapter_t;

	try {

	symmetry_element_set_t set("sym_elem_1");
	symmetry_element_set_adapter_t adapter(set);

	symmetry_element_set_adapter_t::iterator i = adapter.begin();
	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"i != adapter.end() in empty set.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the type conversion by the adapter
 **/
void symmetry_element_set_adapter_test::test_2()
	throw(libtest::test_exception) {

	static const char *testname =
		"symmetry_element_set_adapter_test::test_2()";

	typedef symmetry_element_set<2, double> symmetry_element_set_t;
	typedef symmetry_element_set_adapter< 2, double, sym_elem_1<2> >
		symmetry_element_set_adapter_t;

	try {

	symmetry_element_set_t set("sym_elem_1");
	sym_elem_1<2> elem1(1), elem2(2);
	set.insert(elem1);
	set.insert(elem2);
	symmetry_element_set_adapter_t adapter(set);

	symmetry_element_set_adapter_t::iterator i = adapter.begin();
	if(i == adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"i == adapter.end() in non-empty set.");
	}
	size_t m1 = adapter.get_elem(i).get_m();
	i++;

	if(i == adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Unexpected i == adapter.end()");
	}
	size_t m2 = adapter.get_elem(i).get_m();
	i++;

	if(i != adapter.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Unexpected i != adapter.end()");
	}

	if(!(m1 == 1 && m2 == 2) && !(m1 == 2 && m2 == 1)) {
		fail_test(testname, __FILE__, __LINE__,
			"Unexpected symmetry elements returned.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor

