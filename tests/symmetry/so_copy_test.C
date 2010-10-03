#include <typeinfo>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/btod/transf_double.h>
#include "so_copy_test.h"

namespace libtensor {


void so_copy_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
}


namespace so_copy_test_ns {


class symelem1 : public symmetry_element_i<4, double> {
public:
	static const char *k_typ;
private:
	size_t n;
public:
	symelem1(size_t n_) : n(n_) { }
	symelem1(const symelem1 &e) : n(e.n) { }
	virtual ~symelem1() { }
	virtual const char *get_type() const { return k_typ; }
	virtual symmetry_element_i<4, double> *clone() const {
		return new symelem1(*this);
	}
	virtual const mask<4> &get_mask() const { throw 0; }
	virtual void permute(const permutation<4>&) { throw 0; }
	virtual bool is_valid_bis(const block_index_space<4>&) const {
		return true;
	}
	virtual bool is_allowed(const index<4>&) const { throw 0; }
	virtual void apply(index<4>&) const { throw 0; }
	virtual void apply(index<4>&, transf<4, double>&) const { throw 0; }
	size_t get_n() const { return n; }
};


class symelem2 : public symmetry_element_i<4, double> {
public:
	static const char *k_typ;
private:
	size_t m;
public:
	symelem2(size_t m_) : m(m_) { }
	symelem2(const symelem2 &e) : m(e.m) { }
	virtual ~symelem2() { }
	virtual const char *get_type() const { return k_typ; }
	virtual symmetry_element_i<4, double> *clone() const {
		return new symelem2(*this);
	}
	virtual const mask<4> &get_mask() const { throw 0; }
	virtual void permute(const permutation<4>&) { throw 0; }
	virtual bool is_valid_bis(const block_index_space<4>&) const {
		return true;
	}
	virtual bool is_allowed(const index<4>&) const { throw 0; }
	virtual void apply(index<4>&) const { throw 0; }
	virtual void apply(index<4>&, transf<4, double>&) const { throw 0; }
	size_t get_m() const { return m; }
};


const char *symelem1::k_typ = "symelem1";
const char *symelem2::k_typ = "symelem2";


} // namespace so_copy_test_ns
using namespace so_copy_test_ns;


/**	\test Copy of empty %symmetry in 4-space.
 **/
void so_copy_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_copy_test::test_1()";

	try {

	index<4> i1, i2;
	i2[0] = 5; i2[1] = 5; i2[2] = 10; i2[3] = 10;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));

	symmetry<4, double> sym1(bis), sym2(bis), sym3(bis);

	symelem1 elem1(1);
	sym3.insert(elem1);

	so_copy<4, double>(sym1).perform(sym2);
	so_copy<4, double>(sym1).perform(sym3);

	symmetry<4, double>::iterator j2 = sym2.begin();
	if(j2 != sym2.end()) {
		fail_test(testname, __FILE__, __LINE__, "j2 != sym2.end()");
	}

	symmetry<4, double>::iterator j3 = sym3.begin();
	if(j3 != sym3.end()) {
		fail_test(testname, __FILE__, __LINE__, "j3 != sym3.end()");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Copy of %symmetry that contains one element in 4-space.
 **/
void so_copy_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "so_copy_test::test_2()";

	typedef symmetry_element_set<4, double> symmetry_element_set_t;

	try {

	index<4> i1, i2;
	i2[0] = 5; i2[1] = 5; i2[2] = 10; i2[3] = 10;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));

	symmetry<4, double> sym1(bis), sym2(bis), sym3(bis);

	symelem1 elem1(22);
	symelem2 elem2(1);
	sym1.insert(elem1);
	sym3.insert(elem2);

	so_copy<4, double>(sym1).perform(sym2);
	so_copy<4, double>(sym1).perform(sym3);

	symmetry<4, double>::iterator j2 = sym2.begin();
	if(j2 == sym2.end()) {
		fail_test(testname, __FILE__, __LINE__, "j2 == sym2.end()");
	}
	if(sym2.get_subset(j2).get_id().compare(symelem1::k_typ) != 0) {
		fail_test(testname, __FILE__, __LINE__,
			"Bad symmetry id in sym2.");
	}

	const symmetry_element_set_t &subset21 = sym2.get_subset(j2);
	symmetry_element_set_t::const_iterator jj2 = subset21.begin();
	if(jj2 == subset21.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"jj2 == subset21.end()");
	}
	try {
		const symelem1 &elem1i = dynamic_cast<const symelem1&>(
			subset21.get_elem(jj2));
		if(elem1i.get_n() != 22) {
			fail_test(testname, __FILE__, __LINE__, "Bad elem1i.");
		}
	} catch(std::bad_cast &e) {
		fail_test(testname, __FILE__, __LINE__, "bad_cast for elem1i");
	}

	jj2++;
	if(jj2 != subset21.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"jj2 != subset21.end()");
	}

	j2++;
	if(j2 != sym2.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one subset in sym2.");
	}

	symmetry<4, double>::iterator j3 = sym3.begin();
	if(j3 == sym3.end()) {
		fail_test(testname, __FILE__, __LINE__, "j3 == sym3.end()");
	}
	if(sym3.get_subset(j3).get_id().compare(symelem1::k_typ) != 0) {
		fail_test(testname, __FILE__, __LINE__,
			"Bad symmetry id in sym3.");
	}

	const symmetry_element_set_t &subset31 = sym3.get_subset(j3);
	symmetry_element_set_t::const_iterator jj3 = subset31.begin();
	if(jj3 == subset31.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"jj3 == subset31.end()");
	}
	try {
		const symelem1 &elem1i = dynamic_cast<const symelem1&>(
			subset31.get_elem(jj3));
		if(elem1i.get_n() != 22) {
			fail_test(testname, __FILE__, __LINE__, "Bad elem1i.");
		}
	} catch(std::bad_cast &e) {
		fail_test(testname, __FILE__, __LINE__, "bad_cast for elem1i");
	}

	jj3++;
	if(jj3 != subset31.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"jj3 != subset31.end()");
	}

	j3++;
	if(j3 != sym3.end()) {
		fail_test(testname, __FILE__, __LINE__,
			"Expected only one subset in sym3.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


void so_copy_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "so_copy_test::test_3()";

	try {

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


} // namespace libtensor
