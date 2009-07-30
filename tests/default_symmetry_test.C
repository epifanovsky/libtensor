#include <set>
#include <libtensor.h>
#include "default_symmetry_test.h"

namespace libtensor {

void default_symmetry_test::perform() throw(libtest::test_exception) {

	test_iterator();
}

void default_symmetry_test::test_iterator() throw(libtest::test_exception) {

	index<2> i1, i2;
	i2[0] = 4; i2[1] = 5;
	dimensions<2> dims(index_range<2>(i1, i2));

	default_symmetry<2, int> sym(dims);
	orbit_iterator<2, int> iter = sym.get_orbits();

	typedef std::set< index<2> > set_t;
	set_t idx;

	while(!iter.end()) {
		index<2> i(iter.get_index());
		if(idx.find(i) != idx.end()) {
			fail_test("default_symmetry_test::test_iterator()",
				__FILE__, __LINE__, "Repeated index detected");
		}
		idx.insert(i);
		iter.next();
	}

	if(idx.size() != dims.get_size()) {
		fail_test("default_symmetry_test::test_iterator()",
			__FILE__, __LINE__, "Incorrect total number of orbits");
	}

	index<2> ii;
	do {
		if(idx.find(ii) == idx.end()) {
			fail_test("default_symmetry_test::test_iterator()",
				__FILE__, __LINE__, "Incomplete orbit set");
		}
	} while(dims.inc_index(ii));

}

} // namespace libtensor
