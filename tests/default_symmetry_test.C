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
	const orbit_iterator_handler_i<2, int> &oih = sym.get_oi_handler();

	typedef std::set< index<2> > set_t;
	set_t idx;
	index<2> ii;
	bool end = !oih.on_begin(ii);

	while(!end) {
		if(idx.find(ii) != idx.end()) {
			fail_test("default_symmetry_test::test_iterator()",
				__FILE__, __LINE__, "Repeated index detected");
		}
		idx.insert(ii);
		end = !oih.on_next(ii);
	}

	if(idx.size() != dims.get_size()) {
		fail_test("default_symmetry_test::test_iterator()",
			__FILE__, __LINE__, "Incorrect total number of orbits");
	}

	ii[0] = 0; ii[1] = 0;
	do {
		if(idx.find(ii) == idx.end()) {
			fail_test("default_symmetry_test::test_iterator()",
				__FILE__, __LINE__, "Incomplete orbit set");
		}
	} while(dims.inc_index(ii));

}

} // namespace libtensor
