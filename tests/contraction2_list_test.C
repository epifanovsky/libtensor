#include "contraction2_list_test.h"

namespace libtensor {

void contraction2_list_test::perform() throw (libtest::test_exception) {
	contraction2_list<4> list;
	if(list.get_length() != 0) {
		fail_test("contraction2_list_test::perform()", __FILE__,
			__LINE__, "New list is not empty");
	}

	list.append(10, 2, 3, 4);
	if(list.get_length() != 1) {
		fail_test("contraction2_list_test::perform()", __FILE__,
			__LINE__, "Incorrect length of the list");
	}
	size_t i = list.get_first();
	if(i != list.get_last()) {
		fail_test("contraction2_list_test::perform()", __FILE__,
			__LINE__, "Termination condition doesn't work");
	}
}

} // namespace libtensor

