#include <libtensor/iface/letter.h>
#include "letter_test.h"

namespace libtensor {

void letter_test::perform() throw(libtest::test_exception) {
	letter i, j, k, l;
	letter &i_ref(i);

	if(i != i) {
		fail_test("letter_test::perform()", __FILE__, __LINE__,
			"Failed comparison operator: i != i");
	}
	if(!(i == i)) {
		fail_test("letter_test::perform()", __FILE__, __LINE__,
			"Failed comparison operator: i == i");
	}
	if(i != i_ref) {
		fail_test("letter_test::perform()", __FILE__, __LINE__,
			"Failed comparison operator: i != i_ref");
	}
	if(!(i == i_ref)) {
		fail_test("letter_test::perform()", __FILE__, __LINE__,
			"Failed comparison operator: i == i_ref");
	}
	if(!(i != j)) {
		fail_test("letter_test::perform()", __FILE__, __LINE__,
			"Failed comparison operator: i != j");
	}
	if(i == j) {
		fail_test("letter_test::perform()", __FILE__, __LINE__,
			"Failed comparison operator: i == j");
	}

}

} // namespace libtensor

