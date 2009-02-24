#include "immutable_test.h"

namespace libtensor {

void immutable_test::perform() throw(libtest::test_exception) {
	immutable im;
	if(im.is_immutable()) {
		fail_test("immutable_test::perform()", __FILE__, __LINE__,
			"New object must be mutable");
	}
	im.set_immutable();
	if(!im.is_immutable()) {
		fail_test("immutable_test::perform()", __FILE__, __LINE__,
			"set_immutable() failed");
	}
}

} // namespace libtensor

