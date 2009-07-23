#include <libtensor.h>
#include "perm_symmetry_test.h"

namespace libtensor {

void perm_symmetry_test::perform() throw(libtest::test_exception) {

	test_is_same();
}

void perm_symmetry_test::test_is_same() throw(libtest::test_exception) {

	const char *testname = "perm_symmetry_test::test_is_same()";

	perm_symmetry<4, double> psym1, psym2;
	symmetry_i<4, double> &sym1 = psym1, &sym2 = psym2;

	if(!psym1.is_same(psym2)) {
		fail_test(testname, __FILE__, __LINE__,
			"!psym1.is_same(psym2)");
	}
	if(!sym1.is_same(sym2)) {
		fail_test(testname, __FILE__, __LINE__,	"!sym1.is_same(sym2)");
	}
}

} // namespace libtensor
