#include <cstdio>
#include "lehmer_code_test.h"

namespace libtensor {

void lehmer_code_test::perform() throw(libtest::test_exception) {
	test_code(2);
	test_code(3);
	test_code(4);
	test_code(5);
	test_code(6);
}

void lehmer_code_test::test_code(const size_t order)
	throw(libtest::test_exception) {

	size_t fact = 1;
	for(size_t i=2; i<=order; i++) fact*=i; 

	permutation *plist[fact];
	bool failed = false, fail_unique, fail_cons;
	size_t code;
	for(code=0; code<fact && !failed; code++) {
		permutation *p = new permutation(
			lehmer_code::get_instance().code2perm(code));

		// Check that each code belongs to a unique permutation
		fail_unique = false;
		for(size_t i=0; i<code && !fail_unique; i++) {
			if(plist[i]->equals(*p)) fail_unique = true;
		}

		// Check if reverse conversion is consistent
		if(!fail_unique) {
			size_t c = lehmer_code::get_instance().perm2code(*p);
			if(c != code) fail_cons = true;
		}
		failed = fail_unique || fail_cons;
		plist[code] = p;
	}

	for(size_t i=0; i<code; i++) delete plist[i];

	if(fail_unique) {
		char s[1024];
		snprintf(s, 1024, "Permutation code %lu is not unique", code-1);
		fail_test("lehmer_code_test::test_code(const size_t)",
			__FILE__, __LINE__, s);
	}
	if(fail_cons) {
		char s[1024];
		snprintf(s, 1024, "Permutation code %lu is not consistent",
			code-1);
		fail_test("lehmer_code_test::test_code(const size_t)",
			__FILE__, __LINE__, s);
	}
}

} // namespace libtensor

