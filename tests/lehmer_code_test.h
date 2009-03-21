#ifndef LIBTENSOR_LEHMER_CODE_TEST_H
#define LIBTENSOR_LEHMER_CODE_TEST_H

#include <libtest.h>
#include "lehmer_code.h"

namespace libtensor {

/**	\brief Tests the libtensor::lehmer_code class

	\ingroup libtensor_tests
**/
class lehmer_code_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	//!	Tests the code for a given %tensor order
	template<size_t N>
	void test_code() throw(libtest::test_exception);
};

template<size_t N>
void lehmer_code_test::test_code() throw(libtest::test_exception) {
	size_t fact = 1;
	for(size_t i=2; i<=N; i++) fact*=i; 

	permutation<N> *plist[fact];
	bool failed = false, fail_unique = false, fail_cons = false;
	size_t code, code_cons;
	for(code=0; code<fact && !failed; code++) {
		permutation<N> *p = new permutation<N>(
			lehmer_code<N>::get_instance().code2perm(code));

		// Check that each code belongs to a unique permutation
		fail_unique = false;
		for(size_t i=0; i<code && !fail_unique; i++) {
			if(plist[i]->equals(*p)) fail_unique = true;
		}

		// Check if reverse conversion is consistent
		if(!fail_unique) {
			size_t c = lehmer_code<N>::get_instance().perm2code(*p);
			if(c != code) {
				fail_cons = true;
				code_cons = c;
			}
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
		snprintf(s, 1024,
			"Permutation code %lu is not consistent (%lu)",
			code-1, code_cons);
		fail_test("lehmer_code_test::test_code(const size_t)",
			__FILE__, __LINE__, s);
	}
}

} // namespace libtensor

#endif // LIBTENSOR_LEHMER_CODE_TEST_H

