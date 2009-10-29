#ifndef LIBTENSOR_TOD_MKDELTA_TEST_H
#define LIBTENSOR_TOD_MKDELTA_TEST_H

#include <libtest.h>

namespace libtensor {

class tod_mkdelta_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1(size_t ni, size_t na) throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_MKDELTA_TEST_H
