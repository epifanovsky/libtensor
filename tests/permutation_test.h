#ifndef __LIBTENSOR_PERMUTATION_TEST_H
#define __LIBTENSOR_PERMUTATION_TEST_H

#include <libtest.h>

namespace libtensor {

template<class Perm>
class permutation_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);
};

template<class Perm>
void permutation_test<Perm>::perform() throw(libtest::test_exception) {
}

}

#endif // __LIBTENSOR_PERMUTATION_TEST_H

