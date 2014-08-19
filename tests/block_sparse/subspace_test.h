#ifndef SUBSPACE_TEST_H
#define SUBSPACE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

class subspace_test : public libtest::unit_test 
{
public:
    virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor



#endif /* SUBSPACE_TEST_H */
