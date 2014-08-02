#ifndef MEMORY_RESERVE_TEST_H
#define MEMORY_RESERVE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor
{

class memory_reserve_test: public libtest::unit_test
{
public:
    virtual void perform() throw(libtest::test_exception);
private:
};

} /* namespace libtensor */


#endif /* MEMORY_RESERVE_TEST_H */
