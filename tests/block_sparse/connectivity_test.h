#ifndef CONNECTIVITY_TEST_H
#define CONNECTIVITY_TEST_H

#include <libtest/unit_test.h>

namespace libtensor
{

class connectivity_test: public libtest::unit_test
{
public:
    virtual void perform() throw(libtest::test_exception);
private:
};

} /* namespace libtensor */


#endif /* CONNECTIVITY_TEST_H */
