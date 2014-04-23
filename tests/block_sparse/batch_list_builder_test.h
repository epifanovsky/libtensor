#ifndef BATCH_LIST_BUILDER_TEST_H
#define BATCH_LIST_BUILDER_TEST_H

#include <libtest/unit_test.h>

namespace libtensor
{

class batch_list_builder_test : public libtest::unit_test
{
public:
    virtual void perform() throw(libtest::test_exception);
private:
    void test_get_batch_list_dense() throw(libtest::test_exception);
};

} /* namespace batch_list_builder */


#endif /* BATCH_LIST_BUILDER_TEST_H */
