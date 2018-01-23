#ifndef LIBTEST_LIBTEST_SUITE_H
#define LIBTEST_LIBTEST_SUITE_H

#include <libtest/libtest.h>
#include "test_suite_test.h"

namespace libtest {


/** \brief Suite of tests on libtest
 **/
class libtest_suite : public test_suite {
private:
    unit_test_factory<test_suite_test> m_utf_test_suite;

public:
    /** \brief Initializes the suite
     **/
    libtest_suite();
};


} // namespace libtest

#endif // LIBTEST_LIBTEST_SUITE_H
