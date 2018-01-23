#ifndef LIBTEST_SUITE_EVENT_HANDLER_H
#define LIBTEST_SUITE_EVENT_HANDLER_H

#include "test_exception.h"

namespace libtest {


/** \brief Test suite event handlers: base class
 **/
class suite_event_handler {
public:
    virtual void on_suite_start(const char *suite) = 0;
    virtual void on_suite_end(const char *suite) = 0;
    virtual void on_test_start(const char *test) = 0;
    virtual void on_test_end_success(const char *test) = 0;
    virtual void on_test_end_exception(const char *test,
        const test_exception &exc) = 0;

};


} // namespace libtest

#endif // LIBTEST_SUITE_EVENT_HANDLER_H

