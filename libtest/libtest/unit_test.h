#ifndef LIBTEST_UNIT_TEST_H
#define LIBTEST_UNIT_TEST_H

#include "test_exception.h"

namespace libtest {

/** \brief Base class for unit tests
    \ingroup libtest

    This class is to be used as a base class for all unit tests.
    Deriving classes must implement the perform() method.

    If the test fails, the implementation should throw a test_exception
    via fail_test() with all the necessary information. Otherwise, it
    should return cleanly. No other exceptions are permitted.

    Example:
    \code
    using libtest::test_exception;
    using libtest::unit_test;

    class addition_t : public unit_test {
    public:
        virtual void perform() throw(test_exception);
    };

    void addition_t::perform() throw(test_exception) {
        double a = 2.0;
        double b = 3.0;
        double c = 5.0;

        double d = a + b;
        if(d != c) fail_test("addition_t::perform", __FILE__, __LINE__,
            "d = a + b");
    }
    \endcode    
 **/
class unit_test {
protected:
    /** \brief Fails the test
        \param where Test routine that failed the test
        \param src Source file (via __FILE__)
        \param lineno Line number (via __LINE__)
        \param what Failure cause or message

        This method throws a test_exception using the provided
        information. It never returns to the calling routine, so
        necessary cleanup like freeing memory may be necessary prior to
        calling fail_test().
     **/
    void fail_test(const char *where, const char *src, unsigned lineno,
        const char *what) throw(test_exception);

public:
    /** \brief Virtual destructor
     **/
    virtual ~unit_test() { }

    /** \brief Performs the test (to be implemented by derived classes)
     **/
    virtual void perform() throw(test_exception) = 0;

};


} // namespace libtest

#endif // LIBTEST_UNIT_TEST_H

