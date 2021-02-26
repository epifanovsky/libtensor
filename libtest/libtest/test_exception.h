#ifndef LIBTEST_TEST_EXCEPTION_H
#define LIBTEST_TEST_EXCEPTION_H

#include <exception>

namespace libtest {


/** \brief Passes along information about exceptions
    \ingroup libtest

    Using details about an exception, this class creates a message of
    the format: <tt>[routine (srcfile, lineno)] message</tt>
 **/
class test_exception : public std::exception {
private:
    char m_what[1024];

public:
    /** \brief Creates an exception
        \param where Routine name (class and method) where the exception occured
        \param src Name of the source file (via __FILE__)
        \param lineno Line number (via __LINE__)
        \param what Description of the exception
     **/
    test_exception(const char *where, const char *src, unsigned lineno,
        const char *what);

    /** \brief Virtual destructor
     **/
    virtual ~test_exception() throw();

    /** \brief Returns the cause of the exception
     **/
    virtual const char *what() const throw() {
        return m_what;
    }

};


} // namespace libtest

#endif // LIBTEST_TEST_EXCEPTION_H

