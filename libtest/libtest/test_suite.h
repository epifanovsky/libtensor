#ifndef LIBTEST_TEST_SUITE_H
#define LIBTEST_TEST_SUITE_H

#include <map>
#include <string>

#include "unit_test_factory.h"
#include "suite_event_handler.h"

namespace libtest {

/** \brief Suite of unit tests
    \ingroup libtest
 **/
class test_suite {
private:
    std::string m_name; //!< Test suite name
    std::map<std::string,unit_test_factory_i*> m_tests; //!< All tests
    suite_event_handler *m_handler; //!< Event handler

protected:
    /** \brief Adds a unit test to the suite
        \param name Test name
        \param utf Unit test factory
     **/
    void add_test(const char *name, unit_test_factory_i &utf);

public:
    /** \brief Constructor
        \param name Test suite name
     **/
    test_suite(const char *name);

    /** \brief Virtual destructor
     **/
    virtual ~test_suite();

    /** \brief Returns the number of tests in the suite
     **/
    unsigned get_num_tests() const;

    /** \brief Runs a single test from the suite
        \param name Test name
     **/
    unsigned run_test(const char *name);

    /** \brief Runs all the tests in the suite
     **/
    unsigned run_all_tests();

    /** \brief Sets the event handler
     **/
    void set_handler(suite_event_handler *handler);

private:
    void evt_suite_start();
    void evt_suite_end();
    void evt_test_start(const char *test);
    void evt_test_end_success(const char *test);
    void evt_test_end_exception(const char *test, const test_exception &exc);

};


} // namespace libtest

#endif // LIBTEST_TEST_SUITE_H

