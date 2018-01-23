#ifndef LIBTEST_TEST_SUITE_TEST_H
#define LIBTEST_TEST_SUITE_TEST_H

#include <libtest/test_suite.h>
#include <libtest/unit_test.h>

namespace libtest {


/** \brief Tests the test_suite class
 **/
class test_suite_test : public unit_test {
private:
    class test_1 : public unit_test {
    private:
        static bool m_ran;
    public:
        void perform() throw(test_exception) {
            m_ran = true;
        }
        static void reset() {
            m_ran = false;
        }
        static bool ran() {
            return m_ran;
        }
    };

    class test_2 : public unit_test {
    private:
        static bool m_ran;
    public:
        void perform() throw(test_exception) {
            m_ran = true;
        }
        static void reset() {
            m_ran = false;
        }
        static bool ran() {
            return m_ran;
        }
    };

    class test_suite_impl : public test_suite {
    private:
        unit_test_factory<test_1> m_utf_test_1;
        unit_test_factory<test_2> m_utf_test_2;

    public:
        test_suite_impl() : test_suite("test_suite_impl") {
            add_test("test_1", m_utf_test_1);
            add_test("test_2", m_utf_test_2);
        }
    };

public:
    virtual void perform() throw(test_exception);

};


} // namespace libtest

#endif // LIBTEST_TEST_SUITE_TEST_H

