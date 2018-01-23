#ifndef LIBUTIL_THREAD_POOL_TEST_H
#define LIBUTIL_THREAD_POOL_TEST_H

#include <libtest/unit_test.h>

namespace libutil {


/** \brief Tests the libutil::thread_pool class

     \ingroup libutil_tests
 **/
class thread_pool_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1a() throw(libtest::test_exception);
    void test_1b() throw(libtest::test_exception);
    void test_2(size_t n, size_t nthr, size_t ncpu)
        throw(libtest::test_exception);
    void test_2_serial(size_t n) throw(libtest::test_exception);
    void test_3(size_t n, size_t nthr, size_t ncpu)
        throw(libtest::test_exception);
    void test_3_serial(size_t n) throw(libtest::test_exception);
    void test_4(size_t n, size_t nthr, size_t ncpu)
        throw(libtest::test_exception);
    void test_4_serial(size_t n) throw(libtest::test_exception);
    void test_5(size_t nthr, size_t ncpu) throw(libtest::test_exception);

    void test_exc_1(size_t n, size_t nthr, size_t ncpu);
    void test_exc_1_serial(size_t n);
    void test_exc_2(size_t n, size_t nthr, size_t ncpu);
    void test_exc_2_serial(size_t n);

};


} // namespace libutil

#endif // LIBUTIL_THREAD_POOL_TEST_H
