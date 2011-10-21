#ifndef LIBTENSOR_TOD_IMPORT_RAW_STREAM_TEST_H
#define LIBTENSOR_TOD_IMPORT_RAW_STREAM_TEST_H

#include <libtest/unit_test.h>
#include <libtensor/core/dimensions.h>

namespace libtensor {


/**	\brief Tests the libtensor::tod_import_raw_stream class

    \ingroup libtensor_tests_tod
 **/
class tod_import_raw_stream_test: public libtest::unit_test {
public:
    virtual void perform() throw (libtest::test_exception);

private:
    template<size_t N>
    void test_1(const dimensions<N> &dims, const index_range<N> &ir)
        throw (libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_IMPORT_RAW_STREAM_TEST_H
