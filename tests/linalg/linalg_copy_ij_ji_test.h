#ifndef LIBTENSOR_LINALG_COPY_IJ_JI_TEST_H
#define LIBTENSOR_LINALG_COPY_IJ_JI_TEST_H

#include "linalg_test_base.h"

namespace libtensor {


/** \brief Tests the libtensor::linalg class (copy_ij_ji)

    \ingroup libtensor_tests_linalg
 **/
class linalg_copy_ij_ji_test : public linalg_test_base {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_copy_ij_ji(size_t ni, size_t nj, size_t sja, size_t sic);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_COPY_IJ_JI_TEST_H
