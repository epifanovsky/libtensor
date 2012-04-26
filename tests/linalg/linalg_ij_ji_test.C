#include <sstream>
#include <libtensor/exception.h>
#include <libtensor/linalg/linalg.h>
#include "linalg_ij_ji_test.h"

namespace libtensor {


void linalg_ij_ji_test::perform() throw(libtest::test_exception) {

    test_ij_ji(1, 1, 1, 1);
    test_ij_ji(1, 2, 1, 2);
    test_ij_ji(2, 1, 2, 1);
    test_ij_ji(16, 16, 16, 16);
    test_ij_ji(3, 17, 5, 17);
    test_ij_ji(2, 2, 2, 3);
    test_ij_ji(2, 2, 4, 3);
}


void linalg_ij_ji_test::test_ij_ji(size_t ni, size_t nj, size_t sja,
    size_t sic) throw(libtest::test_exception) {

    std::ostringstream ss;
    ss << "linalg_ij_ji_test::test_ij_ji("
        << ni << ", " << nj << ", " << sja << ", " << sic << ")";
    std::string tnss = ss.str();

    double *a = 0, *c = 0, *c_ref = 0;
    double d = 0.0;

    try {

    size_t sza = nj * sja, szc = ni * sic;

    a = new double[sza];
    c = new double[szc];
    c_ref = new double[szc];

    for(size_t i = 0; i < sza; i++) a[i] = drand48();
    for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

    d = 0.0;
    linalg::ij_ji(ni, nj, a, sja, c, sic);
    linalg_base_generic::ij_ji(ni, nj, a, sja, c_ref, sic);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__,
                "Incorrect result.");
        }
    }

    delete [] a; a = 0;
    delete [] c; c = 0;
    delete [] c_ref; c_ref = 0;

    } catch(exception &e) {
        delete [] a; a = 0;
        delete [] c; c = 0;
        delete [] c_ref; c_ref = 0;
        fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
    } catch(...) {
        delete [] a; a = 0;
        delete [] c; c = 0;
        delete [] c_ref; c_ref = 0;
        throw;
    }
}


} // namespace libtensor
