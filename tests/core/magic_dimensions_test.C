#include <sstream>
#include <libtensor/core/print_dimensions.h>
#include <libtensor/core/impl/magic_dimensions_impl.h>
#include "magic_dimensions_test.h"

namespace libtensor {


void magic_dimensions_test::perform() throw(libtest::test_exception) {

    test_1();
//    test_2();
//    test_3();
}


void magic_dimensions_test::test_1() {

    static const char *testname = "magic_dimensions_test::test_1";

    try {

        for(size_t i = 0; i < 16; i++) {

            index<2> i1, i2;
            i2[0] = i; i2[1] = i + 1;
            dimensions<2> dims(index_range<2>(i1, i2));
            magic_dimensions<2> mdims(dims, true);

            size_t aj = (i * i + i + 1) / 2;
            index<2> j, j_ref;
            j_ref[0] = aj / (i + 1);
            j_ref[1] = aj - j_ref[0] * (i + 1);
            j[0] = aj / mdims[0];
            j[1] = aj  - j[0] * dims[0];

            if(!j.equals(j_ref)) {
                std::ostringstream ss;
                ss << "Bad conversion " << j_ref << " -> " << aj << " -> " << j
                    << " within " << dims;
                fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
            }
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
