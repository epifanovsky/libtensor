#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/ctf_dense_tensor/ctf_symmetry.h>
#include "../compare_ref.h"
#include "ctf_symmetry_test.h"

namespace libtensor {


void ctf_symmetry_test::perform() throw(libtest::test_exception) {

    ctf::init();

    try {

        test_1();
        test_2();

    } catch(...) {
        ctf::exit();
        throw;
    }

    ctf::exit();
}


void ctf_symmetry_test::test_1() {

    static const char testname[] = "ctf_symmetry_test::test_1()";

    try {

    ctf_symmetry<4, double> sym;
    int s[4];
    int s_ref[4] = { NS, NS, NS, NS };

    for(int i = 0; i < 4; i++) s[i] = -999;

    sym.write(s);

    for(int i = 0; i < 4; i++) {
        if(s[i] != s_ref[i]) {
            std::ostringstream ss;
            ss << "Result doesn't match reference: "
               << "s[" << i << "] = " << s[i] << "; "
               << "s_ref[" << i << "] = " << s_ref[i];
            fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
        }
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_symmetry_test::test_2() {

    static const char testname[] = "ctf_symmetry_test::test_2()";

    typedef std_allocator<double> allocator_t;

    try {


    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

