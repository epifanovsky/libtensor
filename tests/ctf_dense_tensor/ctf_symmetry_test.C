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
        test_3();
        test_4();
        test_5();

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

    sequence<4, unsigned> grp(0), symind(0);
    grp[0] = 0; grp[1] = 0; grp[2] = 1; grp[3] = 1;
    ctf_symmetry<4, double> sym(grp, symind);
    int s[4];
    int s_ref_1[4] = { SY, NS, SY, NS };
    int s_ref_2[4] = { NS, NS, NS, NS };
    int s_ref_3[4] = { SY, NS, SY, NS };

    for(int i = 0; i < 4; i++) s[i] = -999;
    sym.write(s);

    for(int i = 0; i < 4; i++) {
        if(s[i] != s_ref_1[i]) {
            std::ostringstream ss;
            ss << "Result doesn't match reference (1): "
               << "s[" << i << "] = " << s[i] << "; "
               << "s_ref_1[" << i << "] = " << s_ref_1[i];
            fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
        }
    }

    sym.permute(permutation<4>().permute(1, 2));
    for(int i = 0; i < 4; i++) s[i] = -999;
    sym.write(s);

    for(int i = 0; i < 4; i++) {
        if(s[i] != s_ref_2[i]) {
            std::ostringstream ss;
            ss << "Result doesn't match reference (2): "
               << "s[" << i << "] = " << s[i] << "; "
               << "s_ref_2[" << i << "] = " << s_ref_2[i];
            fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
        }
    }

    sym.permute(permutation<4>().permute(1, 2));
    for(int i = 0; i < 4; i++) s[i] = -999;
    sym.write(s);

    for(int i = 0; i < 4; i++) {
        if(s[i] != s_ref_3[i]) {
            std::ostringstream ss;
            ss << "Result doesn't match reference (3): "
               << "s[" << i << "] = " << s[i] << "; "
               << "s_ref_3[" << i << "] = " << s_ref_3[i];
            fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
        }
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_symmetry_test::test_3() {

    static const char testname[] = "ctf_symmetry_test::test_3()";

    typedef std_allocator<double> allocator_t;

    try {

    sequence<4, unsigned> grp(0), symind(0);
    grp[0] = 0; grp[1] = 0; grp[2] = 1; grp[3] = 1;
    symind[0] = 1; symind[1] = 0;
    ctf_symmetry<4, double> sym(grp, symind);
    int s[4];
    int s_ref_1[4] = { SY, NS, AS, NS };
    int s_ref_2[4] = { NS, NS, NS, NS };
    int s_ref_3[4] = { SY, NS, AS, NS };
    int s_ref_4[4] = { AS, NS, SY, NS };

    for(int i = 0; i < 4; i++) s[i] = -999;
    sym.write(s);

    for(int i = 0; i < 4; i++) {
        if(s[i] != s_ref_1[i]) {
            std::ostringstream ss;
            ss << "Result doesn't match reference (1): "
               << "s[" << i << "] = " << s[i] << "; "
               << "s_ref_1[" << i << "] = " << s_ref_1[i];
            fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
        }
    }

    sym.permute(permutation<4>().permute(1, 2));
    for(int i = 0; i < 4; i++) s[i] = -999;
    sym.write(s);

    for(int i = 0; i < 4; i++) {
        if(s[i] != s_ref_2[i]) {
            std::ostringstream ss;
            ss << "Result doesn't match reference (2): "
               << "s[" << i << "] = " << s[i] << "; "
               << "s_ref_2[" << i << "] = " << s_ref_2[i];
            fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
        }
    }

    sym.permute(permutation<4>().permute(1, 2));
    for(int i = 0; i < 4; i++) s[i] = -999;
    sym.write(s);

    for(int i = 0; i < 4; i++) {
        if(s[i] != s_ref_3[i]) {
            std::ostringstream ss;
            ss << "Result doesn't match reference (3): "
               << "s[" << i << "] = " << s[i] << "; "
               << "s_ref_3[" << i << "] = " << s_ref_3[i];
            fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
        }
    }

    sym.permute(permutation<4>().permute(0, 2).permute(1, 3));
    for(int i = 0; i < 4; i++) s[i] = -999;
    sym.write(s);

    for(int i = 0; i < 4; i++) {
        if(s[i] != s_ref_4[i]) {
            std::ostringstream ss;
            ss << "Result doesn't match reference (4): "
               << "s[" << i << "] = " << s[i] << "; "
               << "s_ref_4[" << i << "] = " << s_ref_4[i];
            fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
        }
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_symmetry_test::test_4() {

    static const char testname[] = "ctf_symmetry_test::test_4()";

    typedef std_allocator<double> allocator_t;

    try {

    sequence<4, unsigned> grp(0), symind(0);
    grp[0] = 0; grp[1] = 0; grp[2] = 1; grp[3] = 1;
    symind[0] = 1; symind[1] = 1;
    ctf_symmetry<4, double> sym(grp, symind);
    int s[4];
    int s_ref_1[4] = { AS, NS, AS, NS };
    int s_ref_2[4] = { NS, NS, NS, NS };
    int s_ref_3[4] = { AS, NS, AS, NS };

    for(int i = 0; i < 4; i++) s[i] = -999;
    sym.write(s);

    for(int i = 0; i < 4; i++) {
        if(s[i] != s_ref_1[i]) {
            std::ostringstream ss;
            ss << "Result doesn't match reference (1): "
               << "s[" << i << "] = " << s[i] << "; "
               << "s_ref_1[" << i << "] = " << s_ref_1[i];
            fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
        }
    }

    sym.permute(permutation<4>().permute(1, 2));
    for(int i = 0; i < 4; i++) s[i] = -999;
    sym.write(s);

    for(int i = 0; i < 4; i++) {
        if(s[i] != s_ref_2[i]) {
            std::ostringstream ss;
            ss << "Result doesn't match reference (2): "
               << "s[" << i << "] = " << s[i] << "; "
               << "s_ref_2[" << i << "] = " << s_ref_2[i];
            fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
        }
    }

    sym.permute(permutation<4>().permute(1, 2));
    for(int i = 0; i < 4; i++) s[i] = -999;
    sym.write(s);

    for(int i = 0; i < 4; i++) {
        if(s[i] != s_ref_3[i]) {
            std::ostringstream ss;
            ss << "Result doesn't match reference (3): "
               << "s[" << i << "] = " << s[i] << "; "
               << "s_ref_3[" << i << "] = " << s_ref_3[i];
            fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
        }
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_symmetry_test::test_5() {

    static const char testname[] = "ctf_symmetry_test::test_5()";

    try {

    sequence<3, unsigned> grp(0), symind(0);
    grp[0] = 0; grp[1] = 0; grp[2] = 0;
    symind[0] = 0;
    ctf_symmetry<3, double> sym(grp, symind);
    int s[3];
    int s_ref[3] = { SY, SY, NS };

    for(int i = 0; i < 3; i++) s[i] = -999;

    sym.write(s);

    for(int i = 0; i < 3; i++) {
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


} // namespace libtensor

