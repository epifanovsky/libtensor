#include <ctf.hpp>
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
        test_convfac_1();
        test_convfac_2();
        test_convfac_3();
        test_convfac_4();
        test_convfac_5();
        test_convfac_6();
        test_convfac_7();

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


void ctf_symmetry_test::test_convfac_1() {

    static const char testname[] = "ctf_symmetry_test::test_convfac_1()";

    try {

    sequence<4, unsigned> grp1(0), tag1(0), grp2(0), tag2(0);
    grp1[0] = 0; grp1[1] = 0; grp1[2] = 1; grp1[3] = 1;
    tag1[0] = 0; tag1[1] = 0;
    grp2[0] = 0; grp2[1] = 0; grp2[2] = 1; grp2[3] = 1;
    tag2[0] = 1; tag2[1] = 0;
    ctf_symmetry<4, double> sym1(grp1, tag1), sym2(grp2, tag2);

    double z = ctf_symmetry<4, double>::symconv_factor(sym1, 0, sym2, 0);
    double z_ref = 0.0;

    if(z != z_ref) {
        std::ostringstream ss;
        ss << "Result doesn't match reference: "
           << "z = " << z << "; " << "z_ref = " << z_ref;
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_symmetry_test::test_convfac_2() {

    static const char testname[] = "ctf_symmetry_test::test_convfac_2()";

    try {

    sequence<4, unsigned> grp1(0), tag1(0), grp2(0), tag2(0);
    grp1[0] = 0; grp1[1] = 0; grp1[2] = 1; grp1[3] = 2;
    tag1[0] = 0; tag1[1] = 0; tag1[2] = 0;
    grp2[0] = 0; grp2[1] = 0; grp2[2] = 1; grp2[3] = 1;
    tag2[0] = 0; tag2[1] = 0;
    ctf_symmetry<4, double> sym1(grp1, tag1), sym2(grp2, tag2);

    double z = ctf_symmetry<4, double>::symconv_factor(sym1, 0, sym2, 0);
    double z_ref = 0.5;

    if(z != z_ref) {
        std::ostringstream ss;
        ss << "Result doesn't match reference: "
           << "z = " << z << "; " << "z_ref = " << z_ref;
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_symmetry_test::test_convfac_3() {

    static const char testname[] = "ctf_symmetry_test::test_convfac_3()";

    try {

    sequence<4, unsigned> grp1(0), tag1(0), grp2(0), tag2(0);
    grp1[0] = 0; grp1[1] = 0; grp1[2] = 1; grp1[3] = 1;
    tag1[0] = 0; tag1[1] = 0;
    grp2[0] = 0; grp2[1] = 0; grp2[2] = 1; grp2[3] = 2;
    tag2[0] = 0; tag2[1] = 0; tag2[2] = 0;
    ctf_symmetry<4, double> sym1(grp1, tag1), sym2(grp2, tag2);

    double z = ctf_symmetry<4, double>::symconv_factor(sym1, 0, sym2, 0);
    double z_ref = 1.0;

    if(z != z_ref) {
        std::ostringstream ss;
        ss << "Result doesn't match reference: "
           << "z = " << z << "; " << "z_ref = " << z_ref;
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_symmetry_test::test_convfac_4() {

    static const char testname[] = "ctf_symmetry_test::test_convfac_4()";

    try {

    sequence<4, unsigned> grp(0), tag(0);
    grp[0] = 0; grp[1] = 1; grp[2] = 2; grp[3] = 3;
    ctf_symmetry<4, double> sym(grp, tag);

    double z = ctf_symmetry<4, double>::symconv_factor(sym, 0,
        permutation<4>().permute(0, 1));
    double z_ref = 1.0;

    if(z != z_ref) {
        std::ostringstream ss;
        ss << "Result doesn't match reference: "
           << "z = " << z << "; " << "z_ref = " << z_ref;
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_symmetry_test::test_convfac_5() {

    static const char testname[] = "ctf_symmetry_test::test_convfac_5()";

    try {

    sequence<4, unsigned> grp(0), tag(0);
    grp[0] = 0; grp[1] = 0; grp[2] = 2; grp[3] = 3;
    ctf_symmetry<4, double> sym(grp, tag);

    double z = ctf_symmetry<4, double>::symconv_factor(sym, 0,
        permutation<4>().permute(0, 1));
    double z_ref = 1.0;

    if(z != z_ref) {
        std::ostringstream ss;
        ss << "Result doesn't match reference: "
           << "z = " << z << "; " << "z_ref = " << z_ref;
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_symmetry_test::test_convfac_6() {

    static const char testname[] = "ctf_symmetry_test::test_convfac_6()";

    try {

    sequence<4, unsigned> grp(0), tag(0);
    grp[0] = 0; grp[1] = 1; grp[2] = 0; grp[3] = 3;
    ctf_symmetry<4, double> sym(grp, tag);

    double z = ctf_symmetry<4, double>::symconv_factor(sym, 0,
        permutation<4>().permute(0, 1));
    double z_ref = 0.5;

    if(z != z_ref) {
        std::ostringstream ss;
        ss << "Result doesn't match reference: "
           << "z = " << z << "; " << "z_ref = " << z_ref;
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_symmetry_test::test_convfac_7() {

    static const char testname[] = "ctf_symmetry_test::test_convfac_7()";

    try {

    sequence<2, unsigned> grp(0), tag(0);
    ctf_symmetry<2, double> sym(grp, tag);
    tag[0] = 1;
    sym.add_component(grp, tag);


    double z0 = ctf_symmetry<2, double>::symconv_factor(sym, 0,
        permutation<2>().permute(0, 1));
    double z1 = ctf_symmetry<2, double>::symconv_factor(sym, 1,
        permutation<2>().permute(0, 1));
    double z0_ref = 1.0, z1_ref = 1.0;

    if(z0 != z0_ref) {
        std::ostringstream ss;
        ss << "Result doesn't match reference: "
           << "z0 = " << z0 << "; " << "z0_ref = " << z0_ref;
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(z1 != z1_ref) {
        std::ostringstream ss;
        ss << "Result doesn't match reference: "
           << "z1 = " << z1 << "; " << "z1_ref = " << z1_ref;
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

