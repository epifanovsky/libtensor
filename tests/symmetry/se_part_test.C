#include <sstream>
#include <set>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/symmetry/se_part.h>
#include "se_part_test.h"

namespace libtensor {


void se_part_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
    test_3a();
    test_3b();
    test_4();
    test_5();
    test_6();
    test_perm_1();
    test_perm_2();
    test_perm_3();
    test_perm_4();
    test_perm_5();
    test_exc();
}


/** \test Two partitions, one block in each partition (2-dim)
 **/
void se_part_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "se_part_test::test_1()";

    try {

        index<2> i1, i2;
        i2[0] = 9; i2[1] = 9;
        block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
        mask<2> m11;
        m11[0] = true; m11[1] = true;
        bis.split(m11, 5);

        index<2> i00, i01, i10, i11;
        i01[0] = 0; i01[1] = 1;
        i10[0] = 1; i10[1] = 0;
        i11[0] = 1; i11[1] = 1;

        se_part<2, double> elem1(bis, m11, 2);
        elem1.add_map(i00, i11);

        if(!elem1.is_allowed(i00)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!elem1.is_allowed(i00)");
        }
        if(!elem1.is_allowed(i01)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!elem1.is_allowed(i01)");
        }
        if(!elem1.is_allowed(i10)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!elem1.is_allowed(i10)");
        }
        if(!elem1.is_allowed(i11)) {
            fail_test(testname, __FILE__, __LINE__,
                    "!elem1.is_allowed(i11)");
        }

        index<2> i00a(i00), i01a(i01), i10a(i10), i11a(i11);
        elem1.apply(i00a);
        if(!i00a.equals(i11)) {
            fail_test(testname, __FILE__, __LINE__, "!i00a.equals(i11)");
        }
        elem1.apply(i01a);
        if(!i01a.equals(i01)) {
            fail_test(testname, __FILE__, __LINE__, "!i01a.equals(i01)");
        }
        elem1.apply(i10a);
        if(!i10a.equals(i10)) {
            fail_test(testname, __FILE__, __LINE__, "!i10a.equals(i10)");
        }
        elem1.apply(i11a);
        if(!i11a.equals(i00)) {
            fail_test(testname, __FILE__, __LINE__, "!i11a.equals(i00)");
        }

        index<2> i00b(i00), i01b(i01), i10b(i10), i11b(i11);
        tensor_transf<2, double> tr00, tr01, tr10, tr11;
        elem1.apply(i00b, tr00);
        elem1.apply(i01b, tr01);
        elem1.apply(i10b, tr10);
        elem1.apply(i11b, tr11);
        if(!i00b.equals(i11)) {
            fail_test(testname, __FILE__, __LINE__, "!i00b.equals(i11)");
        }
        if(tr00.get_scalar_tr().get_coeff() != 1.0) {
            fail_test(testname, __FILE__, __LINE__,
                    "tr00.get_coeff() != 1.0");
        }
        if(!tr00.get_perm().is_identity()) {
            fail_test(testname, __FILE__, __LINE__,
                    "!tr00.get_perm().is_identity()");
        }
        if(!i01b.equals(i01)) {
            fail_test(testname, __FILE__, __LINE__, "!i01b.equals(i01)");
        }
        if(tr01.get_scalar_tr().get_coeff() != 1.0) {
            fail_test(testname, __FILE__, __LINE__,
                    "tr01.get_coeff() != 1.0");
        }
        if(!tr01.get_perm().is_identity()) {
            fail_test(testname, __FILE__, __LINE__,
                    "!tr01.get_perm().is_identity()");
        }
        if(!i10b.equals(i10)) {
            fail_test(testname, __FILE__, __LINE__, "!i10b.equals(i10)");
        }
        if(tr10.get_scalar_tr().get_coeff() != 1.0) {
            fail_test(testname, __FILE__, __LINE__,
                    "tr10.get_coeff() != 1.0");
        }
        if(!tr10.get_perm().is_identity()) {
            fail_test(testname, __FILE__, __LINE__,
                    "!tr10.get_perm().is_identity()");
        }
        if(!i11b.equals(i00)) {
            fail_test(testname, __FILE__, __LINE__, "!i11b.equals(i00)");
        }
        if(tr11.get_scalar_tr().get_coeff() != 1.0) {
            fail_test(testname, __FILE__, __LINE__,
                    "tr11.get_coeff() != 1.0");
        }
        if(!tr11.get_perm().is_identity()) {
            fail_test(testname, __FILE__, __LINE__,
                    "!tr11.get_perm().is_identity()");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Two partitions, two blocks in each partition (2-dim)
 **/
void se_part_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "se_part_test::test_2()";

    try {

        index<2> i1, i2;
        i2[0] = 9; i2[1] = 9;
        block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
        mask<2> m11;
        m11[0] = true; m11[1] = true;
        bis.split(m11, 2);
        bis.split(m11, 5);
        bis.split(m11, 7);

        index<2> i00, i01, i02, i03, i10, i11, i12, i13, i20, i21, i22, i23,
        i30, i31, i32, i33;
        i01[0] = 0; i01[1] = 1;
        i02[0] = 0; i02[1] = 2;
        i03[0] = 0; i03[1] = 3;
        i10[0] = 1; i10[1] = 0;
        i11[0] = 1; i11[1] = 1;
        i12[0] = 1; i12[1] = 2;
        i13[0] = 1; i13[1] = 3;
        i20[0] = 2; i20[1] = 0;
        i21[0] = 2; i21[1] = 1;
        i22[0] = 2; i22[1] = 2;
        i23[0] = 2; i23[1] = 3;
        i30[0] = 3; i30[1] = 0;
        i31[0] = 3; i31[1] = 1;
        i32[0] = 3; i32[1] = 2;
        i33[0] = 3; i33[1] = 3;

        se_part<2, double> elem1(bis, m11, 2);
        elem1.add_map(i00, i11);

        index<2> i00a(i00), i01a(i01), i02a(i02), i03a(i03),
                i10a(i10), i11a(i11), i12a(i12), i13a(i13),
                i20a(i20), i21a(i21), i22a(i22), i23a(i23),
                i30a(i30), i31a(i31), i32a(i32), i33a(i33);
        elem1.apply(i00a);
        if(!i00a.equals(i22)) {
            fail_test(testname, __FILE__, __LINE__, "!i00a.equals(i22)");
        }
        elem1.apply(i01a);
        if(!i01a.equals(i23)) {
            fail_test(testname, __FILE__, __LINE__, "!i01a.equals(i23)");
        }
        elem1.apply(i02a);
        if(!i02a.equals(i02)) {
            fail_test(testname, __FILE__, __LINE__, "!i02a.equals(i02)");
        }
        elem1.apply(i03a);
        if(!i03a.equals(i03)) {
            fail_test(testname, __FILE__, __LINE__, "!i03a.equals(i03)");
        }
        elem1.apply(i10a);
        if(!i10a.equals(i32)) {
            fail_test(testname, __FILE__, __LINE__, "!i10a.equals(i32)");
        }
        elem1.apply(i11a);
        if(!i11a.equals(i33)) {
            fail_test(testname, __FILE__, __LINE__, "!i11a.equals(i33)");
        }
        elem1.apply(i12a);
        if(!i12a.equals(i12)) {
            fail_test(testname, __FILE__, __LINE__, "!i12a.equals(i12)");
        }
        elem1.apply(i13a);
        if(!i13a.equals(i13)) {
            fail_test(testname, __FILE__, __LINE__, "!i13a.equals(i13)");
        }
        elem1.apply(i20a);
        if(!i20a.equals(i20)) {
            fail_test(testname, __FILE__, __LINE__, "!i20a.equals(i20)");
        }
        elem1.apply(i21a);
        if(!i21a.equals(i21)) {
            fail_test(testname, __FILE__, __LINE__, "!i21a.equals(i21)");
        }
        elem1.apply(i22a);
        if(!i22a.equals(i00)) {
            fail_test(testname, __FILE__, __LINE__, "!i22a.equals(i00)");
        }
        elem1.apply(i23a);
        if(!i23a.equals(i01)) {
            fail_test(testname, __FILE__, __LINE__, "!i23a.equals(i01)");
        }
        elem1.apply(i30a);
        if(!i30a.equals(i30)) {
            fail_test(testname, __FILE__, __LINE__, "!i30a.equals(i30)");
        }
        elem1.apply(i31a);
        if(!i31a.equals(i31)) {
            fail_test(testname, __FILE__, __LINE__, "!i31a.equals(i31)");
        }
        elem1.apply(i32a);
        if(!i32a.equals(i10)) {
            fail_test(testname, __FILE__, __LINE__, "!i32a.equals(i10)");
        }
        elem1.apply(i33a);
        if(!i33a.equals(i11)) {
            fail_test(testname, __FILE__, __LINE__, "!i33a.equals(i11)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Two partitions, two or three blocks in each partition (4-dim),
        block sizes vary for different dimensions
 **/
void se_part_test::test_3a() throw(libtest::test_exception) {

    static const char *testname = "se_part_test::test_3a()";

    try {

        index<4> i1, i2;
        i2[0] = 9; i2[1] = 9; i2[2] = 19; i2[3] = 19;
        block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
        mask<4> m1100, m0011, m1111;
        m1100[0] = true; m1100[1] = true;
        m0011[2] = true; m0011[3] = true;
        m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
        bis.split(m1100, 2);
        bis.split(m1100, 5);
        bis.split(m1100, 7);
        bis.split(m0011, 3);
        bis.split(m0011, 6);
        bis.split(m0011, 10);
        bis.split(m0011, 13);
        bis.split(m0011, 16);

        index<4> i0000, i0011, i0033, i0101, i0110, i0134, i1001, i1100, i1111,
        i2200, i2233, i2301, i2334;
        i0011[0] = 0; i0011[1] = 0; i0011[2] = 1; i0011[3] = 1;
        i0033[0] = 0; i0033[1] = 0; i0033[2] = 3; i0033[3] = 3;
        i0101[0] = 0; i0101[1] = 1; i0101[2] = 0; i0101[3] = 1;
        i0110[0] = 0; i0110[1] = 1; i0110[2] = 1; i0110[3] = 0;
        i0134[0] = 0; i0134[1] = 1; i0134[2] = 3; i0134[3] = 4;
        i1001[0] = 1; i1001[1] = 0; i1001[2] = 0; i1001[3] = 1;
        i1100[0] = 1; i1100[1] = 1; i1100[2] = 0; i1100[3] = 0;
        i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;
        i2200[0] = 2; i2200[1] = 2; i2200[2] = 0; i2200[3] = 0;
        i2233[0] = 2; i2233[1] = 2; i2233[2] = 3; i2233[3] = 3;
        i2301[0] = 2; i2301[1] = 3; i2301[2] = 0; i2301[3] = 1;
        i2334[0] = 2; i2334[1] = 3; i2334[2] = 3; i2334[3] = 4;

        se_part<4, double> elem1(bis, m1111, 2);
        elem1.add_map(i0000, i1111);
        elem1.add_map(i0000, i0011, scalar_transf<double>(2.0));
        elem1.add_map(i1100, i1111, scalar_transf<double>(0.5));
        elem1.add_map(i0110, i1001);

        std::set< index<4> > orbit;

        //  [0000]->[2200]->[0033]->[2233]
        orbit.insert(i0000);
        orbit.insert(i2200);
        orbit.insert(i0033);
        orbit.insert(i2233);
        index<4> i0000a(i0000);
        while(!orbit.empty()) {
            elem1.apply(i0000a);
            if(orbit.find(i0000a) == orbit.end()) {
                std::ostringstream ss;
                ss << "Invalid or duplicate index in the orbit of "
                        << i0000 << ": " << i0000a << ".";
                fail_test(testname, __FILE__, __LINE__,
                        ss.str().c_str());
            } else {
                orbit.erase(i0000a);
            }
        }

        //  [0101]->[0134]->[2301]->[2334]
        orbit.insert(i0101);
        orbit.insert(i0134);
        orbit.insert(i2301);
        orbit.insert(i2334);
        index<4> i0101a(i0101);
        while(!orbit.empty()) {
            elem1.apply(i0101a);
            if(orbit.find(i0101a) == orbit.end()) {
                std::ostringstream ss;
                ss << "Invalid or duplicate index in the orbit of "
                        << i0101 << ": " << i0101a << ".";
                fail_test(testname, __FILE__, __LINE__,
                        ss.str().c_str());
            } else {
                orbit.erase(i0101a);
            }
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Two partitions, two or three blocks in each partition (4-dim),
        block sizes vary for different dimensions.
 **/
void se_part_test::test_3b() throw(libtest::test_exception) {

    static const char *testname = "se_part_test::test_3b()";

    try {

        index<4> i1, i2;
        i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 19;
        block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
        mask<4> m1110, m0001, m1111;
        m1110[0] = true; m1110[1] = true; m1110[2] = true; m0001[3] = true;
        m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
        bis.split(m1110, 2);
        bis.split(m1110, 5);
        bis.split(m1110, 7);
        bis.split(m0001, 3);
        bis.split(m0001, 6);
        bis.split(m0001, 10);
        bis.split(m0001, 13);
        bis.split(m0001, 16);

        index<4> i0000, i0011, i0023, i0101, i0110, i0124, i1001, i1100, i1111,
        i2200, i2223, i2301, i2324;
        i0011[0] = 0; i0011[1] = 0; i0011[2] = 1; i0011[3] = 1;
        i0023[0] = 0; i0023[1] = 0; i0023[2] = 2; i0023[3] = 3;
        i0101[0] = 0; i0101[1] = 1; i0101[2] = 0; i0101[3] = 1;
        i0110[0] = 0; i0110[1] = 1; i0110[2] = 1; i0110[3] = 0;
        i0124[0] = 0; i0124[1] = 1; i0124[2] = 2; i0124[3] = 4;
        i1001[0] = 1; i1001[1] = 0; i1001[2] = 0; i1001[3] = 1;
        i1100[0] = 1; i1100[1] = 1; i1100[2] = 0; i1100[3] = 0;
        i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;
        i2200[0] = 2; i2200[1] = 2; i2200[2] = 0; i2200[3] = 0;
        i2223[0] = 2; i2223[1] = 2; i2223[2] = 2; i2223[3] = 3;
        i2301[0] = 2; i2301[1] = 3; i2301[2] = 0; i2301[3] = 1;
        i2324[0] = 2; i2324[1] = 3; i2324[2] = 2; i2324[3] = 4;

        se_part<4, double> elem1(bis, m1111, 2);
        elem1.add_map(i0000, i1111);
        elem1.add_map(i0000, i0011, scalar_transf<double>(-1.));
        elem1.add_map(i1100, i1111, scalar_transf<double>(-1.));
        elem1.add_map(i0110, i1001);

        std::set< index<4> > orbit;

        //  [0000]->[2200]->[0033]->[2233]
        orbit.insert(i0000);
        orbit.insert(i2200);
        orbit.insert(i0023);
        orbit.insert(i2223);
        index<4> i0000a(i0000);
        while(!orbit.empty()) {
            elem1.apply(i0000a);
            if(orbit.find(i0000a) == orbit.end()) {
                std::ostringstream ss;
                ss << "Invalid or duplicate index in the orbit of "
                        << i0000 << ": " << i0000a << ".";
                fail_test(testname, __FILE__, __LINE__,
                        ss.str().c_str());
            } else {
                orbit.erase(i0000a);
            }
        }

        //  [0101]->[0124]->[2301]->[2324]
        orbit.insert(i0101);
        orbit.insert(i0124);
        orbit.insert(i2301);
        orbit.insert(i2324);
        index<4> i0101a(i0101);
        while(!orbit.empty()) {
            elem1.apply(i0101a);
            if(orbit.find(i0101a) == orbit.end()) {
                std::ostringstream ss;
                ss << "Invalid or duplicate index in the orbit of "
                        << i0101 << ": " << i0101a << ".";
                fail_test(testname, __FILE__, __LINE__,
                        ss.str().c_str());
            } else {
                orbit.erase(i0101a);
            }
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Two partitions, two or three blocks in each partition (4-dim),
        block sizes vary for different dimensions.
 **/
void se_part_test::test_4() throw(libtest::test_exception) {

    static const char *testname = "se_part_test::test_4()";

    try {

        index<4> i1, i2;
        i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
        block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
        mask<4> m1111;
        m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
        bis.split(m1111, 2);
        bis.split(m1111, 5);
        bis.split(m1111, 7);

        index<4> i0000, i0011, i0101, i0110, i1001, i1100, i1111;
        i0011[0] = 0; i0011[1] = 0; i0011[2] = 1; i0011[3] = 1;
        i0101[0] = 0; i0101[1] = 1; i0101[2] = 0; i0101[3] = 1;
        i0110[0] = 0; i0110[1] = 1; i0110[2] = 1; i0110[3] = 0;
        i1001[0] = 1; i1001[1] = 0; i1001[2] = 0; i1001[3] = 1;
        i1100[0] = 1; i1100[1] = 1; i1100[2] = 0; i1100[3] = 0;
        i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;

        scalar_transf<double> tr0, tr(0.5);
        se_part<4, double> elem(bis, m1111, 2);
        elem.add_map(i0000, i0011, tr);
        elem.add_map(i1111, i1100, tr);
        elem.add_map(i0011, i1100);

        if (! elem.map_exists(i0000, i1100)) {
            fail_test(testname, __FILE__, __LINE__, "Missing map: 0000->1100.");
        }
        if (elem.get_transf(i0000, i1100) != tr) {
            fail_test(testname, __FILE__, __LINE__, "Wrong transf: 0000->1100.");
        }
        if (! elem.map_exists(i0000, i1111)) {
            fail_test(testname, __FILE__, __LINE__, "Missing map: 0000->1111.");
        }
        if (elem.get_transf(i0000, i1111) != tr0) {
            fail_test(testname, __FILE__, __LINE__, "Wrong transf: 0000->1100.");
        }
        if (! elem.map_exists(i0011, i1111)) {
            fail_test(testname, __FILE__, __LINE__, "Missing map: 0011->1111.");
        }
        if (elem.get_transf(i1111, i0011) != tr) {
            fail_test(testname, __FILE__, __LINE__, "Wrong transf: 1111->0011.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Two partitions, two blocks in each partition (2-dim), forbidden
 **/
void se_part_test::test_5() throw(libtest::test_exception) {

    static const char *testname = "se_part_test::test_5()";

    try {

        index<2> i1, i2;
        i2[0] = 9; i2[1] = 9;
        block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
        mask<2> m11;
        m11[0] = true; m11[1] = true;
        bis.split(m11, 2);
        bis.split(m11, 5);
        bis.split(m11, 7);

        index<2> i00, i01, i02, i03, i10, i11, i12, i13, i20, i21, i22, i23,
        i30, i31, i32, i33;
        i01[0] = 0; i01[1] = 1;
        i02[0] = 0; i02[1] = 2;
        i03[0] = 0; i03[1] = 3;
        i10[0] = 1; i10[1] = 0;
        i11[0] = 1; i11[1] = 1;
        i12[0] = 1; i12[1] = 2;
        i13[0] = 1; i13[1] = 3;
        i20[0] = 2; i20[1] = 0;
        i21[0] = 2; i21[1] = 1;
        i22[0] = 2; i22[1] = 2;
        i23[0] = 2; i23[1] = 3;
        i30[0] = 3; i30[1] = 0;
        i31[0] = 3; i31[1] = 1;
        i32[0] = 3; i32[1] = 2;
        i33[0] = 3; i33[1] = 3;

        se_part<2, double> elem1(bis, m11, 2);
        elem1.add_map(i00, i11);
        elem1.mark_forbidden(i01);
        elem1.mark_forbidden(i10);

        index<2> i00a(i00), i01a(i01), i02a(i02), i03a(i03),
                i10a(i10), i11a(i11), i12a(i12), i13a(i13),
                i20a(i20), i21a(i21), i22a(i22), i23a(i23),
                i30a(i30), i31a(i31), i32a(i32), i33a(i33);
        elem1.apply(i00a);
        if(!i00a.equals(i22)) {
            fail_test(testname, __FILE__, __LINE__, "!i00a.equals(i22)");
        }
        elem1.apply(i01a);
        if(!i01a.equals(i23)) {
            fail_test(testname, __FILE__, __LINE__, "!i01a.equals(i23)");
        }
        if(elem1.is_allowed(i02)) {
            fail_test(testname, __FILE__, __LINE__, "i02 is allowed.");
        }
        if(elem1.is_allowed(i03)) {
            fail_test(testname, __FILE__, __LINE__, "i03 is allowed.");
        }
        elem1.apply(i10a);
        if(!i10a.equals(i32)) {
            fail_test(testname, __FILE__, __LINE__, "!i10a.equals(i32)");
        }
        elem1.apply(i11a);
        if(!i11a.equals(i33)) {
            fail_test(testname, __FILE__, __LINE__, "!i11a.equals(i33)");
        }
        if(elem1.is_allowed(i12)) {
            fail_test(testname, __FILE__, __LINE__, "i12 is allowed.");
        }
        if(elem1.is_allowed(i13)) {
            fail_test(testname, __FILE__, __LINE__, "i13 is allowed.");
        }
        if(elem1.is_allowed(i20)) {
            fail_test(testname, __FILE__, __LINE__, "i20 is allowed.");
        }
        if(elem1.is_allowed(i21)) {
            fail_test(testname, __FILE__, __LINE__, "i21 is allowed.");
        }
        elem1.apply(i22a);
        if(!i22a.equals(i00)) {
            fail_test(testname, __FILE__, __LINE__, "!i22a.equals(i00)");
        }
        elem1.apply(i23a);
        if(!i23a.equals(i01)) {
            fail_test(testname, __FILE__, __LINE__, "!i23a.equals(i01)");
        }
        if(elem1.is_allowed(i30)) {
            fail_test(testname, __FILE__, __LINE__, "i30 is allowed.");
        }
        if(elem1.is_allowed(i31)) {
            fail_test(testname, __FILE__, __LINE__, "i31 is allowed.");
        }
        elem1.apply(i32a);
        if(!i32a.equals(i10)) {
            fail_test(testname, __FILE__, __LINE__, "!i32a.equals(i10)");
        }
        elem1.apply(i33a);
        if(!i33a.equals(i11)) {
            fail_test(testname, __FILE__, __LINE__, "!i33a.equals(i11)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Test creation of maps between forbidden partitions and
        forbidden partitions in existing maps
 **/
void se_part_test::test_6() throw(libtest::test_exception) {

    static const char *testname = "se_part_test::test_6()";

    try {

        index<2> i1, i2;
        i2[0] = 7; i2[1] = 9;
        block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
        mask<2> m01, m10, m11;
        m10[0] = true; m01[1] = true;
        m11[0] = true; m11[1] = true;
        bis.split(m10, 2);
        bis.split(m10, 4);
        bis.split(m10, 6);
        bis.split(m01, 2);
        bis.split(m01, 5);
        bis.split(m01, 7);

        index<2> i00, i01, i10, i11;
        i01[0] = 0; i01[1] = 1;
        i10[0] = 1; i10[1] = 0;
        i11[0] = 1; i11[1] = 1;
        scalar_transf<double> tr0, tr1(-1.);

        se_part<2, double> elem1(bis, m11, 2), elem2(bis, m11, 2);
        elem1.add_map(i00, i11, tr0);
        elem1.add_map(i01, i10, tr1);
        elem1.mark_forbidden(i01);

        if (elem1.map_exists(i00, i01)) {
            fail_test(testname, __FILE__, __LINE__, "Map (i00->i01) exists.");
        }
        if (elem1.map_exists(i00, i10)) {
            fail_test(testname, __FILE__, __LINE__, "Map (i00->i10) exists.");
        }
        if (! elem1.map_exists(i00, i11)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Map (i00->i11) does not exist.");
        }
        if (elem1.get_transf(i00, i11) != tr0) {
            fail_test(testname, __FILE__, __LINE__, "Map (i00->i10) !=  tr0.");
        }
        if (! elem1.is_forbidden(i01)) {
            fail_test(testname, __FILE__, __LINE__, "i01 is not forbidden.");
        }
        if (elem1.map_exists(i01, i10)) {
            fail_test(testname, __FILE__, __LINE__, "Map (i01->i10) exists.");
        }
        if (elem1.map_exists(i01, i11)) {
            fail_test(testname, __FILE__, __LINE__, "Map (i01->i11) exists.");
        }
        if (! elem1.is_forbidden(i10)) {
            fail_test(testname, __FILE__, __LINE__, "i10 is not forbidden.");
        }
        if (elem1.map_exists(i10, i11)) {
            fail_test(testname, __FILE__, __LINE__, "Map (i10->i11) exists.");
        }


        elem1.add_map(i00, i01, tr1);

        if (! elem1.map_exists(i00, i01)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Map (i00->i01) does not exist.");
        }
        if (elem1.get_transf(i00, i01) != tr1) {
            fail_test(testname, __FILE__, __LINE__, "Map (i00->i01) != tr1.");
        }
        if (elem1.map_exists(i00, i10)) {
            fail_test(testname, __FILE__, __LINE__, "Map (i00->i10) exists.");
        }
        if (! elem1.map_exists(i00, i11)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Map (i00->i11) does not exist.");
        }
        if (elem1.get_transf(i00, i11) != tr0) {
            fail_test(testname, __FILE__, __LINE__, "Map (i00->i11) != tr0.");
        }
        if (elem1.is_forbidden(i01)) {
            fail_test(testname, __FILE__, __LINE__, "i01 is forbidden.");
        }
        if (elem1.map_exists(i01, i10)) {
            fail_test(testname, __FILE__, __LINE__, "Map (i01->i10) exists.");
        }
        if (! elem1.map_exists(i01, i11)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Map (i01->i11) does not exist.");
        }
        if (elem1.get_transf(i01, i11) != tr1) {
            fail_test(testname, __FILE__, __LINE__, "Map (i00->i11) != tr1.");
        }
        if (! elem1.is_forbidden(i10)) {
            fail_test(testname, __FILE__, __LINE__, "i10 is not forbidden.");
        }
        if (elem1.map_exists(i10, i11)) {
            fail_test(testname, __FILE__, __LINE__, "Map (i10->i11) exists.");
        }

        elem1.add_map(i10, i11, tr1);

        if (! elem1.map_exists(i00, i01)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Map (i00->i01) does not exist.");
        }
        if (elem1.get_transf(i00, i01) != tr1) {
            fail_test(testname, __FILE__, __LINE__, "Map (i00->i01) != tr1.");
        }
        if (! elem1.map_exists(i00, i10)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Map (i00->i10) does not exist.");
        }
        if (elem1.get_transf(i00, i10) != tr1) {
            fail_test(testname, __FILE__, __LINE__, "Map (i00->i01) != tr1.");
        }
        if (! elem1.map_exists(i00, i11)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Map (i00->i11) does not exist.");
        }
        if (elem1.get_transf(i00, i11) != tr0) {
            fail_test(testname, __FILE__, __LINE__, "Map (i00->i11) != tr0.");
        }
        if (elem1.is_forbidden(i01)) {
            fail_test(testname, __FILE__, __LINE__, "i01 is forbidden.");
        }
        if (! elem1.map_exists(i01, i10)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Map (i01->i10) does not exist.");
        }
        if (elem1.get_transf(i01, i10) != tr0) {
            fail_test(testname, __FILE__, __LINE__, "Map (i01->i10) != tr0.");
        }
        if (! elem1.map_exists(i01, i11)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Map (i01->i11) does not exist.");
        }
        if (elem1.get_transf(i01, i11) != tr1) {
            fail_test(testname, __FILE__, __LINE__, "Map (i00->i11) != tr1.");
        }
        if (elem1.is_forbidden(i10)) {
            fail_test(testname, __FILE__, __LINE__, "i10 is forbidden.");
        }
        if (! elem1.map_exists(i10, i11)) {
            fail_test(testname, __FILE__, __LINE__,
                    "Map (i10->i11) does not exist.");
        }
        if (elem1.get_transf(i10, i11) != tr1) {
            fail_test(testname, __FILE__, __LINE__, "Map (i10->i11) != tr1.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Permutation of se_part: two partitions, two or three blocks in
        each partition (4-dim), block sizes vary for different dimensions
 **/
void se_part_test::test_perm_1() throw(libtest::test_exception) {

    static const char *testname = "se_part_test::test_perm_1()";

    try {

        index<4> i1, i2;
        i2[0] = 9; i2[1] = 9; i2[2] = 19; i2[3] = 19;
        block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
        mask<4> m1100, m0011, m1111;
        m1100[0] = true; m1100[1] = true;
        m0011[2] = true; m0011[3] = true;
        m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
        bis.split(m1100, 2);
        bis.split(m1100, 5);
        bis.split(m1100, 7);
        bis.split(m0011, 3);
        bis.split(m0011, 6);
        bis.split(m0011, 10);
        bis.split(m0011, 13);
        bis.split(m0011, 16);

        index<4> i0000, i0011, i0001, i0010;
        i0011[2] = 1; i0011[3] = 1; i0010[2] = 1; i0001[3] = 1;
        scalar_transf<double> tr0;

        se_part<4, double> elem(bis, m0011, 2);
        elem.add_map(i0000, i0011, tr0);
        elem.add_map(i0001, i0010, tr0);

        permutation<4> perm; perm.permute(0, 1); // leaves mask unaffected

        bis.permute(perm);
        index<4> i1b, i2b;
        i2b[2] = 1; i2b[3] = 1;
        dimensions<4> pdims(index_range<4>(i1b, i2b));

        elem.permute(perm);

        if (! bis.equals(elem.get_bis())) {
            fail_test(testname, __FILE__, __LINE__, "Wrong bis.");
        }
        if (! pdims.equals(elem.get_pdims())) {
            fail_test(testname, __FILE__, __LINE__, "Wrong pdims.");
        }
        if (! elem.map_exists(i0000, i0011)) {
            fail_test(testname, __FILE__, __LINE__, "Missing map: 0000->0011.");
        }
        if (! elem.map_exists(i0001, i0010)) {
            fail_test(testname, __FILE__, __LINE__, "Missing map: 0001->0010.");
        }


    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Permutation of se_part: two partitions, two or three blocks in
        each partition (4-dim), block sizes vary for different dimensions
 **/
void se_part_test::test_perm_2() throw(libtest::test_exception) {

    static const char *testname = "se_part_test::test_perm_2()";

    try {

        index<4> i1, i2;
        i2[0] = 9; i2[1] = 9; i2[2] = 19; i2[3] = 19;
        block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
        mask<4> m1100, m0011, m1111;
        m1100[0] = true; m1100[1] = true;
        m0011[2] = true; m0011[3] = true;
        m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
        bis.split(m1100, 2);
        bis.split(m1100, 5);
        bis.split(m1100, 7);
        bis.split(m0011, 3);
        bis.split(m0011, 6);
        bis.split(m0011, 10);
        bis.split(m0011, 13);
        bis.split(m0011, 16);

        index<4> i0000, i0011, i0001, i0010, i0101, i0100;
        i0011[2] = 1; i0011[3] = 1; i0010[2] = 1; i0001[3] = 1;
        i0100[1] = 1; i0101[1] = 1; i0101[3] = 1;

        se_part<4, double> elem(bis, m0011, 2);
        scalar_transf<double> tr0;
        elem.add_map(i0000, i0011, tr0);
        elem.add_map(i0001, i0010, tr0);

        permutation<4> perm; perm.permute(0, 1).permute(1, 2);

        bis.permute(perm);
        index<4> i1b, i2b;
        i2b[1] = 1; i2b[3] = 1;
        dimensions<4> pdims(index_range<4>(i1b, i2b));
        mask<4> msk;
        msk[1] = true; msk[3] = true;

        elem.permute(perm);

        if (! bis.equals(elem.get_bis())) {
            fail_test(testname, __FILE__, __LINE__, "Wrong bis.");
        }
        if (! pdims.equals(elem.get_pdims())) {
            fail_test(testname, __FILE__, __LINE__, "Wrong pdims.");
        }
        if (! elem.map_exists(i0000, i0101)) {
            fail_test(testname, __FILE__, __LINE__, "Missing map: 0000->0011.");
        }
        if (! elem.map_exists(i0001, i0100)) {
            fail_test(testname, __FILE__, __LINE__, "Missing map: 0001->0010.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Permutation of se_part: two partitions, two or three blocks in
        each partition (4-dim), block sizes vary for different dimensions
 **/
void se_part_test::test_perm_3() throw(libtest::test_exception) {

    static const char *testname = "se_part_test::test_perm_3()";

    try {

        index<4> i1, i2;
        i2[0] = 9; i2[1] = 9; i2[2] = 19; i2[3] = 19;
        block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
        mask<4> m1100, m0011, m1111;
        m1100[0] = true; m1100[1] = true;
        m0011[2] = true; m0011[3] = true;
        m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
        bis.split(m1100, 2);
        bis.split(m1100, 5);
        bis.split(m1100, 7);
        bis.split(m0011, 3);
        bis.split(m0011, 6);
        bis.split(m0011, 10);
        bis.split(m0011, 13);
        bis.split(m0011, 16);

        index<4> i0000, i0011, i0001, i0010;
        i0011[2] = 1; i0011[3] = 1; i0010[2] = 1; i0001[3] = 1;

        scalar_transf<double> tr0;
        se_part<4, double> elem(bis, m0011, 2);
        elem.add_map(i0000, i0011, tr0);
        elem.add_map(i0001, i0010, tr0);


        permutation<4> perm; perm.permute(2, 3);

        bis.permute(perm);
        index<4> i1b, i2b;
        i2b[2] = 1; i2b[3] = 1;
        dimensions<4> pdims(index_range<4>(i1b, i2b));

        elem.permute(perm);

        if (! bis.equals(elem.get_bis())) {
            fail_test(testname, __FILE__, __LINE__, "Wrong bis.");
        }
        if (! pdims.equals(elem.get_pdims())) {
            fail_test(testname, __FILE__, __LINE__, "Wrong pdims.");
        }
        if (! elem.map_exists(i0000, i0011)) {
            fail_test(testname, __FILE__, __LINE__, "Missing map: 0000->0011.");
        }
        if (! elem.map_exists(i0010, i0001)) {
            fail_test(testname, __FILE__, __LINE__, "Missing map: 0010->0001.");
        }


    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Permutation of se_part: two partitions, two or three blocks in
        each partition (4-dim), block sizes vary for different dimensions
 **/
void se_part_test::test_perm_4() throw(libtest::test_exception) {

    static const char *testname = "se_part_test::test_perm_4()";

    try {

        index<4> i1, i2;
        i2[0] = 9; i2[1] = 9; i2[2] = 19; i2[3] = 19;
        block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
        mask<4> m1100, m0011, m1111;
        m1100[0] = true; m1100[1] = true;
        m0011[2] = true; m0011[3] = true;
        m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
        bis.split(m1100, 2);
        bis.split(m1100, 5);
        bis.split(m1100, 7);
        bis.split(m0011, 3);
        bis.split(m0011, 6);
        bis.split(m0011, 10);
        bis.split(m0011, 13);
        bis.split(m0011, 16);

        index<4> i0000, i0110, i1001, i0101, i1010, i1111;
        i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
        i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
        i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;
        scalar_transf<double> tr0;

        se_part<4, double> elem(bis, m1111, 2);
        elem.add_map(i0000, i1001, tr0);
        elem.add_map(i1001, i0110, tr0);
        elem.add_map(i0110, i1111, tr0);


        permutation<4> perm; perm.permute(0, 1);

        bis.permute(perm);
        index<4> i1b, i2b;
        i2b[0] = 1; i2b[1] = 1; i2b[2] = 1; i2b[3] = 1;
        dimensions<4> pdims(index_range<4>(i1b, i2b));

        elem.permute(perm);

        if (! bis.equals(elem.get_bis())) {
            fail_test(testname, __FILE__, __LINE__, "Wrong bis.");
        }
        if (! pdims.equals(elem.get_pdims())) {
            fail_test(testname, __FILE__, __LINE__, "Wrong pdims.");
        }
        if (! elem.map_exists(i0000, i1111)) {
            fail_test(testname, __FILE__, __LINE__, "Missing map: 0000->1111.");
        }
        if (! elem.map_exists(i0101, i1010)) {
            fail_test(testname, __FILE__, __LINE__, "Missing map: 0101->1010.");
        }


    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Permutation of se_part: two partitions, two blocks in
        each partition (2-dim), block sizes vary for different dimensions,
        forbidden partitions
 **/
void se_part_test::test_perm_5() throw(libtest::test_exception) {

    static const char *testname = "se_part_test::test_perm_5()";

    try {

        index<2> i1, i2;
        i2[0] = 9; i2[1] = 19;
        block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
        mask<2> m01, m10, m11;
        m10[0] = true; m01[1] = true;
        m11[0] = true; m11[1] = true;
        bis.split(m10, 2);
        bis.split(m10, 5);
        bis.split(m10, 7);
        bis.split(m01, 3);
        bis.split(m01, 6);
        bis.split(m01, 10);
        bis.split(m01, 13);
        bis.split(m01, 16);

        index<2> i00, i01, i10, i11;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;
        scalar_transf<double> tr0;

        se_part<2, double> elem(bis, m11, 2);
        elem.add_map(i01, i11, tr0);
        elem.mark_forbidden(i00);
        elem.mark_forbidden(i10);

        permutation<2> perm; perm.permute(0, 1);
        bis.permute(perm);

        dimensions<2> pdims(index_range<2>(i00, i11));
        mask<2> msk;
        msk[0] = true; msk[1] = true;

        elem.permute(perm);

        if (! bis.equals(elem.get_bis())) {
            fail_test(testname, __FILE__, __LINE__, "Wrong bis.");
        }
        if (! pdims.equals(elem.get_pdims())) {
            fail_test(testname, __FILE__, __LINE__, "Wrong pdims.");
        }
        if (! elem.map_exists(i10, i11)) {
            fail_test(testname, __FILE__, __LINE__, "Missing map: 10->11.");
        }
        if (! elem.is_forbidden(i00)) {
            fail_test(testname, __FILE__, __LINE__, "i00 is allowed.");
        }
        if (! elem.is_forbidden(i01)) {
            fail_test(testname, __FILE__, __LINE__, "i01 is allowed.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Two partitions, blocks with unequal size in each partition (2-dim)
 **/
void se_part_test::test_exc() throw(libtest::test_exception) {

    static const char *testname = "se_part_test::test_exc()";

    index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    block_index_space<2> bisa(dimensions<2>(index_range<2>(i1, i2)));
    block_index_space<2> bisb(dimensions<2>(index_range<2>(i1, i2)));
    block_index_space<2> bisc(dimensions<2>(index_range<2>(i1, i2)));
    mask<2> m01, m10, m11;
    m10[0] = true; m01[1] = true;
    m11[0] = true; m11[1] = true;
    bisa.split(m11, 3);
    bisa.split(m11, 5);
    bisa.split(m11, 7);

    bisb.split(m01, 5);
    bisb.split(m10, 2);
    bisb.split(m10, 4);
    bisb.split(m10, 5);
    bisb.split(m10, 7);

    bisb.split(m01, 2);
    bisb.split(m01, 4);
    bisb.split(m01, 5);
    bisb.split(m01, 6);
    bisb.split(m01, 7);
    bisb.split(m10, 5);

    bool failed = false;
    try {
        se_part<2, double> elem1(bisa, m11, 2);
    } catch(exception &e) {
        failed = true;
    }

    if (! failed)
        fail_test(testname, __FILE__, __LINE__,
                "Illegal se_part created without exception (elem1.");

    failed = false;
    try {
        se_part<2, double> elem2(bisb, m11, 2);
    } catch(exception &e) {
        failed = true;
    }

    if (! failed)
        fail_test(testname, __FILE__, __LINE__,
                "Illegal se_part created without exception (elem2.");

    failed = false;
    try {
        se_part<2, double> elem3(bisc, m11, 2);
    } catch(exception &e) {
        failed = true;
    }

    if (! failed)
        fail_test(testname, __FILE__, __LINE__,
                "Illegal se_part created without exception (elem3.");
}

} // namespace libtensor
