#include <sstream>
#include <string>
#include <libtensor/core/mask.h>
#include <libtensor/core/permutation.h>
#include "permutation_test.h"

namespace libtensor {

void permutation_test::perform() throw(libtest::test_exception) {

	test_ctor();
	test_permute();
	test_apply_mask_1();
	test_apply_mask_2();
	test_apply_mask_3();
	test_apply();
	test_print();
}

void permutation_test::test_ctor() throw(libtest::test_exception) {
	char sc[8];
	int si[8];

	// Order = 2

	permutation<2> p2;
	if(!p2.is_identity()) {
		fail_test("permutation_test::test_ctor()", __FILE__,
			__LINE__, "!p2.is_identity()");
	}
	strcpy(sc, "ij");
	p2.apply(2, sc);
	if(strncmp(sc, "ij", 2)) {
		fail_test("permutation_test::test_ctor()", __FILE__,
			__LINE__, "New permutation is not identity (char, 2)");
	}
	for(int i=0; i<8; i++) si[i]=i;
	p2.apply(2, si);
	if(si[0]!=0 || si[1]!=1) {
		fail_test("permutation_test::test_ctor()", __FILE__,
			__LINE__, "New permutation is not identity (int, 2)");
	}

	permutation<2> p2a(p2);
	if(!p2a.is_identity()) fail_test("permutation_test::test_ctor()",
		__FILE__, __LINE__, "!p2a.is_identity()");
	if(!p2a.equals(p2)) fail_test("permutation_test::test_ctor()",
		__FILE__, __LINE__, "!p2a.equals(p2)");
	strcpy(sc, "ij");
	p2a.apply(2, sc);
	if(strncmp(sc, "ij", 2))
		fail_test("permutation_test::test_ctor()", __FILE__,
		__LINE__, "Permutation copy is not identity (char, 2)");
	for(int i=0; i<8; i++) si[i]=i;
	p2a.apply(2, si);
	if(si[0]!=0 || si[1]!=1)
		fail_test("permutation_test::test_ctor()", __FILE__,
		__LINE__, "Permutation copy is not identity (int, 2)");

	// Order = 3

	permutation<3> p3;
	if(!p3.is_identity()) {
		fail_test("permutation_test::test_ctor()", __FILE__,
			__LINE__, "!p3.is_identity()");
	}
	strcpy(sc, "ijk");
	p3.apply(3, sc);
	if(strncmp(sc, "ijk", 3)) {
		fail_test("permutation_test::test_ctor()", __FILE__,
			__LINE__, "New permutation is not identity (char, 3)");
	}
	for(int i=0; i<8; i++) si[i]=i;
	p3.apply(3, si);
	if(si[0]!=0 || si[1]!=1 || si[2]!=2) {
		fail_test("permutation_test::test_ctor()", __FILE__,
			__LINE__, "New permutation is not identity (int, 3)");
	}

	permutation<3> p3a(p3);
	if(!p3a.is_identity()) {
		fail_test("permutation_test::test_ctor()", __FILE__,
			__LINE__, "!p3a.is_identity()");
	}
	if(!p3a.equals(p3)) {
		fail_test("permutation_test::test_ctor()", __FILE__,
			__LINE__, "!p3a.equals(p3)");
	}
	strcpy(sc, "ijk");
	p3a.apply(3, sc);
	if(strncmp(sc, "ijk", 3)) {
		fail_test("permutation_test::test_ctor()", __FILE__,
			__LINE__, "Permutation copy is not identity (char, 3)");
	}
	for(int i=0; i<8; i++) si[i]=i;
	p3a.apply(3, si);
	if(si[0]!=0 || si[1]!=1 || si[2]!=2) {
		fail_test("permutation_test::test_ctor()", __FILE__,
			__LINE__, "Permutation copy is not identity (int, 3)");
	}

	// Order = 4

	permutation<4> p4;
	if(!p4.is_identity()) fail_test("permutation_test::test_ctor()",
		__FILE__, __LINE__, "!p4.is_identity()");

	permutation<4> p4a;
	if(!p4a.is_identity()) fail_test("permutation_test::test_ctor()",
		__FILE__, __LINE__, "!p4a.is_identity()");
	if(!p4a.equals(p4)) fail_test("permutation_test::test_ctor()",
		__FILE__, __LINE__, "!p4a.equals(p4)");

	permutation<5> p5;
	if(!p5.is_identity()) fail_test("permutation_test::test_ctor()",
		__FILE__, __LINE__, "!p5.is_identity()");

	permutation<5> p5a;
	if(!p5a.is_identity()) fail_test("permutation_test::test_ctor()",
		__FILE__, __LINE__, "!p5a.is_identity()");
	if(!p5a.equals(p5)) fail_test("permutation_test::test_ctor()",
		__FILE__, __LINE__, "!p5a.equals(p5)");

	permutation<6> p6;
	if(!p6.is_identity()) fail_test("permutation_test::test_ctor()",
		__FILE__, __LINE__, "!p6.is_identity()");
	permutation<6> p6a(p6);
	if(!p6a.is_identity()) fail_test("permutation_test::test_ctor()",
		__FILE__, __LINE__, "!p6a.is_identity()");
	if(!p6a.equals(p6)) fail_test("permutation_test::test_ctor()",
		__FILE__, __LINE__, "!p6a.equals(p6)");
}

void permutation_test::test_permute() throw(libtest::test_exception) {
	permutation<2> p2;

	char s2[8]; strcpy(s2, "ij");
	int i2[8]; i2[0] = 100; i2[1] = 200;

	p2.permute(0,1);
	p2.apply(2, s2);
	if(strncmp(s2, "ji", 2)) {
		fail_test("permutation_test::test_permute()",
			__FILE__, __LINE__, "[0,1] permutation failed in char");
	}
	p2.apply(2, i2);
	if(i2[0]!=200  || i2[1]!=100) {
		fail_test("permutation_test::test_permute()",
			__FILE__, __LINE__, "[0,1] permutation failed in int");
	}
	p2.permute(0,1);
	if(!p2.is_identity()) {
		fail_test("permutation_test::test_permute()", __FILE__,
			__LINE__, "Double permutation not recognized");
	}
	strcpy(s2, "ij");
	i2[0] = 100; i2[1] = 200;
	p2.apply(2, s2);
	if(strncmp(s2, "ij", 2)) {
		fail_test("permutation_test::test_permute()", __FILE__,
			__LINE__, "[0,1] double permutation failed in char");
	}
	p2.apply(2, i2);
	if(i2[0]!=100 || i2[1]!=200) {
		fail_test("permutation_test::test_permute()", __FILE__,
			__LINE__, "[0,1] double permutation failed in int");
	}

	permutation<4> p4;
	char s4[8];

	p4.permute(0,1).permute(2,3);
	strcpy(s4, "ijkl");
	p4.apply(4, s4);
	if(strncmp(s4, "jilk", 4)) {
		fail_test("permutation_test::test_permute()", __FILE__,
			__LINE__, "[0,1]+[2,3] permutation failed in char");
	}

	bool ok = false;
	try {
		p4.permute(1,5);
	} catch(exception e) {
		ok = true;
	} catch(...) {
		fail_test("permutation_test::test_permute()",
			__FILE__, __LINE__, "Incorrect exception type");
	}
	if(!ok) {
		fail_test("permutation_test::test_permute()", __FILE__,
			__LINE__, "Expected an exception, it was missing");
	}

}

void permutation_test::test_apply_mask_1() throw(libtest::test_exception) {

	static const char *testname = "permutation_test::test_apply_mask_1()";

	permutation<4> p0, p1, p2;
	p1.permute(2, 3);
	p2.permute(0, 1).permute(2, 3);
	mask<4> m1, m2;
	m1[0] = true; m1[1] = true;
	m2[2] = true; m2[3] = true;

	p2.apply_mask(m1);
	if(!p2.equals(p1)) {
		std::ostringstream ss;
		ss << "Mask 1100 failed: " << p2 << " vs. " << p1 << " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	p2.apply_mask(m2);
	if(!p2.equals(p0)) {
		std::ostringstream ss;
		ss << "Mask 0011 failed: " << p2 << " vs. " << p0 << " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
}

void permutation_test::test_apply_mask_2() throw(libtest::test_exception) {

	static const char *testname = "permutation_test::test_apply_mask_2()";

	permutation<4> p1, p2;
	p1.permute(0, 1).permute(1, 2).permute(2, 3);
	mask<4> m1;
	m1[0] = true; m1[1] = true;

	p1.apply_mask(m1);
	if(!p1.equals(p2)) {
		std::ostringstream ss;
		ss << "Mask 1100 failed: " << p1 << " vs. " << p2 << " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

}

void permutation_test::test_apply_mask_3() throw(libtest::test_exception) {

	static const char *testname = "permutation_test::test_apply_mask_3()";

	permutation<6> p1, p2;
	p1.permute(0, 1).permute(1, 2).permute(2, 3).permute(4, 5);
	p2.permute(4, 5);
	mask<6> m1;
	m1[0] = true; m1[1] = true;

	p1.apply_mask(m1);
	if(!p1.equals(p2)) {
		std::ostringstream ss;
		ss << "Mask 110000 failed: " << p1 << " vs. " << p2 << " (ref).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

}

void permutation_test::test_invert() throw(libtest::test_exception) {
}

void permutation_test::test_apply() throw(libtest::test_exception) {
	bool ok = false;
	permutation<2> p2; permutation<4> p4;
	char s2[8], s4[8];
	strcpy(s2, "ijkl"); strcpy(s4, "ijkl");

	ok = false;
	try {
		p2.apply(2, s2);
		ok = true;
	} catch(...) {
	}
	if(!ok) fail_test("permutation_test::test_apply()",
		__FILE__, __LINE__, "Unexpected exception");

	ok = false;
	try {
		p2.apply(4, s2);
	} catch(exception e) {
		ok = true;
	} catch(...) {
		fail_test("permutation_test::test_apply()",
			__FILE__, __LINE__, "Incorrect exception type");
	}
	if(!ok) fail_test("permutation_test::test_apply()",
		__FILE__, __LINE__, "Expected an exception, it was missing");

	ok = false;
	try {
		p4.apply(4, s4);
		ok = true;
	} catch(...) {
	}
	if(!ok) fail_test("permutation_test::test_apply()",
		__FILE__, __LINE__, "Unexpected exception");

	ok = false;
	try {
		p4.apply(2, s4);
	} catch(exception e) {
		ok = true;
	} catch(...) {
		fail_test("permutation_test::test_apply()",
			__FILE__, __LINE__, "Incorrect exception type");
	}
	if(!ok) fail_test("permutation_test::test_apply()",
		__FILE__, __LINE__, "Expected an exception, it was missing");
}

void permutation_test::test_print() throw(libtest::test_exception) {
	permutation<2> p2;
	std::ostringstream ss;
	ss << p2 << p2;
}

} // namespace libtensor

