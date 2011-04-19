#include <sstream>
#include <string>
#include <cstring>
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
	test_apply_1();
	test_print();
}


void permutation_test::test_ctor() throw(libtest::test_exception) {

	sequence<2, char> sc2('\0');
	sequence<2, int> si2(0);

	// Order = 2

	permutation<2> p2;
	if(!p2.is_identity()) {
		fail_test("permutation_test::test_ctor()", __FILE__,
			__LINE__, "!p2.is_identity()");
	}
	sc2[0] = 'i'; sc2[1] = 'j';
	p2.apply(sc2);
	if(sc2[0] != 'i' || sc2[1] != 'j') {
		fail_test("permutation_test::test_ctor()", __FILE__,
			__LINE__, "New permutation is not identity (char, 2)");
	}
	si2[0] = 0; si2[1] = 1;
	p2.apply(si2);
	if(si2[0] != 0 || si2[1] != 1) {
		fail_test("permutation_test::test_ctor()", __FILE__,
			__LINE__, "New permutation is not identity (int, 2)");
	}

	permutation<2> p2a(p2);
	if(!p2a.is_identity()) fail_test("permutation_test::test_ctor()",
		__FILE__, __LINE__, "!p2a.is_identity()");
	if(!p2a.equals(p2)) fail_test("permutation_test::test_ctor()",
		__FILE__, __LINE__, "!p2a.equals(p2)");
	sc2[0] = 'i'; sc2[1] = 'j';
	p2a.apply(sc2);
	if(sc2[0] != 'i' || sc2[1] != 'j')
		fail_test("permutation_test::test_ctor()", __FILE__,
		__LINE__, "Permutation copy is not identity (char, 2)");
	si2[0] = 0; si2[1] = 1;
	p2a.apply(si2);
	if(si2[0] != 0 || si2[1] != 1)
		fail_test("permutation_test::test_ctor()", __FILE__,
		__LINE__, "Permutation copy is not identity (int, 2)");

	// Order = 3

	sequence<3, char> sc3('\0');
	sequence<3, int> si3(0);

	permutation<3> p3;
	if(!p3.is_identity()) {
		fail_test("permutation_test::test_ctor()", __FILE__,
			__LINE__, "!p3.is_identity()");
	}
	sc3[0] = 'i'; sc3[1] = 'j'; sc3[2] = 'k';
	p3.apply(sc3);
	if(sc3[0] != 'i' || sc3[1] != 'j' || sc3[2] != 'k') {
		fail_test("permutation_test::test_ctor()", __FILE__,
			__LINE__, "New permutation is not identity (char, 3)");
	}
	si3[0] = 0; si3[1] = 1; si3[2] = 2;
	p3.apply(si3);
	if(si3[0] != 0 || si3[1] != 1 || si3[2] != 2) {
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
	sc3[0] = 'i'; sc3[1] = 'j'; sc3[2] = 'k';
	p3a.apply(sc3);
	if(sc3[0] != 'i' || sc3[1] != 'j' || sc3[2] != 'k') {
		fail_test("permutation_test::test_ctor()", __FILE__,
			__LINE__, "Permutation copy is not identity (char, 3)");
	}
	si3[0] = 0; si3[1] = 1; si3[2] = 2;
	p3a.apply(si3);
	if(si3[0] != 0 || si3[1] != 1 || si3[2] != 2) {
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

	sequence<2, char> s2('\0');
	s2[0] = 'i'; s2[1] = 'j';
	sequence<2, int> i2(0);
	i2[0] = 100; i2[1] = 200;

	p2.permute(0, 1);
	p2.apply(s2);
	if(s2[0] != 'j' || s2[1] != 'i') {
		fail_test("permutation_test::test_permute()",
			__FILE__, __LINE__, "[0,1] permutation failed in char");
	}
	p2.apply(i2);
	if(i2[0] != 200  || i2[1] != 100) {
		fail_test("permutation_test::test_permute()",
			__FILE__, __LINE__, "[0,1] permutation failed in int");
	}
	p2.permute(0,1);
	if(!p2.is_identity()) {
		fail_test("permutation_test::test_permute()", __FILE__,
			__LINE__, "Double permutation not recognized");
	}
	s2[0] = 'i'; s2[1] = 'j';
	i2[0] = 100; i2[1] = 200;
	p2.apply(s2);
	if(s2[0] != 'i' || s2[1] != 'j') {
		fail_test("permutation_test::test_permute()", __FILE__,
			__LINE__, "[0,1] double permutation failed in char");
	}
	p2.apply(i2);
	if(i2[0] != 100 || i2[1] != 200) {
		fail_test("permutation_test::test_permute()", __FILE__,
			__LINE__, "[0,1] double permutation failed in int");
	}

	permutation<4> p4;
	sequence<4, char> s4('\0');

	p4.permute(0, 1).permute(2, 3);
	s4[0] = 'i'; s4[1] = 'j'; s4[2] = 'k'; s4[3] = 'l';
	p4.apply(s4);
	if(s4[0] != 'j' || s4[1] != 'i' || s4[2] != 'l' || s4[3] != 'k') {
		fail_test("permutation_test::test_permute()", __FILE__,
			__LINE__, "[0,1]+[2,3] permutation failed in char");
	}

	bool ok = false;
	try {
		p4.permute(1, 5);
	} catch(exception &e) {
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


void permutation_test::test_apply_1() throw(libtest::test_exception) {

	static const char *testname = "permutation_test::test_apply_1()";

	try {

	sequence<2, char> seq2a('\0'), seq2b('\0');
	seq2a[0] = 'a'; seq2a[1] = 'b';
	seq2b[0] = 'b'; seq2b[1] = 'a';
	sequence<4, char> seq4a('\0'), seq4b('\0');
	seq4a[0] = 'a'; seq4a[1] = 'b'; seq4a[2] = 'c'; seq4a[3] = 'd';
	seq4b[0] = 'b'; seq4b[1] = 'c'; seq4b[2] = 'd'; seq4b[3] = 'a';

	permutation<2> p2; p2.permute(0, 1);
	permutation<4> p4; p4.permute(0, 1).permute(1, 2).permute(2, 3);

	p2.apply(seq2a);
	p4.apply(seq4a);

	std::string s2a(2, ' '), s2b(2, ' ');
	for(size_t i = 0; i < 2; i++) {
		s2a[i] = seq2a[i];
		s2b[i] = seq2b[i];
	}
	if(s2a.compare(s2b) != 0) {
		std::ostringstream ss;
		ss << "Test (ab->ba) failed: [" << s2a << "] vs [" << s2b
			<< "] (expected).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	std::string s4a(4, ' '), s4b(4, ' ');
	for(size_t i = 0; i < 4; i++) {
		s4a[i] = seq4a[i];
		s4b[i] = seq4b[i];
	}
	if(s4a.compare(s4b) != 0) {
		std::ostringstream ss;
		ss << "Test (abcd->bcda) failed: [" << s4a << "] vs ["
			<< s4b << "] (expected).";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void permutation_test::test_print() throw(libtest::test_exception) {
	permutation<2> p2;
	std::ostringstream ss;
	ss << p2 << p2;
}


} // namespace libtensor
