#include <libtensor.h>
#include "letter_expr_test.h"

namespace libtensor {

void letter_expr_test::perform() throw(libtest::test_exception) {
	test_contains();
	test_permutation();
}

void letter_expr_test::test_contains() throw(libtest::test_exception) {
	letter i, j, k, l;

	if (!letter_expr<1>(i).contains(i)) {
		fail_test("letter_expr_test::test_contains()", __FILE__,
			__LINE__, "Failed index location: (i).contains(i)");
	}

	if(!(i|j|k).contains(i)) {
		fail_test("letter_expr_test::test_contains()", __FILE__,
			__LINE__, "Failed index location: (i|j|k).contains(i)");
	}
	if((i|j|k).index_of(i) != 0) {
		fail_test("letter_expr_test::test_contains()",
			__FILE__, __LINE__,
			"Failed index determination: (i|j|k).index_of(i)");
	}
	if(!(i|j|k).contains(j)) {
		fail_test("letter_expr_test::test_contains()",
			__FILE__, __LINE__,
			"Failed index location: (i|j|k).contains(j)");
	}
	if((i|j|k).index_of(j) != 1) {
		fail_test("letter_expr_test::test_contains()",
			__FILE__, __LINE__,
			"Failed index determination: (i|j|k).index_of(j)");
	}
	if(!(i|j|k).contains(k)) {
		fail_test("letter_expr_test::test_contains()",
			__FILE__, __LINE__,
			"Failed index location: (i|j|k).contains(k)");
	}
	if((i|j|k).index_of(k) != 2) {
		fail_test("letter_expr_test::test_contains()",
			__FILE__, __LINE__,
			"Failed index determination: (i|j|k).index_of(k)");
	}
	if((i|j|k).contains(l)) {
		fail_test("letter_expr_test::test_contains()",
			__FILE__, __LINE__,
			"Failed index location: (i|j|k).contains(l)");
	}

	if((i|j|k).letter_at(0) != i) {
		fail_test("letter_expr_test::test_contains()",
			__FILE__, __LINE__,
			"Failed letter location: (i|j|k).letter_at(0) #1");
	}
	if((i|j|k).letter_at(0) == j) {
		fail_test("letter_expr_test::test_contains()",
			__FILE__, __LINE__,
			"Failed letter location: (i|j|k).letter_at(0) #2");
	}
	if((i|j|k).letter_at(0) == k) {
		fail_test("letter_expr_test::test_contains()",
			__FILE__, __LINE__,
			"Failed letter location: (i|j|k).letter_at(0) #2");
	}
}

void letter_expr_test::test_permutation() throw(libtest::test_exception) {
	letter a, b, c, d;
	permutation<4> p, p1, p2, p3, p4, p5;

	p = (a|b|c|d).permutation_of(a|b|c|d);
	if(!p.equals(p1)) {
		fail_test("letter_expr_test::test_permutation()", __FILE__,
			__LINE__, "Failed permutation test abcd<-abcd");
	}

	p = (a|b|c|d).permutation_of(a|b|d|c);
	p2.permute(2, 3);
	if(!p.equals(p2)) {
		fail_test("letter_expr_test::test_permutation()", __FILE__,
			__LINE__, "Failed permutation test abcd<-abdc");
	}

	p = (a|b|c|d).permutation_of(b|a|d|c);
	p3.permute(0, 1).permute(2, 3);
	if(!p.equals(p3)) {
		fail_test("letter_expr_test::test_permutation()", __FILE__,
			__LINE__, "Failed permutation test abcd<-badc");
	}

	p = (a|b|c|d).permutation_of(a|c|d|b);
	p4.permute(1, 3).permute(2, 3);
	if(!p.equals(p4)) {
		fail_test("letter_expr_test::test_permutation()", __FILE__,
			__LINE__, "Failed permutation test abcd<-acdb");
	}

	p = (a|b|c|d).permutation_of(b|c|d|a);
	p5.permute(2, 3).permute(1, 2).permute(0, 1);
	if(!p.equals(p5)) {
		fail_test("letter_expr_test::test_permutation()", __FILE__,
			__LINE__, "Failed permutation test abcd<-bcda");
	}
}

} // namespace libtensor
