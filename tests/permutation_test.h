#ifndef __LIBTENSOR_PERMUTATION_TEST_H
#define __LIBTENSOR_PERMUTATION_TEST_H

#include <libtest.h>

namespace libtensor {

template<class Perm>
class permutation_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	//! Tests the constructors
	void test_ctor() throw(libtest::test_exception);

	//! Tests the permute method
	void test_permute() throw(libtest::test_exception);
};

template<class Perm>
void permutation_test<Perm>::perform() throw(libtest::test_exception) {
	test_ctor();
	test_permute();
}

template<class Perm>
void permutation_test<Perm>::test_ctor() throw(libtest::test_exception) {
	int seq[8];

	Perm p2(2);
	if(!p2.is_identity()) fail_test("permutation_test<Perm>::test_ctor()",
		__FILE__, __LINE__, "!p2.is_identity()");
	for(int i=0; i<8; i++) seq[i]=i;
	p2.apply(2, seq);
	if(seq[0]!=0 || seq[1]!=1)
		fail_test("permutation_test<Perm>::test_ctor()",
		__FILE__, __LINE__, "New permutation is not identity");

	Perm p2a(p2);
	if(!p2a.is_identity()) fail_test("permutation_test<Perm>::test_ctor()",
		__FILE__, __LINE__, "!p2a.is_identity()");
	if(!p2a.equals(p2)) fail_test("permutation_test<Perm>::test_ctor()",
		__FILE__, __LINE__, "!p2a.equals(p2)");

	Perm p3(3);
	if(!p3.is_identity()) fail_test("permutation_test<Perm>::test_ctor()",
		__FILE__, __LINE__, "!p3.is_identity()");

	Perm p3a(p3);
	if(!p3a.is_identity()) fail_test("permutation_test<Perm>::test_ctor()",
		__FILE__, __LINE__, "!p3a.is_identity()");
	if(!p3a.equals(p3)) fail_test("permutation_test<Perm>::test_ctor()",
		__FILE__, __LINE__, "!p3a.equals(p3)");

	Perm p4(4);
	if(!p4.is_identity()) fail_test("permutation_test<Perm>::test_ctor()",
		__FILE__, __LINE__, "!p4.is_identity()");

	Perm p4a(p4);
	if(!p4a.is_identity()) fail_test("permutation_test<Perm>::test_ctor()",
		__FILE__, __LINE__, "!p4a.is_identity()");
	if(!p4a.equals(p4)) fail_test("permutation_test<Perm>::test_ctor()",
		__FILE__, __LINE__, "!p4a.equals(p4)");

	Perm p5(5);
	if(!p5.is_identity()) fail_test("permutation_test<Perm>::test_ctor()",
		__FILE__, __LINE__, "!p5.is_identity()");

	Perm p5a(p5);
	if(!p5a.is_identity()) fail_test("permutation_test<Perm>::test_ctor()",
		__FILE__, __LINE__, "!p5a.is_identity()");
	if(!p5a.equals(p5)) fail_test("permutation_test<Perm>::test_ctor()",
		__FILE__, __LINE__, "!p5a.equals(p5)");

	Perm p6(6);
	if(!p6.is_identity()) fail_test("permutation_test<Perm>::test_ctor()",
		__FILE__, __LINE__, "!p6.is_identity()");
	Perm p6a(p6);
	if(!p6a.is_identity()) fail_test("permutation_test<Perm>::test_ctor()",
		__FILE__, __LINE__, "!p6a.is_identity()");
	if(!p6a.equals(p6)) fail_test("permutation_test<Perm>::test_ctor()",
		__FILE__, __LINE__, "!p6a.equals(p6)");
}

template<class Perm>
void permutation_test<Perm>::test_permute() throw(libtest::test_exception) {
	Perm p2(2);

	int seq[8];
	for(int i=0; i<8; i++) seq[i]=i;

	p2.permute(0,1);
	p2.apply(2, seq);
	if(seq[0]!=1 || seq[1]!=0)
		fail_test("permutation_test<Perm>::test_permute()",
		__FILE__, __LINE__, "p2.permute(0,1);");
}

}

#endif // __LIBTENSOR_PERMUTATION_TEST_H

