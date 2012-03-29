#include <typeinfo>
#include <libtensor/btod/scalar_transf_double.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/se_part.h>
#include <libtensor/symmetry/so_symmetrize.h>
#include "../compare_ref.h"
#include "so_symmetrize_test.h"

namespace libtensor {


void so_symmetrize_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_4();
	test_5();
}


/**	\test Symmetrization of empty symmetry in 2-space.
 **/
void so_symmetrize_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "so_symmetrize_test::test_1()";

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	mask<2> m;
	m[0] = true; m[1] = true;
	bis.split(m, 2);
	bis.split(m, 5);

	symmetry<2, double> sym1(bis), sym2(bis), sym2_ref(bis);

	scalar_transf<double> tr0;
	sym2_ref.insert(se_perm<2, double>(
		permutation<2>().permute(0, 1), tr0));

	sequence<2, size_t> seq1(0), seq2(1);
	seq1[0] = 1; seq1[1] = 2;
	so_symmetrize<2, double>(sym1, seq1, seq2, tr0, tr0).perform(sym2);

	compare_ref<2>::compare(testname, sym2, sym2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Anti-symmetrization of empty symmetry in 2-space.
 **/
void so_symmetrize_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "so_symmetrize_test::test_2()";

	try {

	index<2> i1, i2;
	i2[0] = 10; i2[1] = 10;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	mask<2> m;
	m[0] = true; m[1] = true;
	bis.split(m, 2);
	bis.split(m, 5);

	symmetry<2, double> sym1(bis), sym2(bis), sym2_ref(bis);

	scalar_transf<double> tr0, tr1(-1.);
	sym2_ref.insert(se_perm<2, double>(
		permutation<2>().permute(0, 1), tr1));

    sequence<2, size_t> seq1(0), seq2(1);
    seq1[0] = 1; seq1[1] = 2;
	so_symmetrize<2, double>(sym1, seq1, seq2, tr1, tr1).perform(sym2);

	compare_ref<2>::compare(testname, sym2, sym2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Symmetrization of S2*S2 in 4-space.
 **/
void so_symmetrize_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "so_symmetrize_test::test_3()";

	try {

	index<4> i1, i2;
	i2[0] = 10; i2[1] = 10; i2[2] = 10; i2[3] = 10;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));
	mask<4> m;
	m[0] = true; m[1] = true; m[2] = true; m[3] = true;
	bis.split(m, 2);
	bis.split(m, 5);

	symmetry<4, double> sym1(bis), sym2(bis), sym2_ref(bis);

	scalar_transf<double> tr0;
	sym1.insert(se_perm<4, double>(
	        permutation<4>().permute(0, 1), tr0));
	sym1.insert(se_perm<4, double>(
	        permutation<4>().permute(2, 3), tr0));
	sym2_ref.insert(se_perm<4, double>(
	        permutation<4>().permute(0, 1), tr0));
    sym2_ref.insert(se_perm<4, double>(
            permutation<4>().permute(2, 3), tr0));
    sym2_ref.insert(se_perm<4, double>(
	        permutation<4>().permute(0, 2).permute(1, 3), tr0));

    sequence<4, size_t> seq1(0), seq2(0);
    seq1[0] = seq1[1] = 1; seq1[2] = seq1[3] = 2;
    seq2[0] = seq2[2] = 1; seq2[1] = seq2[3] = 2;
	so_symmetrize<4, double>(sym1, seq1, seq2, tr0, tr0).perform(sym2);

	compare_ref<4>::compare(testname, sym2, sym2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Symmetrization of 2x2 partition symmetry in 2-space.
 **/
void so_symmetrize_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "so_symmetrize_test::test_4()";

	try {

	index<2> i1, i2;
	i2[0] = 9; i2[1] = 9;
	block_index_space<2> bis(dimensions<2>(index_range<2>(i1, i2)));
	mask<2> m;
	m[0] = true; m[1] = true;
	bis.split(m, 5);

	symmetry<2, double> sym1(bis), sym2(bis), sym2_ref(bis);

	index<2> i00, i01, i10, i11;
	i10[0] = 1; i01[1] = 1; i11[0] = 1; i11[1] = 1;
    scalar_transf<double> tr0;

	se_part<2, double> separt(bis, m, 2);
	separt.add_map(i00, i11, tr0);
	separt.add_map(i01, i10, tr0);

	se_perm<2, double> seperm(permutation<2>().permute(0, 1), tr0);

	sym1.insert(separt);
	sym2_ref.insert(seperm);
	sym2_ref.insert(separt);

    sequence<2, size_t> seq1(0), seq2(0);
    seq1[0] = 1; seq1[1] = 2; seq2[0] = 1; seq2[1] = 1;
	so_symmetrize<2, double>(sym1, seq1, seq2, tr0, tr0).perform(sym2);

	compare_ref<2>::compare(testname, sym2, sym2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


/**	\test Symmetrization of mixed perm/part symmetry in 4-space.
		Case for \f$ b_{ijkl} = P_+(ij) a_{ik} a_{jl} \f$.
 **/
void so_symmetrize_test::test_5() throw(libtest::test_exception) {

	static const char *testname = "so_symmetrize_test::test_5()";

	//	ijkl     +  jikl
	//
	//	A D B C     A'D'B'C'    A+A' D+D' B+B' C+C' ( P S Q R )
	//	D A C B  +  B'C'A'D' =  D+B' A+C' C+A' B+D' ( W T V U )
	//	B C A D     D'A'C'B'    B+D' C+A' A+C' D+B' ( U V T W )
	//	C B D A     C'B'D'A'    C+C' B+B' D+D' A+A' ( R Q S P )
	//
	//	                        In addition, V = T', W = U'
	//	                        (not accounted for by perm symmetry)

	try {

	index<4> i4a, i4b;
	i4b[0] = 9; i4b[1] = 9; i4b[2] = 9; i4b[3] = 9;
	block_index_space<4> bis4(dimensions<4>(index_range<4>(i4a, i4b)));
	mask<4> m4;
	m4[0] = true; m4[1] = true; m4[2] = true; m4[3] = true;
	bis4.split(m4, 5);

	symmetry<4, double> sym1(bis4), sym2(bis4), sym2_ref(bis4);

	index<4> i0000, i1111, i0001, i1110, i0010, i1101, i0011, i1100,
		i0100, i1011, i0101, i1010, i0110, i1001, i1000, i0111;
	i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;
	i1110[0] = 1; i1110[1] = 1; i1110[2] = 1; i0001[3] = 1;
	i1101[0] = 1; i1101[1] = 1; i0010[2] = 1; i1101[3] = 1;
	i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
	i1011[0] = 1; i0100[1] = 1; i1011[2] = 1; i1011[3] = 1;
	i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
	i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
	i1000[0] = 1; i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
    scalar_transf<double> tr0, tr1(-1.0);

	se_part<4, double> separt4a(bis4, m4, 2);
	separt4a.add_map(i0000, i0101, tr0);
	separt4a.add_map(i0101, i1010, tr0);
	separt4a.add_map(i1010, i1111, tr0);
	separt4a.add_map(i0010, i0111, tr0);
	separt4a.add_map(i0111, i1000, tr0);
	separt4a.add_map(i1000, i1101, tr0);
	separt4a.add_map(i0011, i0110, tr0);
	separt4a.add_map(i0110, i1001, tr0);
	separt4a.add_map(i1001, i1100, tr0);
	separt4a.add_map(i0001, i0100, tr0);
	separt4a.add_map(i0100, i1011, tr0);
	separt4a.add_map(i1011, i1110, tr0);

	se_part<4, double> separt4b(bis4, m4, 2);
	separt4b.add_map(i0000, i1111, tr0);
	separt4b.add_map(i0001, i1110, tr0);
	separt4b.add_map(i0010, i1101, tr0);
	separt4b.add_map(i0011, i1100, tr0);
	separt4b.add_map(i0100, i1011, tr0);
	separt4b.add_map(i0101, i1010, tr0);
	separt4b.add_map(i0110, i1001, tr0);
	separt4b.add_map(i0111, i1000, tr0);

	se_perm<4, double> seperm4(permutation<4>().permute(0, 1), tr0);

	sym1.insert(separt4a);
	sym2_ref.insert(seperm4);
	sym2_ref.insert(separt4b);

    sequence<4, size_t> seq1(0), seq2(0);
    seq1[0] = 1; seq1[1] = 2; seq2[0] = 1; seq2[1] = 1;
	so_symmetrize<4, double>(sym1, seq1, seq2, tr0, tr0).perform(sym2);

	compare_ref<4>::compare(testname, sym2, sym2_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


} // namespace libtensor
