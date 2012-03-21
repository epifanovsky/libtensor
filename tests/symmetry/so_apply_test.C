#include <typeinfo>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_apply.h>
//#include <libtensor/btod/transf_double.h>
#include "../compare_ref.h"
#include "so_apply_test.h"

namespace libtensor {


void so_apply_test::perform() throw(libtest::test_exception) {

	test_1( true, true, false);
    test_1(false, true, false);
	test_1( true, false, false);
	test_1( true, false,  true);
    test_1(false, false, false);
    test_1(false, false,  true);
	test_2( true, true, false);
    test_2(false, true, false);
	test_2( true, false, false);
	test_2( true, false,  true);
    test_2(false, false, false);
    test_2(false, false,  true);
	test_3( true, true, false);
    test_3(false, true, false);
	test_3( true, false, false);
	test_3( true, false,  true);
    test_3(false, false, false);
    test_3(false, false,  true);
}


/**	\test Empty %symmetry in 4-space.
 **/
void so_apply_test::test_1(bool keep_zero,
		bool is_asym, bool sign) throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "so_apply_test::test_1(" << keep_zero << ", " << is_asym << ", "
			<< sign << ")";

	try {

	index<4> i1, i2;
	i2[0] = 5; i2[1] = 5; i2[2] = 10; i2[3] = 10;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));

	symmetry<4, double> sym1(bis), sym2(bis);
	permutation<4> perm1;

	so_apply<4, double>(sym1, perm1, keep_zero, is_asym, sign).perform(sym2);

	symmetry<4, double>::iterator j2 = sym2.begin();
	if(j2 != sym2.end()) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, "j2 != sym2.end()");
	}

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}

}


/**	\test Non-empty perm %symmetry in 4-space.
 **/
void so_apply_test::test_2(bool keep_zero,
		bool is_asym, bool sign) throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "so_apply_test::test_2(" << keep_zero << ", " << is_asym << ", "
			<< sign << ")";

	try {

	index<4> i1, i2;
	i2[0] = 5; i2[1] = 5; i2[2] = 10; i2[3] = 10;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));

	symmetry<4, double> sym1(bis), sym2(bis), sym2_ref(bis);
	permutation<4> perm1;

	sym2.insert(se_perm<4, double>(permutation<4>().permute(0, 1), true));
	sym2.insert(se_perm<4, double>(permutation<4>().permute(2, 3), false));

	if (! is_asym) {
		sym2_ref.insert(se_perm<4, double>(
				permutation<4>().permute(0, 1), true));
		sym2_ref.insert(se_perm<4, double>(
				permutation<4>().permute(2, 3), sign ? true : false));
	}

	so_apply<4, double>(sym1, perm1, keep_zero, is_asym, sign).perform(sym2);

	compare_ref<4>::compare(tnss.str().c_str(), sym2, sym2_ref);

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}

}


/**	\test Non-empty perm %symmetry in 4-space with permutation.
 **/
void so_apply_test::test_3(bool keep_zero,
		bool is_asym, bool sign) throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "so_apply_test::test_3(" << keep_zero << ", "
	        << is_asym << ", " << sign << ")";

	try {

	index<4> i1, i2;
	i2[0] = 5; i2[1] = 5; i2[2] = 10; i2[3] = 10;
	block_index_space<4> bis(dimensions<4>(index_range<4>(i1, i2)));

	symmetry<4, double> sym1(bis);
	permutation<4> perm1; perm1.permute(0, 1).permute(1, 2);

	bis.permute(perm1);
	symmetry<4, double> sym2(bis), sym2_ref(bis);

	sym1.insert(se_perm<4, double>(permutation<4>().permute(0, 1), true));
	sym1.insert(se_perm<4, double>(permutation<4>().permute(2, 3), false));

	if (! is_asym) {
		sym2_ref.insert(se_perm<4, double>(
				permutation<4>().permute(0, 2), true));
		sym2_ref.insert(se_perm<4, double>(
				permutation<4>().permute(1, 3), sign ? true : false));
	}

	so_apply<4, double>(sym1, perm1, keep_zero, is_asym, sign).perform(sym2);

	compare_ref<4>::compare(tnss.str().c_str(), sym2, sym2_ref);

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}

}


} // namespace libtensor
