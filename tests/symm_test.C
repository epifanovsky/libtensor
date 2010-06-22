#include <libtensor/btod/btod_random.h>
#include <libtensor/iface/iface.h>
#include "compare_ref.h"
#include "symm_test.h"

namespace libtensor {


void symm_test::perform() throw(libtest::test_exception) {

	libvmm::vm_allocator<double>::vmm().init(
		16, 16, 16777216, 16777216, 0.90, 0.05);

	try {

		//~ test_symm2_contr_tt_1();
		//~ test_symm2_contr_ee_1();
		//~ test_asymm2_contr_tt_1();
		//~ test_asymm2_contr_tt_2();
		//~ test_asymm2_contr_tt_3();
		//~ test_asymm2_contr_tt_4();
		test_asymm2_contr_tt_5();
		test_asymm2_contr_ee_1();
		test_asymm2_contr_ee_2();

		test_symm22_t_1();
		test_asymm22_t_1();
		test_symm22_t_2();
		test_asymm22_t_2();

		test_symm22_e_1();
		test_asymm22_e_1();
		test_symm22_e_2();
		test_asymm22_e_2();

	} catch(...) {
		libvmm::vm_allocator<double>::vmm().shutdown();
		throw;
	}

	libvmm::vm_allocator<double>::vmm().shutdown();
}


void symm_test::test_symm2_contr_tt_1() throw(libtest::test_exception) {

	const char *testname = "symm_test::test_symm2_contr_tt_1()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<2> sp_ia(sp_i|sp_a);
	bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
	bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

	btensor<2> t1(sp_ia);
	btensor<4> t2(sp_bcdi);
	btensor<4> t3(sp_abcd), t3_ref(sp_abcd), t3_ref_tmp(sp_abcd);

	btod_random<2>().perform(t1);
	btod_random<4>().perform(t2);
	t1.set_immutable();
	t2.set_immutable();

	contraction2<1, 3, 1> contr;
	contr.contract(0, 3);
	btod_contract2<1, 3, 1> op(contr, t1, t2);
	op.perform(t3_ref_tmp);
	permutation<4> perm; perm.permute(0, 1);
	btod_copy<4>(t3_ref_tmp).perform(t3_ref);
	btod_copy<4>(t3_ref_tmp, perm).perform(t3_ref, 1.0);

	letter a, b, c, d, i;
	t3(a|b|c|d) = symm(a|b, contract(i, t1(i|a), t2(b|c|d|i)));

	compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void symm_test::test_symm2_contr_ee_1() throw(libtest::test_exception) {

	const char *testname = "symm_test::test_symm2_contr_ee_1()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<2> sp_ia(sp_i|sp_a);
	bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
	bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

	btensor<4> t1a(sp_bcdi), t1b(sp_bcdi), t1(sp_bcdi);
	btensor<2> t2a(sp_ia), t2b(sp_ia), t2(sp_ia);
	btensor<4> t3(sp_abcd), t3_ref_tmp(sp_abcd), t3_ref(sp_abcd);

	btod_random<4>().perform(t1a);
	btod_random<4>().perform(t1b);
	btod_random<2>().perform(t2a);
	btod_random<2>().perform(t2b);
	t1a.set_immutable();
	t1b.set_immutable();
	t2a.set_immutable();
	t2b.set_immutable();

	btod_add<4> op_add1(t1a);
	op_add1.add_op(t1b);
	op_add1.perform(t1);
	t1.set_immutable();

	btod_add<2> op_add2(t2a);
	op_add2.add_op(t2b);
	op_add2.perform(t2);
	t2.set_immutable();

	permutation<4> perm_c; perm_c.permute(1, 2).permute(1, 3);
	contraction2<3, 1, 1> contr(perm_c);
	contr.contract(3, 0);
	btod_contract2<3, 1, 1> op(contr, t1, t2);
	op.perform(t3_ref_tmp);

	permutation<4> perm_sym; perm_sym.permute(0, 1);
	btod_add<4> op_sym(t3_ref_tmp);
	op_sym.add_op(t3_ref_tmp, perm_sym);
	op_sym.perform(t3_ref);

	letter a, b, c, d, i;
	t3(a|b|c|d) = symm(a|b,
		contract(i, t1a(a|c|d|i) + t1b(a|c|d|i), t2a(i|b) + t2b(i|b)));

	compare_ref<4>::compare(testname, t3, t3_ref, 2e-14);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void symm_test::test_asymm2_contr_tt_1() throw(libtest::test_exception) {

	const char *testname = "asym_contract_test::test_asymm2_contr_tt_1()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<2> sp_ia(sp_i|sp_a);
	bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
	bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

	btensor<2> t1(sp_ia);
	btensor<4> t2(sp_bcdi);
	btensor<4> t3(sp_abcd), t3_ref(sp_abcd), t3_ref_tmp(sp_abcd);

	btod_random<2>().perform(t1);
	btod_random<4>().perform(t2);
	t1.set_immutable();
	t2.set_immutable();

	contraction2<1, 3, 1> contr;
	contr.contract(0, 3);
	btod_contract2<1, 3, 1> op(contr, t1, t2);
	op.perform(t3_ref_tmp);
	permutation<4> perm; perm.permute(0, 1);
	btod_copy<4>(t3_ref_tmp).perform(t3_ref);
	btod_copy<4>(t3_ref_tmp, perm).perform(t3_ref, -1.0);

	letter a, b, c, d, i;
	t3(a|b|c|d) = asymm(a|b, contract(i, t1(i|a), t2(b|c|d|i)));

	compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void symm_test::test_asymm2_contr_tt_2() throw(libtest::test_exception) {

	const char *testname = "symm_test::test_asymm2_contr_tt_2()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<2> sp_ab(sp_a&sp_a);
	bispace<4> sp_ijab(sp_i&sp_i|sp_a&sp_a);

	btensor<4> t0(sp_ijab);
	btensor<4> t1(sp_ijab);
	btensor<2> t2(sp_ab);
	btensor<4> t3(sp_ijab), t3_ref(sp_ijab), t3_ref_tmp(sp_ijab);

	btod_random<4>().perform(t0);
	btod_random<4>().perform(t1);
	btod_random<2>().perform(t2);
	t0.set_immutable();
	t1.set_immutable();
	t2.set_immutable();

	contraction2<3, 1, 1> contr;
	contr.contract(3, 1);
	btod_contract2<3, 1, 1> op(contr, t1, t2);
	op.perform(t3_ref_tmp);
	permutation<4> perm; perm.permute(2, 3);
	btod_copy<4>(t0).perform(t3_ref);
	btod_copy<4>(t3_ref_tmp).perform(t3_ref, 1.0);
	btod_copy<4>(t3_ref_tmp, perm).perform(t3_ref, -1.0);

	letter i, j, a, b, c;
	t3(i|j|a|b) = t0(i|j|a|b) + asymm(a|b, contract(c, t1(i|j|a|c), t2(b|c)));

	compare_ref<4>::compare(testname, t3, t3_ref, 2e-14);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void symm_test::test_asymm2_contr_tt_3() throw(libtest::test_exception) {

	const char *testname = "symm_test::test_asymm2_contr_tt_3()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<2> sp_ab(sp_a&sp_a);
	bispace<4> sp_ijab(sp_i&sp_i|sp_a&sp_a);

	btensor<4> t1(sp_ijab);
	btensor<2> t2(sp_ab);
	btensor<4> t3(sp_ijab), t3_ref(sp_ijab), t3_ref_tmp(sp_ijab);

	btod_random<4>().perform(t1);
	btod_random<2>().perform(t2);
	t1.set_immutable();
	t2.set_immutable();

	contraction2<3, 1, 1> contr;
	contr.contract(3, 1);
	btod_contract2<3, 1, 1> op(contr, t1, t2);
	op.perform(t3_ref_tmp);
	permutation<4> perm; perm.permute(2, 3);
	btod_copy<4>(t3_ref_tmp).perform(t3_ref, 1.5);
	btod_copy<4>(t3_ref_tmp, perm).perform(t3_ref, -1.5);

	letter i, j, a, b, c;
	t3(i|j|a|b) = 1.5 * asymm(a|b, contract(c, t1(i|j|a|c), t2(b|c)));

	compare_ref<4>::compare(testname, t3, t3_ref, 2e-14);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void symm_test::test_asymm2_contr_tt_4() throw(libtest::test_exception) {

	const char *testname = "symm_test::test_asymm2_contr_tt_4()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<2> sp_ab(sp_a&sp_a);
	bispace<4> sp_ijab(sp_i&sp_i|sp_a&sp_a);

	btensor<4> t1(sp_ijab);
	btensor<2> t2(sp_ab);
	btensor<4> t3(sp_ijab), t3_ref(sp_ijab), t3_ref_tmp(sp_ijab);

	btod_random<4>().perform(t1);
	btod_random<2>().perform(t2);
	t1.set_immutable();
	t2.set_immutable();

	contraction2<3, 1, 1> contr;
	contr.contract(3, 1);
	btod_contract2<3, 1, 1> op(contr, t1, t2);
	op.perform(t3_ref_tmp);
	permutation<4> perm; perm.permute(2, 3);
	btod_copy<4>(t3_ref_tmp).perform(t3_ref, 3.0);
	btod_copy<4>(t3_ref_tmp, perm).perform(t3_ref, -3.0);

	letter i, j, a, b, c;
	t3(i|j|a|b) = 1.5 * asymm(a|b, contract(c, t1(i|j|a|c), t2(b|c))) * 2.0;

	compare_ref<4>::compare(testname, t3, t3_ref, 2e-14);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void symm_test::test_asymm2_contr_tt_5() throw(libtest::test_exception) {

	const char *testname = "symm_test::test_asymm2_contr_tt_5()";

	try {

	bispace<1> sp_i(10), sp_a(20), sp_k(11);
	sp_i.split(3).split(5);
	sp_a.split(6).split(13);
	bispace<4> sp_ijka(sp_i&sp_i|sp_k|sp_a), sp_kija(sp_k|sp_i&sp_i|sp_a);
	bispace<4> sp_ijab(sp_i&sp_i|sp_a&sp_a);

	btensor<4> t1(sp_ijka), t2(sp_kija), t3(sp_ijab), t4(sp_ijka),
		t4_ref(sp_ijka), tt1(sp_ijka);

	btod_random<4>().perform(t1);
	btod_random<4>().perform(t2);
	btod_random<4>().perform(t3);
	t1.set_immutable();
	t2.set_immutable();
	t3.set_immutable();

	contraction2<2, 2, 2> contr(permutation<4>().permute(0, 2));
	contr.contract(1, 1);
	contr.contract(3, 3);
	btod_contract2<2, 2, 2> op_contr(contr, t2, t3);
	btod_symmetrize<4> op_symm(op_contr, 0, 1, false);
	op_symm.perform(tt1);
	btod_copy<4>(t1).perform(t4_ref);
	btod_copy<4>(tt1).perform(t4_ref, -1.0);

	letter i, j, k, l, a, b, c;
	t4(i|j|k|a) = t1(i|j|k|a) - asymm(i|j, contract(l|c,
		t2(k|l|j|c), t3(i|l|a|c)));

	{
		block_tensor_ctrl<4, double> c4(t4), c4_ref(t4_ref);
		symmetry<4, double> sym4(sp_ijka.get_bis()),
			sym4_ref(sp_ijka.get_bis());
		so_copy<4, double>(c4.req_const_symmetry()).perform(sym4);
		so_copy<4, double>(c4_ref.req_const_symmetry()).perform(sym4_ref);
		compare_ref<4>::compare(testname, sym4, sym4_ref);
	}

	compare_ref<4>::compare(testname, t4, t4_ref, 5e-14);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void symm_test::test_asymm2_contr_ee_1() throw(libtest::test_exception) {

	const char *testname = "symm_test::test_asymm2_contr_ee_1()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<2> sp_ia(sp_i|sp_a);
	bispace<4> sp_bcdi(sp_a|sp_a|sp_a|sp_i);
	bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

	btensor<4> t1a(sp_bcdi), t1b(sp_bcdi), t1(sp_bcdi);
	btensor<2> t2a(sp_ia), t2b(sp_ia), t2(sp_ia);
	btensor<4> t3(sp_abcd), t3_ref_tmp(sp_abcd), t3_ref(sp_abcd);

	btod_random<4>().perform(t1a);
	btod_random<4>().perform(t1b);
	btod_random<2>().perform(t2a);
	btod_random<2>().perform(t2b);
	t1a.set_immutable();
	t1b.set_immutable();
	t2a.set_immutable();
	t2b.set_immutable();

	btod_add<4> op_add1(t1a);
	op_add1.add_op(t1b);
	op_add1.perform(t1);
	t1.set_immutable();

	btod_add<2> op_add2(t2a);
	op_add2.add_op(t2b);
	op_add2.perform(t2);
	t2.set_immutable();

	permutation<4> perm_c; perm_c.permute(1, 2).permute(1, 3);
	contraction2<3, 1, 1> contr(perm_c);
	contr.contract(3, 0);
	btod_contract2<3, 1, 1> op(contr, t1, t2);
	op.perform(t3_ref_tmp);

	permutation<4> perm_sym; perm_sym.permute(0, 1);
	btod_add<4> op_sym(t3_ref_tmp);
	op_sym.add_op(t3_ref_tmp, perm_sym, -1.0);
	op_sym.perform(t3_ref);

	letter a, b, c, d, i;
	t3(a|b|c|d) = asymm(a|b,
		contract(i, t1a(a|c|d|i) + t1b(a|c|d|i), t2a(i|b) + t2b(i|b)));

	compare_ref<4>::compare(testname, t3, t3_ref, 2e-14);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void symm_test::test_asymm2_contr_ee_2() throw(libtest::test_exception) {

	const char *testname = "symm_test::test_asymm2_contr_ee_2()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	sp_i.split(5);
	sp_a.split(8); sp_a.split(16);
	bispace<2> sp_ij(sp_i|sp_i);
	bispace<4> sp_ijab(sp_i|sp_i|sp_a|sp_a);

	btensor<2> bta(sp_ij);
	btensor<4> btb(sp_ijab), btc(sp_ijab), btc_ref(sp_ijab);

	permutation<2> p10;
	p10.permute(0, 1);
	se_perm<2, double> sp10(p10, true);
	permutation<4> p1023, p0132;
	p1023.permute(0, 1); p0132.permute(2, 3);
	se_perm<4, double> sp1023(p1023, false), sp0132(p0132, false);

	btod_random<2>().perform(bta);
	btod_random<4>().perform(btb);
	bta.set_immutable();
	btb.set_immutable();

	contraction2<1, 3, 1> contr;
	contr.contract(1, 1);
	btod_contract2<1, 3, 1> op(contr, bta, btb);
	btod_symmetrize<4> op_sym(op, 0, 1, false);
	op_sym.perform(btc_ref);

	letter a, b, i, j, k;
	btc(i|j|a|b) = asymm(i|j, contract(k, bta(i|k), btb(j|k|a|b)));

	compare_ref<4>::compare(testname, btc, btc_ref, 2e-14);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}



/**	\test Tests the symmetrization over two pairs of indexes P+(ij)P+(ab)
		in a %tensor
 **/
void symm_test::test_symm22_t_1() throw(libtest::test_exception) {

	const char *testname = "symm_test::test_symm22_t_1()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<4> sp_ijab(sp_i&sp_i|sp_a&sp_a);

	btensor<4> t1(sp_ijab), t2(sp_ijab), t2_ref(sp_ijab);

	btod_random<4>().perform(t1);
	btod_random<4>().perform(t2);
	t1.set_immutable();

	permutation<4> perm1; perm1.permute(0, 1); // ijab -> jiab
	permutation<4> perm2; perm2.permute(2, 3); // ijab -> ijba
	permutation<4> perm3(perm1); perm3.permute(perm2); // ijab -> jiba
	btod_copy<4>(t1).perform(t2_ref);
	btod_copy<4>(t1, perm1).perform(t2_ref, 1.0);
	btod_copy<4>(t1, perm2).perform(t2_ref, 1.0);
	btod_copy<4>(t1, perm3).perform(t2_ref, 1.0);

	letter i, j, a, b;
	t2(i|j|a|b) = symm(i|j, a|b, t1(i|j|a|b));

	compare_ref<4>::compare(testname, t2, t2_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the anti-symmetrization over two pairs of indexes
		P-(ij)P-(ab) in a %tensor
 **/
void symm_test::test_asymm22_t_1() throw(libtest::test_exception) {

	const char *testname = "symm_test::test_asymm22_t_1()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<4> sp_ijab(sp_i&sp_i|sp_a&sp_a);

	btensor<4> t1(sp_ijab), t2(sp_ijab), t2_ref(sp_ijab);

	btod_random<4>().perform(t1);
	btod_random<4>().perform(t2);
	t1.set_immutable();

	permutation<4> perm1; perm1.permute(0, 1); // ijab -> jiab
	permutation<4> perm2; perm2.permute(2, 3); // ijab -> ijba
	permutation<4> perm3(perm1); perm3.permute(perm2); // ijab -> jiba
	btod_copy<4>(t1).perform(t2_ref);
	btod_copy<4>(t1, perm1).perform(t2_ref, -1.0);
	btod_copy<4>(t1, perm2).perform(t2_ref, -1.0);
	btod_copy<4>(t1, perm3).perform(t2_ref, 1.0);

	letter i, j, a, b;
	t2(i|j|a|b) = asymm(i|j, a|b, t1(i|j|a|b));

	compare_ref<4>::compare(testname, t2, t2_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the symmetrization over two pairs of indexes P+(i|jk)
		in a %tensor
 **/
void symm_test::test_symm22_t_2() throw(libtest::test_exception) {

	const char *testname = "symm_test::test_symm22_t_2()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<4> sp_ijka(sp_i&sp_i&sp_i|sp_a);

	btensor<4> t1(sp_ijka), t2a(sp_ijka), t2b(sp_ijka), t2c(sp_ijka),
		t2d(sp_ijka), t2_ref(sp_ijka);

	btod_random<4>().perform(t1);
	t1.set_immutable();

	permutation<4> perm1; perm1.permute(0, 1); // ijka -> jika
	permutation<4> perm2; perm2.permute(0, 2); // ijka -> kjia
	btod_copy<4>(t1).perform(t2_ref);
	btod_copy<4>(t1, perm1).perform(t2_ref, 1.0);
	btod_copy<4>(t1, perm2).perform(t2_ref, 1.0);

	letter i, j, k, a;
	t2a(i|j|k|a) = symm(i|j, i|k, t1(i|j|k|a));
	t2b(i|j|k|a) = symm(j|i, i|k, t1(i|j|k|a));
	t2c(i|j|k|a) = symm(i|j, k|i, t1(i|j|k|a));
	t2d(i|j|k|a) = symm(j|i, k|i, t1(i|j|k|a));

	compare_ref<4>::compare(testname, t2a, t2_ref, 1e-15);
	compare_ref<4>::compare(testname, t2b, t2_ref, 1e-15);
	compare_ref<4>::compare(testname, t2c, t2_ref, 1e-15);
	compare_ref<4>::compare(testname, t2d, t2_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the anti-symmetrization over two pairs of indexes P-(i|jk)
		in a %tensor
 **/
void symm_test::test_asymm22_t_2() throw(libtest::test_exception) {

	const char *testname = "symm_test::test_asymm22_t_2()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<4> sp_ijka(sp_i&sp_i&sp_i|sp_a);

	btensor<4> t1(sp_ijka), t2a(sp_ijka), t2b(sp_ijka), t2c(sp_ijka),
		t2d(sp_ijka), t2_ref(sp_ijka);

	btod_random<4>().perform(t1);
	t1.set_immutable();

	permutation<4> perm1; perm1.permute(0, 1); // ijka -> jika
	permutation<4> perm2; perm2.permute(0, 2); // ijka -> kjia
	btod_copy<4>(t1).perform(t2_ref);
	btod_copy<4>(t1, perm1).perform(t2_ref, -1.0);
	btod_copy<4>(t1, perm2).perform(t2_ref, -1.0);

	letter i, j, k, a;
	t2a(i|j|k|a) = asymm(i|j, i|k, t1(i|j|k|a));
	t2b(i|j|k|a) = asymm(j|i, i|k, t1(i|j|k|a));
	t2c(i|j|k|a) = asymm(i|j, k|i, t1(i|j|k|a));
	t2d(i|j|k|a) = asymm(j|i, k|i, t1(i|j|k|a));

	compare_ref<4>::compare(testname, t2a, t2_ref, 1e-15);
	compare_ref<4>::compare(testname, t2b, t2_ref, 1e-15);
	compare_ref<4>::compare(testname, t2c, t2_ref, 1e-15);
	compare_ref<4>::compare(testname, t2d, t2_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the symmetrization over two pairs of indexes P+(ij)P+(ab)
		in an expression
 **/
void symm_test::test_symm22_e_1() throw(libtest::test_exception) {

	const char *testname = "symm_test::test_symm22_e_1()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<4> sp_ijab(sp_i&sp_i|sp_a&sp_a);

	btensor<4> t1a(sp_ijab), t1b(sp_ijab), t1(sp_ijab),
		t2(sp_ijab), t2_ref(sp_ijab);

	btod_random<4>().perform(t1a);
	btod_random<4>().perform(t1b);
	btod_random<4>().perform(t2);
	t1a.set_immutable();
	t1b.set_immutable();

	permutation<4> perm1; perm1.permute(0, 1); // ijab -> jiab
	permutation<4> perm2; perm2.permute(2, 3); // ijab -> ijba
	permutation<4> perm3(perm1); perm3.permute(perm2); // ijab -> jiba
	btod_copy<4>(t1a).perform(t1);
	btod_copy<4>(t1b).perform(t1, 1.0);
	btod_copy<4>(t1).perform(t2_ref);
	btod_copy<4>(t1, perm1).perform(t2_ref, 1.0);
	btod_copy<4>(t1, perm2).perform(t2_ref, 1.0);
	btod_copy<4>(t1, perm3).perform(t2_ref, 1.0);

	letter i, j, a, b;
	t2(i|j|a|b) = symm(i|j, a|b, t1a(i|j|a|b) + t1b(i|j|a|b));

	compare_ref<4>::compare(testname, t2, t2_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the anti-symmetrization over two pairs of indexes
		P-(ij)P-(ab) in an expression
 **/
void symm_test::test_asymm22_e_1() throw(libtest::test_exception) {

	const char *testname = "symm_test::test_asymm22_e_1()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<4> sp_ijab(sp_i&sp_i|sp_a&sp_a);

	btensor<4> t1a(sp_ijab), t1b(sp_ijab), t1(sp_ijab),
		t2(sp_ijab), t2_ref(sp_ijab);

	btod_random<4>().perform(t1a);
	btod_random<4>().perform(t1b);
	btod_random<4>().perform(t2);
	t1a.set_immutable();
	t1b.set_immutable();

	permutation<4> perm1; perm1.permute(0, 1); // ijab -> jiab
	permutation<4> perm2; perm2.permute(2, 3); // ijab -> ijba
	permutation<4> perm3(perm1); perm3.permute(perm2); // ijab -> jiba
	btod_copy<4>(t1a).perform(t1);
	btod_copy<4>(t1b).perform(t1, 1.0);
	btod_copy<4>(t1).perform(t2_ref);
	btod_copy<4>(t1, perm1).perform(t2_ref, -1.0);
	btod_copy<4>(t1, perm2).perform(t2_ref, -1.0);
	btod_copy<4>(t1, perm3).perform(t2_ref, 1.0);

	letter i, j, a, b;
	t2(i|j|a|b) = asymm(i|j, a|b, t1a(i|j|a|b) + t1b(i|j|a|b));

	compare_ref<4>::compare(testname, t2, t2_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the symmetrization over two pairs of indexes P+(i|jk)
		in an expression
 **/
void symm_test::test_symm22_e_2() throw(libtest::test_exception) {

	const char *testname = "symm_test::test_symm22_e_2()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<4> sp_ijka(sp_i&sp_i&sp_i|sp_a);

	btensor<4> t1a(sp_ijka), t1b(sp_ijka), t1(sp_ijka), t2a(sp_ijka),
		t2b(sp_ijka), t2c(sp_ijka), t2d(sp_ijka), t2_ref(sp_ijka);

	btod_random<4>().perform(t1a);
	btod_random<4>().perform(t1b);
	t1a.set_immutable();
	t1b.set_immutable();

	permutation<4> perm1; perm1.permute(0, 1); // ijka -> jika
	permutation<4> perm2; perm2.permute(0, 2); // ijka -> kjia
	btod_copy<4>(t1a).perform(t1);
	btod_copy<4>(t1b).perform(t1, 1.0);
	btod_copy<4>(t1).perform(t2_ref);
	btod_copy<4>(t1, perm1).perform(t2_ref, 1.0);
	btod_copy<4>(t1, perm2).perform(t2_ref, 1.0);

	letter i, j, k, a;
	t2a(i|j|k|a) = symm(i|j, i|k, t1a(i|j|k|a) + t1b(i|j|k|a));
	t2b(i|j|k|a) = symm(j|i, i|k, t1a(i|j|k|a) + t1b(i|j|k|a));
	t2c(i|j|k|a) = symm(i|j, k|i, t1a(i|j|k|a) + t1b(i|j|k|a));
	t2d(i|j|k|a) = symm(j|i, k|i, t1a(i|j|k|a) + t1b(i|j|k|a));

	compare_ref<4>::compare(testname, t2a, t2_ref, 1e-15);
	compare_ref<4>::compare(testname, t2b, t2_ref, 1e-15);
	compare_ref<4>::compare(testname, t2c, t2_ref, 1e-15);
	compare_ref<4>::compare(testname, t2d, t2_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\test Tests the anti-symmetrization over two pairs of indexes P-(i|jk)
		in an expression
 **/
void symm_test::test_asymm22_e_2() throw(libtest::test_exception) {

	const char *testname = "symm_test::test_asymm22_e_2()";

	try {

	bispace<1> sp_i(10), sp_a(20);
	bispace<4> sp_ijka(sp_i&sp_i&sp_i|sp_a);

	btensor<4> t1a(sp_ijka), t1b(sp_ijka), t1(sp_ijka), t2a(sp_ijka),
		t2b(sp_ijka), t2c(sp_ijka), t2d(sp_ijka), t2_ref(sp_ijka);

	btod_random<4>().perform(t1a);
	btod_random<4>().perform(t1b);
	t1a.set_immutable();
	t1b.set_immutable();

	permutation<4> perm1; perm1.permute(0, 1); // ijka -> jika
	permutation<4> perm2; perm2.permute(0, 2); // ijka -> kjia
	btod_copy<4>(t1a).perform(t1);
	btod_copy<4>(t1b).perform(t1, 1.0);
	btod_copy<4>(t1).perform(t2_ref);
	btod_copy<4>(t1, perm1).perform(t2_ref, -1.0);
	btod_copy<4>(t1, perm2).perform(t2_ref, -1.0);

	letter i, j, k, a;
	t2a(i|j|k|a) = asymm(i|j, i|k, t1a(i|j|k|a) + t1b(i|j|k|a));
	t2b(i|j|k|a) = asymm(j|i, i|k, t1a(i|j|k|a) + t1b(i|j|k|a));
	t2c(i|j|k|a) = asymm(i|j, k|i, t1a(i|j|k|a) + t1b(i|j|k|a));
	t2d(i|j|k|a) = asymm(j|i, k|i, t1a(i|j|k|a) + t1b(i|j|k|a));

	compare_ref<4>::compare(testname, t2a, t2_ref, 1e-15);
	compare_ref<4>::compare(testname, t2b, t2_ref, 1e-15);
	compare_ref<4>::compare(testname, t2c, t2_ref, 1e-15);
	compare_ref<4>::compare(testname, t2d, t2_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
