#include <cmath>
#include <cstdlib>
#include <ctime>
#include <libvmm.h>
#include "compare_ref.h"
#include "tod_contract2_test.h"

namespace libtensor {

typedef libvmm::std_allocator<double> allocator;

void tod_contract2_test::perform() throw(libtest::test_exception) {
	srand48(time(NULL));

	//test_ij_pq_ijpq(1, 1, 1, 1);
	test_ij_pq_ijpq(2, 2, 2, 2);
	//test_ij_pq_ijpq_a(1, 1, 1, 1, 0.25);
	test_ij_pq_ijpq_a(2, 2, 2, 2, 0.25);


	test_ij_ipqr_jpqr(3, 4, 5, 6, 7);
	test_ij_ipqr_jpqr_a(3, 4, 5, 6, 7, -2.0);

	test_ij_pqir_pqjr(3, 4, 5, 6, 7);
	test_ij_pqir_pqjr_a(3, 4, 5, 6, 7, 2.0);
	test_ij_pqir_pqjr(3, 3, 3, 3, 3);
	test_ij_pqir_pqjr(3, 1, 3, 1, 2);
	test_ij_pqir_pqjr(3, 3, 1, 1, 2);

	test_jikl_pi_jpkl(1, 4, 5, 6, 2);
	test_jikl_pi_jpkl(3, 4, 5, 6, 7);
	test_jikl_pi_jpkl_a(3, 4, 5, 6, 7, 0.0);
	test_jikl_pi_jpkl_a(3, 4, 5, 6, 7, -2.0);

	//test_ijkl_ijp_klp(1, 1, 1, 1, 1); // This one fails
	test_ijkl_ijp_klp(3, 4, 5, 6, 7);
	test_ijkl_ijp_klp(5, 6, 3, 4, 7);
	test_ijkl_ijp_klp(1, 100, 1, 100, 100);
	test_ijkl_ijp_klp_a(3, 4, 5, 6, 7, -2.0);

}

void tod_contract2_test::test_ij_pq_ijpq(size_t ni, size_t nj, size_t np,
	size_t nq) throw(libtest::test_exception) {

	// c_{ij} = \sum_{pq} a_{pq} b_{ijpq}

	index<2> ia1, ia2; ia2[0]=np-1; ia2[1]=nq-1;
	index<4> ib1, ib2; ib2[0]=ni-1; ib2[1]=nj-1; ib2[2]=np-1; ib2[3]=nq-1;
	index<2> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1;
	index_range<2> ira(ia1,ia2); dimensions<2> dima(ira);
	index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
	index_range<2> irc(ic1,ic2); dimensions<2> dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<2, double, allocator> ta(dima); tensor_ctrl<2, double> tca(ta);
	tensor<4, double, allocator> tb(dimb); tensor_ctrl<4, double> tcb(tb);
	tensor<2, double, allocator> tc(dimc); tensor_ctrl<2, double> tcc(tc);
	tensor<2, double, allocator> tc_ref(dimc);
	tensor_ctrl<2, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	// Fill in random input

	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();
	for(size_t i=0; i<szc; i++) dtc1[i]=drand48();

	// Generate reference data

	index<2> ia; index<4> ib; index<2> ic;
	double cij_max = 0.0;
	for(size_t i=0; i<ni; i++) {
	for(size_t j=0; j<nj; j++) {
		ic[0]=i; ic[1]=j;
		double cij = 0.0;
		for(size_t p=0; p<np; p++) {
		for(size_t q=0; q<nq; q++) {
			ia[0]=p; ia[1]=q;
			ib[0]=i; ib[1]=j; ib[2]=p; ib[3]=q;
			cij += dta[dima.abs_index(ia)]*dtb[dimb.abs_index(ib)];
		}
		}
		dtc2[dimc.abs_index(ic)] = cij;
		if(fabs(cij) > cij_max) cij_max = fabs(cij);
	}
	}

	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = NULL;
	tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();

	// Invoke the contraction routine

	permutation<2> permc;
	contraction2<0, 2, 2> contr(permc);
	contr.contract(0, 2);
	contr.contract(1, 3);

	tod_contract2<0, 2, 2> op(contr, ta, tb);
	op.perform(tc);

	// Compare against the reference

	char testname[128];
	snprintf(testname, 128,
		"tod_contract2_test::test_ij_pq_ijpq"
		"(%lu, %lu, %lu, %lu)", ni, nj, np, nq);
	compare_ref<2>::compare(testname, tc, tc_ref, cij_max*k_thresh);
}

void tod_contract2_test::test_ij_pq_ijpq_a(size_t ni, size_t nj, size_t np,
	size_t nq, double d) throw(libtest::test_exception) {

	// c_{ij} = c_{ij} + d \sum_{pq} a_{pq} b_{ijpq}

	index<2> ia1, ia2; ia2[0]=np-1; ia2[1]=nq-1;
	index<4> ib1, ib2; ib2[0]=ni-1; ib2[1]=nj-1; ib2[2]=np-1; ib2[3]=nq-1;
	index<2> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1;
	index_range<2> ira(ia1,ia2); dimensions<2> dima(ira);
	index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
	index_range<2> irc(ic1,ic2); dimensions<2> dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<2, double, allocator> ta(dima); tensor_ctrl<2, double> tca(ta);
	tensor<4, double, allocator> tb(dimb); tensor_ctrl<4, double> tcb(tb);
	tensor<2, double, allocator> tc(dimc); tensor_ctrl<2, double> tcc(tc);
	tensor<2, double, allocator> tc_ref(dimc);
	tensor_ctrl<2, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	// Fill in random input

	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();
	for(size_t i=0; i<szc; i++) dtc1[i]=dtc2[i]=drand48();

	// Generate reference data

	index<2> ia; index<4> ib; index<2> ic;
	double cij_max = 0.0;
	for(size_t i=0; i<ni; i++) {
	for(size_t j=0; j<nj; j++) {
		ic[0]=i; ic[1]=j;
		double cij = 0.0;
		for(size_t p=0; p<np; p++) {
		for(size_t q=0; q<nq; q++) {
			ia[0]=p; ia[1]=q;
			ib[0]=i; ib[1]=j; ib[2]=p; ib[3]=q;
			cij += dta[dima.abs_index(ia)]*dtb[dimb.abs_index(ib)];
		}
		}
		dtc2[dimc.abs_index(ic)] += d*cij;
		if(fabs(cij) > cij_max) cij_max = fabs(cij);
	}
	}

	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = NULL;
	tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();

	// Invoke the contraction routine

	permutation<2> permc;
	contraction2<0, 2, 2> contr(permc);
	contr.contract(0, 2);
	contr.contract(1, 3);

	tod_contract2<0, 2, 2> op(contr, ta, tb);
	op.perform(tc, d);

	// Compare against the reference

	char testname[128];
	snprintf(testname, 128,
		"tod_contract2_test::test_ij_pq_ijpq_a"
		"(%lu, %lu, %lu, %lu)", ni, nj, np, nq);
	compare_ref<2>::compare(testname, tc, tc_ref, cij_max*k_thresh);
}

void tod_contract2_test::test_ij_ipqr_jpqr(size_t ni, size_t nj, size_t np,
	size_t nq, size_t nr) throw(libtest::test_exception) {

	// c_{ij} = \sum_{pqr} a_{ipqr} b_{jpqr}

	index<4> ia1, ia2; ia2[0]=ni-1; ia2[1]=np-1; ia2[2]=nq-1; ia2[3]=nr-1;
	index<4> ib1, ib2; ib2[0]=nj-1; ib2[1]=np-1; ib2[2]=nq-1; ib2[3]=nr-1;
	index<2> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1;
	index_range<4> ira(ia1,ia2); dimensions<4> dima(ira);
	index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
	index_range<2> irc(ic1,ic2); dimensions<2> dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<4, double, allocator> ta(dima); tensor_ctrl<4, double> tca(ta);
	tensor<4, double, allocator> tb(dimb); tensor_ctrl<4, double> tcb(tb);
	tensor<2, double, allocator> tc(dimc); tensor_ctrl<2, double> tcc(tc);
	tensor<2, double, allocator> tc_ref(dimc);
	tensor_ctrl<2, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	// Fill in random input

	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();
	for(size_t i=0; i<szc; i++) dtc1[i]=drand48();

	// Generate reference data

	index<4> ia, ib; index<2> ic;
	double cij_max = 0.0;
	for(size_t i=0; i<ni; i++) {
	for(size_t j=0; j<nj; j++) {
		ic[0]=i; ic[1]=j;
		double cij = 0.0;
		for(size_t p=0; p<np; p++) {
		for(size_t q=0; q<nq; q++) {
		for(size_t r=0; r<nr; r++) {
			ia[0]=i; ia[1]=p; ia[2]=q; ia[3]=r;
			ib[0]=j; ib[1]=p; ib[2]=q; ib[3]=r;
			cij += dta[dima.abs_index(ia)]*dtb[dimb.abs_index(ib)];
		}
		}
		}
		dtc2[dimc.abs_index(ic)] = cij;
		if(fabs(cij) > cij_max) cij_max = fabs(cij);
	}
	}

	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = NULL;
	tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();

	// Invoke the contraction routine

	permutation<2> permc;
	contraction2<1, 1, 3> contr(permc);
	contr.contract(1, 1);
	contr.contract(2, 2);
	contr.contract(3, 3);

	tod_contract2<1, 1, 3> op(contr, ta, tb);
	op.perform(tc);

	// Compare against the reference

	char testname[128];
	snprintf(testname, 128,
		"tod_contract2_test::test_ij_ipqr_jpqr"
		"(%lu, %lu, %lu, %lu, %lu)", ni, nj, np, nq, nr);
	compare_ref<2>::compare(testname, tc, tc_ref, cij_max*k_thresh);
}

void tod_contract2_test::test_ij_ipqr_jpqr_a(size_t ni, size_t nj, size_t np,
	size_t nq, size_t nr, double d) throw(libtest::test_exception) {

	// c_{ij} = c_{ij} + d \sum_{pqr} a_{ipqr} b_{jpqr}

	index<4> ia1, ia2; ia2[0]=ni-1; ia2[1]=np-1; ia2[2]=nq-1; ia2[3]=nr-1;
	index<4> ib1, ib2; ib2[0]=nj-1; ib2[1]=np-1; ib2[2]=nq-1; ib2[3]=nr-1;
	index<2> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1;
	index_range<4> ira(ia1,ia2); dimensions<4> dima(ira);
	index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
	index_range<2> irc(ic1,ic2); dimensions<2> dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<4, double, allocator> ta(dima); tensor_ctrl<4, double> tca(ta);
	tensor<4, double, allocator> tb(dimb); tensor_ctrl<4, double> tcb(tb);
	tensor<2, double, allocator> tc(dimc); tensor_ctrl<2, double> tcc(tc);
	tensor<2, double, allocator> tc_ref(dimc);
	tensor_ctrl<2, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	// Fill in random input

	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();
	for(size_t i=0; i<szc; i++) dtc1[i]=dtc2[i]=drand48();

	// Generate reference data

	index<4> ia, ib; index<2> ic;
	double cij_max = 0.0;
	for(size_t i=0; i<ni; i++) {
	for(size_t j=0; j<nj; j++) {
		ic[0]=i; ic[1]=j;
		double cij = 0.0;
		for(size_t p=0; p<np; p++) {
		for(size_t q=0; q<nq; q++) {
		for(size_t r=0; r<nr; r++) {
			ia[0]=i; ia[1]=p; ia[2]=q; ia[3]=r;
			ib[0]=j; ib[1]=p; ib[2]=q; ib[3]=r;
			cij += dta[dima.abs_index(ia)]*dtb[dimb.abs_index(ib)];
		}
		}
		}
		dtc2[dimc.abs_index(ic)] += d*cij;
		if(fabs(cij) > cij_max) cij_max = fabs(cij);
	}
	}

	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = NULL;
	tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();

	// Invoke the contraction routine

	permutation<2> permc;
	contraction2<1, 1, 3> contr(permc);
	contr.contract(1, 1);
	contr.contract(2, 2);
	contr.contract(3, 3);

	tod_contract2<1, 1, 3> op(contr, ta, tb);
	op.perform(tc, d);

	// Compare against the reference

	char testname[128];
	snprintf(testname, 128,
		"tod_contract2_test::test_ij_ipqr_jpqr_a"
		"(%lu, %lu, %lu, %lu, %lu)", ni, nj, np, nq, nr);
	compare_ref<2>::compare(testname, tc, tc_ref, cij_max*k_thresh);
}

void tod_contract2_test::test_ij_pqir_pqjr(size_t ni, size_t nj,
	size_t np, size_t nq, size_t nr) throw(libtest::test_exception) {

	// c_{ij} = \sum_{pqr} a_{pqir} b_{pqjr}

	index<4> ia1, ia2; ia2[0]=np-1; ia2[1]=nq-1; ia2[2]=ni-1; ia2[3]=nr-1;
	index<4> ib1, ib2; ib2[0]=np-1; ib2[1]=nq-1; ib2[2]=nj-1; ib2[3]=nr-1;
	index<2> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1;
	index_range<4> ira(ia1,ia2); dimensions<4> dima(ira);
	index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
	index_range<2> irc(ic1,ic2); dimensions<2> dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<4, double, allocator> ta(dima); tensor_ctrl<4, double> tca(ta);
	tensor<4, double, allocator> tb(dimb); tensor_ctrl<4, double> tcb(tb);
	tensor<2, double, allocator> tc(dimc); tensor_ctrl<2, double> tcc(tc);
	tensor<2, double, allocator> tc_ref(dimc);
	tensor_ctrl<2, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	// Fill in random input

	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();
	for(size_t i=0; i<szc; i++) dtc1[i]=drand48();

	// Generate reference data

	index<4> ia, ib; index<2> ic;
	double cij_max = 0.0;
	for(size_t i=0; i<ni; i++) {
	for(size_t j=0; j<nj; j++) {
		ic[0]=i; ic[1]=j;
		double cij = 0.0;
		for(size_t p=0; p<np; p++) {
		for(size_t q=0; q<nq; q++) {
		for(size_t r=0; r<nr; r++) {
			ia[0]=p; ia[1]=q; ia[2]=i; ia[3]=r;
			ib[0]=p; ib[1]=q; ib[2]=j; ib[3]=r;
			cij += dta[dima.abs_index(ia)]*dtb[dimb.abs_index(ib)];
		}
		}
		}
		dtc2[dimc.abs_index(ic)] = cij;
		if(fabs(cij) > cij_max) cij_max = fabs(cij);
	}
	}

	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = NULL;
	tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();

	// Invoke the contraction routine

	permutation<2> permc;
	contraction2<1, 1, 3> contr(permc);
	contr.contract(0, 0);
	contr.contract(1, 1);
	contr.contract(3, 3);

	tod_contract2<1, 1, 3> op(contr, ta, tb);
	op.perform(tc);

	// Compare against the reference

	char testname[128];
	snprintf(testname, 128,
		"tod_contract2_test::test_ij_pqir_pqjr"
		"(%lu, %lu, %lu, %lu, %lu)", ni, nj, np, nq, nr);
	compare_ref<2>::compare(testname, tc, tc_ref, cij_max*k_thresh);
}

void tod_contract2_test::test_ij_pqir_pqjr_a(size_t ni, size_t nj, size_t np,
	size_t nq, size_t nr, double d) throw(libtest::test_exception) {

	// c_{ij} = c_{ij} + d \sum_{pqr} a_{pqir} b_{pqjr}

	index<4> ia1, ia2; ia2[0]=np-1; ia2[1]=nq-1; ia2[2]=ni-1; ia2[3]=nr-1;
	index<4> ib1, ib2; ib2[0]=np-1; ib2[1]=nq-1; ib2[2]=nj-1; ib2[3]=nr-1;
	index<2> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1;
	index_range<4> ira(ia1,ia2); dimensions<4> dima(ira);
	index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
	index_range<2> irc(ic1,ic2); dimensions<2> dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<4, double, allocator> ta(dima); tensor_ctrl<4, double> tca(ta);
	tensor<4, double, allocator> tb(dimb); tensor_ctrl<4, double> tcb(tb);
	tensor<2, double, allocator> tc(dimc); tensor_ctrl<2, double> tcc(tc);
	tensor<2, double, allocator> tc_ref(dimc);
	tensor_ctrl<2, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	// Fill in random input

	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();
	for(size_t i=0; i<szc; i++) dtc1[i]=dtc2[i]=drand48();

	// Generate reference data

	index<4> ia, ib; index<2> ic;
	double cij_max = 0.0;
	for(size_t i=0; i<ni; i++) {
	for(size_t j=0; j<nj; j++) {
		ic[0]=i; ic[1]=j;
		double cij = 0.0;
		for(size_t p=0; p<np; p++) {
		for(size_t q=0; q<nq; q++) {
		for(size_t r=0; r<nr; r++) {
			ia[0]=p; ia[1]=q; ia[2]=i; ia[3]=r;
			ib[0]=p; ib[1]=q; ib[2]=j; ib[3]=r;
			cij += dta[dima.abs_index(ia)]*dtb[dimb.abs_index(ib)];
		}
		}
		}
		dtc2[dimc.abs_index(ic)] += d*cij;
		if(fabs(cij) > cij_max) cij_max = fabs(cij);
	}
	}

	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = NULL;
	tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();

	// Invoke the contraction routine

	permutation<2> permc;
	contraction2<1, 1, 3> contr(permc);
	contr.contract(0, 0);
	contr.contract(1, 1);
	contr.contract(3, 3);

	tod_contract2<1, 1, 3> op(contr, ta, tb);
	op.perform(tc, d);

	// Compare against the reference

	char testname[128];
	snprintf(testname, 128,
		"tod_contract2_test::test_ij_pqir_pqjr_a"
		"(%lu, %lu, %lu, %lu, %lu)", ni, nj, np, nq, nr);
	compare_ref<2>::compare(testname, tc, tc_ref, cij_max*k_thresh);
}

void tod_contract2_test::test_jikl_pi_jpkl(size_t ni, size_t nj,
	size_t nk, size_t nl, size_t np) throw(libtest::test_exception) {

	// c_{jikl} = \sum_p a_{pi} b_{jpkl}

	index<2> ia1, ia2; ia2[0]=np-1; ia2[1]=ni-1;
	index<4> ib1, ib2; ib2[0]=nj-1; ib2[1]=np-1; ib2[2]=nk-1; ib2[3]=nl-1;
	index<4> ic1, ic2; ic2[0]=nj-1; ic2[1]=ni-1; ic2[2]=nk-1; ic2[3]=nl-1;

	index_range<2> ira(ia1,ia2); dimensions<2> dima(ira);
	index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
	index_range<4> irc(ic1,ic2); dimensions<4> dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<2, double, allocator> ta(dima); tensor_ctrl<2, double> tca(ta);
	tensor<4, double, allocator> tb(dimb); tensor_ctrl<4, double> tcb(tb);
	tensor<4, double, allocator> tc(dimc); tensor_ctrl<4, double> tcc(tc);
	tensor<4, double, allocator> tc_ref(dimc);
	tensor_ctrl<4, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	// Fill in random input

	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();
	for(size_t i=0; i<szc; i++) dtc1[i]=drand48();

	// Generate reference data

	index<2> ia; index<4> ib, ic;
	double cij_max = 0.0;
	for(size_t j=0; j<nj; j++) {
	for(size_t i=0; i<ni; i++) {
	for(size_t k=0; k<nk; k++) {
	for(size_t l=0; l<nl; l++) {
		ic[0]=j; ic[1]=i; ic[2]=k; ic[3]=l;
		double cjikl = 0.0;
		for(size_t p=0; p<np; p++) {
			ia[0]=p; ia[1]=i;
			ib[0]=j; ib[1]=p; ib[2]=k; ib[3]=l;
			cjikl += dta[dima.abs_index(ia)]*
				dtb[dimb.abs_index(ib)];
		}
		dtc2[dimc.abs_index(ic)] = cjikl;
		if(fabs(cjikl) > cij_max) cij_max = fabs(cjikl);
	}
	}
	}
	}

	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = NULL;
	tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();

	// Invoke the contraction routine

	permutation<4> permc; permc.permute(0, 1);
	contraction2<1, 3, 1> contr(permc);
	contr.contract(0, 1);

	tod_contract2<1, 3, 1> op(contr, ta, tb);
	op.perform(tc);

	// Compare against the reference

	char testname[128];
	snprintf(testname, 128,
		"tod_contract2_test::test_jikl_pi_jpkl"
		"(%lu, %lu, %lu, %lu, %lu)", ni, nj, nk, nl, np);
	compare_ref<4>::compare(testname, tc, tc_ref, cij_max*k_thresh);
}

void tod_contract2_test::test_jikl_pi_jpkl_a(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, double d) throw(libtest::test_exception) {

	// c_{jikl} = c_{jikl} + d \sum_p a_{pi} b_{jpkl}

	index<2> ia1, ia2; ia2[0]=np-1; ia2[1]=ni-1;
	index<4> ib1, ib2; ib2[0]=nj-1; ib2[1]=np-1; ib2[2]=nk-1; ib2[3]=nl-1;
	index<4> ic1, ic2; ic2[0]=nj-1; ic2[1]=ni-1; ic2[2]=nk-1; ic2[3]=nl-1;

	index_range<2> ira(ia1,ia2); dimensions<2> dima(ira);
	index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
	index_range<4> irc(ic1,ic2); dimensions<4> dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<2, double, allocator> ta(dima); tensor_ctrl<2, double> tca(ta);
	tensor<4, double, allocator> tb(dimb); tensor_ctrl<4, double> tcb(tb);
	tensor<4, double, allocator> tc(dimc); tensor_ctrl<4, double> tcc(tc);
	tensor<4, double, allocator> tc_ref(dimc);
	tensor_ctrl<4, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	// Fill in random input

	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();
	for(size_t i=0; i<szc; i++) dtc1[i]=dtc2[i]=drand48();

	// Generate reference data

	index<2> ia; index<4> ib, ic;
	double cij_max = 0.0;
	for(size_t j=0; j<nj; j++) {
	for(size_t i=0; i<ni; i++) {
	for(size_t k=0; k<nk; k++) {
	for(size_t l=0; l<nl; l++) {
		ic[0]=j; ic[1]=i; ic[2]=k; ic[3]=l;
		double cjikl = 0.0;
		for(size_t p=0; p<np; p++) {
			ia[0]=p; ia[1]=i;
			ib[0]=j; ib[1]=p; ib[2]=k; ib[3]=l;
			cjikl += dta[dima.abs_index(ia)]*
				dtb[dimb.abs_index(ib)];
		}
		dtc2[dimc.abs_index(ic)] += d*cjikl;
		if(fabs(cjikl) > cij_max) cij_max = fabs(cjikl);
	}
	}
	}
	}

	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = NULL;
	tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();

	// Invoke the contraction routine

	permutation<4> permc; permc.permute(0, 1);
	contraction2<1, 3, 1> contr(permc);
	contr.contract(0, 1);

	tod_contract2<1, 3, 1> op(contr, ta, tb);
	op.perform(tc, d);

	// Compare against the reference

	char testname[128];
	snprintf(testname, 128,
		"tod_contract2_test::test_jikl_pi_jpkl_a"
		"(%lu, %lu, %lu, %lu, %lu)", ni, nj, nk, nl, np);
	compare_ref<4>::compare(testname, tc, tc_ref, cij_max*k_thresh);
}

void tod_contract2_test::test_ijkl_ijp_klp(size_t ni, size_t nj,
	size_t nk, size_t nl, size_t np) throw(libtest::test_exception) {

	// c_{ijkl} = \sum_{p} a_{ijp} b_{klp}

	index<3> ia1, ia2; ia2[0]=ni-1; ia2[1]=nj-1; ia2[2]=np-1;
	index<3> ib1, ib2; ib2[0]=nk-1; ib2[1]=nl-1; ib2[2]=np-1;
	index<4> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1; ic2[2]=nk-1; ic2[3]=nl-1;
	index_range<3> ira(ia1,ia2); dimensions<3> dima(ira);
	index_range<3> irb(ib1,ib2); dimensions<3> dimb(irb);
	index_range<4> irc(ic1,ic2); dimensions<4> dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<3, double, allocator> ta(dima); tensor_ctrl<3, double> tca(ta);
	tensor<3, double, allocator> tb(dimb); tensor_ctrl<3, double> tcb(tb);
	tensor<4, double, allocator> tc(dimc); tensor_ctrl<4, double> tcc(tc);
	tensor<4, double, allocator> tc_ref(dimc);
	tensor_ctrl<4, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	// Fill in random input

	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();
	for(size_t i=0; i<szc; i++) dtc1[i]=drand48();

	// Generate reference data

	index<3> ia, ib; index<4> ic;
	double cij_max = 0.0;
	for(size_t i=0; i<ni; i++) {
	for(size_t j=0; j<nj; j++) {
	for(size_t k=0; k<nk; k++) {
	for(size_t l=0; l<nl; l++) {
		ic[0]=i; ic[1]=j; ic[2]=k; ic[3]=l;
		double cij = 0.0;
		for(size_t p=0; p<np; p++) {
			ia[0]=i; ia[1]=j; ia[2]=p;
			ib[0]=k; ib[1]=l; ib[2]=p;
			cij += dta[dima.abs_index(ia)]*dtb[dimb.abs_index(ib)];
		}
		dtc2[dimc.abs_index(ic)] = cij;
		if(fabs(cij) > cij_max) cij_max = fabs(cij);
	}
	}
	}
	}

	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = NULL;
	tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();

	// Invoke the contraction routine

	permutation<4> permc;
	contraction2<2, 2, 1> contr(permc);
	contr.contract(2, 2);

	tod_contract2<2, 2, 1> op(contr, ta, tb);
	op.perform(tc);

	// Compare against the reference

	char testname[128];
	snprintf(testname, 128,
		"tod_contract2_test::test_ijkl_ijp_klp"
		"(%lu, %lu, %lu, %lu, %lu)", ni, nj, nk, nl, np);
	compare_ref<4>::compare(testname, tc, tc_ref, cij_max*k_thresh);
}

void tod_contract2_test::test_ijkl_ijp_klp_a(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, double d) throw(libtest::test_exception) {

	// c_{ijkl} = c_{ijkl} + d \sum_{p} a_{ijp} b_{klp}

	index<3> ia1, ia2; ia2[0]=ni-1; ia2[1]=nj-1; ia2[2]=np-1;
	index<3> ib1, ib2; ib2[0]=nk-1; ib2[1]=nl-1; ib2[2]=np-1;
	index<4> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1; ic2[2]=nk-1; ic2[3]=nl-1;
	index_range<3> ira(ia1,ia2); dimensions<3> dima(ira);
	index_range<3> irb(ib1,ib2); dimensions<3> dimb(irb);
	index_range<4> irc(ic1,ic2); dimensions<4> dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<3, double, allocator> ta(dima); tensor_ctrl<3, double> tca(ta);
	tensor<3, double, allocator> tb(dimb); tensor_ctrl<3, double> tcb(tb);
	tensor<4, double, allocator> tc(dimc); tensor_ctrl<4, double> tcc(tc);
	tensor<4, double, allocator> tc_ref(dimc);
	tensor_ctrl<4, double> tcc_ref(tc_ref);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();
	double *dtc1 = tcc.req_dataptr();
	double *dtc2 = tcc_ref.req_dataptr();

	// Fill in random input

	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();
	for(size_t i=0; i<szc; i++) dtc1[i]=dtc2[i]=drand48();

	// Generate reference data

	index<3> ia, ib; index<4> ic;
	double cij_max = 0.0;
	for(size_t i=0; i<ni; i++) {
	for(size_t j=0; j<nj; j++) {
	for(size_t k=0; k<nk; k++) {
	for(size_t l=0; l<nl; l++) {
		ic[0]=i; ic[1]=j; ic[2]=k; ic[3]=l;
		double cij = 0.0;
		for(size_t p=0; p<np; p++) {
			ia[0]=i; ia[1]=j; ia[2]=p;
			ib[0]=k; ib[1]=l; ib[2]=p;
			cij += dta[dima.abs_index(ia)]*dtb[dimb.abs_index(ib)];
		}
		dtc2[dimc.abs_index(ic)] += d*cij;
		if(fabs(cij) > cij_max) cij_max = fabs(cij);
	}
	}
	}
	}

	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	tcc.ret_dataptr(dtc1); dtc1 = NULL;
	tcc_ref.ret_dataptr(dtc2); dtc2 = NULL; tc_ref.set_immutable();

	// Invoke the contraction routine

	permutation<4> permc;
	contraction2<2, 2, 1> contr(permc);
	contr.contract(2, 2);

	tod_contract2<2, 2, 1> op(contr, ta, tb);
	op.perform(tc, d);

	// Compare against the reference

	char testname[128];
	snprintf(testname, 128,
		"tod_contract2_test::test_ijkl_ijp_klp_a"
		"(%lu, %lu, %lu, %lu, %lu)", ni, nj, nk, nl, np);
	compare_ref<4>::compare(testname, tc, tc_ref, cij_max*k_thresh);
}

} // namespace libtensor

