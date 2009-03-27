#include <cmath>
#include <cstdlib>
#include <ctime>
#include "tod_contract2_impl_113_test.h"
#include "tensor.h"
#include "tensor_ctrl.h"

namespace libtensor {

typedef libvmm::std_allocator<double> allocator;

void tod_contract2_impl_113_test::perform() throw(libtest::test_exception) {
	srand48(time(NULL));

	test_ij_ipqr_jpqr(3, 4, 5, 6, 7);
	test_ij_ipqr_jpqr_a(3, 4, 5, 6, 7, -2.0);

	test_ij_pqir_pqjr(3, 4, 5, 6, 7);
	test_ij_pqir_pqjr_a(3, 4, 5, 6, 7, 2.0);
	test_ij_pqir_pqjr(3, 3, 3, 3, 3);
	test_ij_pqir_pqjr(3, 1, 3, 1, 2);
	test_ij_pqir_pqjr(3, 3, 1, 1, 2);
}

void tod_contract2_impl_113_test::test_ij_ipqr_jpqr(size_t ni, size_t nj,
	size_t np, size_t nq, size_t nr) throw(libtest::test_exception) {

	// c_{ij} = \sum_{pqr} a_{ipqr} b_{jpqr}

	index<4> ia1, ia2; ia2[0]=ni-1; ia2[1]=np-1; ia2[2]=nq-1; ia2[3]=nr-1;
	index<4> ib1, ib2; ib2[0]=nj-1; ib2[1]=np-1; ib2[2]=nq-1; ib2[3]=nr-1;
	index<2> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1;
	index_range<4> ira(ia1,ia2); dimensions<4> dima(ira);
	index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
	index_range<2> irc(ic1,ic2); dimensions<2> dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<4,double,allocator> ta(dima); tensor_ctrl<4,double> tca(ta);
	tensor<4,double,allocator> tb(dimb); tensor_ctrl<4,double> tcb(tb);
	tensor<2,double,allocator> tc(dimc); tensor_ctrl<2,double> tcc(tc);
	double *dta = new double[sza];
	double *dtb = new double[szb];
	double *dtc = new double[szc];

	// Fill in random input

	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();

	double *ptr = tca.req_dataptr();
	for(size_t i=0; i<sza; i++) ptr[i]=dta[i];
	tca.ret_dataptr(ptr);
	ptr = tcb.req_dataptr();
	for(size_t i=0; i<szb; i++) ptr[i]=dtb[i];
	tcb.ret_dataptr(ptr);
	ta.set_immutable(); tb.set_immutable();

	// Generate reference data

	index<4> ia, ib; index<2> ic;
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
		dtc[dimc.abs_index(ic)] = cij;
	}
	}

	// Invoke the contraction routine

	permutation<4> perma, permb; permutation<2> permc;
	const double *dcta = tca.req_const_dataptr();
	const double *dctb = tcb.req_const_dataptr();
	double *dctc = tcc.req_dataptr();
	tod_contract2_impl<1,1,3>::contract(dctc, dimc, permc, dcta, dima,
		perma, dctb, dimb, permb);
	tca.ret_dataptr(dcta); dcta = NULL;
	tcb.ret_dataptr(dctb); dctb = NULL;

	// Compare against the reference

	bool ok = true;
	size_t ielem;
	double dfail_ref, dfail_act;
	for(ielem=0; ielem<szc; ielem++) {
		if(fabs(dctc[ielem]-dtc[ielem])>fabs(dtc[ielem])*5e-15) {
			dfail_ref = dtc[ielem]; dfail_act = dctc[ielem];
			ok=false; break;
		}
	}

	tcc.ret_dataptr(dctc); dctc = NULL;

	delete [] dtc; delete [] dtb; delete [] dta;

	if(!ok) {
		char method[1024], msg[1024];
		snprintf(method, 1024, "tod_contract2_impl_113_test::"
			"test_ij_ipqr_jpqr(%lu,%lu,%lu,%lu,%lu)",
			ni, nj, np, nq, nr);
		snprintf(msg, 1024, "contract() result does not match "
			"reference at element %lu: %lg(ref) vs %lg(act), "
			"%lg(diff)", ielem, dfail_ref, dfail_act,
			dfail_act-dfail_ref);
		fail_test(method, __FILE__, __LINE__, msg);
	}
}

void tod_contract2_impl_113_test::test_ij_ipqr_jpqr_a(size_t ni, size_t nj,
	size_t np, size_t nq, size_t nr, double d)
	throw(libtest::test_exception) {

	// c_{ij} = c_{ij} + d \sum_{pqr} a_{ipqr} b_{jpqr}

	index<4> ia1, ia2; ia2[0]=ni-1; ia2[1]=np-1; ia2[2]=nq-1; ia2[3]=nr-1;
	index<4> ib1, ib2; ib2[0]=nj-1; ib2[1]=np-1; ib2[2]=nq-1; ib2[3]=nr-1;
	index<2> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1;
	index_range<4> ira(ia1,ia2); dimensions<4> dima(ira);
	index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
	index_range<2> irc(ic1,ic2); dimensions<2> dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<4,double,allocator> ta(dima); tensor_ctrl<4,double> tca(ta);
	tensor<4,double,allocator> tb(dimb); tensor_ctrl<4,double> tcb(tb);
	tensor<2,double,allocator> tc(dimc); tensor_ctrl<2,double> tcc(tc);
	double *dta = new double[sza];
	double *dtb = new double[szb];
	double *dtc = new double[szc];

	// Fill in random input

	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();
	for(size_t i=0; i<szc; i++) dtc[i]=drand48();

	double *ptr = tca.req_dataptr();
	for(size_t i=0; i<sza; i++) ptr[i]=dta[i];
	tca.ret_dataptr(ptr);
	ptr = tcb.req_dataptr();
	for(size_t i=0; i<szb; i++) ptr[i]=dtb[i];
	tcb.ret_dataptr(ptr);
	ptr = tcc.req_dataptr();
	for(size_t i=0; i<szc; i++) ptr[i]=dtc[i];
	tcc.ret_dataptr(ptr);
	ta.set_immutable(); tb.set_immutable();

	// Generate reference data

	index<4> ia, ib; index<2> ic;
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
		dtc[dimc.abs_index(ic)] += d*cij;
	}
	}

	// Invoke the contraction routine

	permutation<4> perma, permb; permutation<2> permc;
	const double *dcta = tca.req_const_dataptr();
	const double *dctb = tcb.req_const_dataptr();
	double *dctc = tcc.req_dataptr();
	tod_contract2_impl<1,1,3>::contract(dctc, dimc, permc, dcta, dima,
		perma, dctb, dimb, permb);
	tca.ret_dataptr(dcta); dcta = NULL;
	tcb.ret_dataptr(dctb); dctb = NULL;

	// Compare against the reference

	bool ok = true;
	size_t ielem;
	double dfail_ref, dfail_act;
	for(ielem=0; ielem<szc; ielem++) {
		if(fabs(dctc[ielem]-dtc[ielem])>fabs(dtc[ielem])*5e-15) {
			dfail_ref = dtc[ielem]; dfail_act = dctc[ielem];
			ok=false; break;
		}
	}

	tcc.ret_dataptr(dctc); dctc = NULL;

	delete [] dtc; delete [] dtb; delete [] dta;

	if(!ok) {
		char method[1024], msg[1024];
		snprintf(method, 1024, "tod_contract2_impl_113_test::"
			"test_ij_ipqr_jpqr_a(%lu,%lu,%lu,%lu,%lu,%lg)",
			ni, nj, np, nq, nr, d);
		snprintf(msg, 1024, "contract() result does not match "
			"reference at element %lu: %lg(ref) vs %lg(act), "
			"%lg(diff)", ielem, dfail_ref, dfail_act,
			dfail_act-dfail_ref);
		fail_test(method, __FILE__, __LINE__, msg);
	}
}

void tod_contract2_impl_113_test::test_ij_pqir_pqjr(size_t ni, size_t nj,
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

	tensor<4,double,allocator> ta(dima); tensor_ctrl<4,double> tca(ta);
	tensor<4,double,allocator> tb(dimb); tensor_ctrl<4,double> tcb(tb);
	tensor<2,double,allocator> tc(dimc); tensor_ctrl<2,double> tcc(tc);
	double *dta = new double[sza];
	double *dtb = new double[szb];
	double *dtc = new double[szc];

	// Fill in random input

	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();

	double *ptr = tca.req_dataptr();
	for(size_t i=0; i<sza; i++) ptr[i]=dta[i];
	tca.ret_dataptr(ptr);
	ptr = tcb.req_dataptr();
	for(size_t i=0; i<szb; i++) ptr[i]=dtb[i];
	tcb.ret_dataptr(ptr);
	ta.set_immutable(); tb.set_immutable();

	// Generate reference data

	index<4> ia, ib; index<2> ic;
	for(size_t i=0; i<dimc[0]; i++) {
	for(size_t j=0; j<dimc[1]; j++) {
		ic[0]=i; ic[1]=j;
		double cij = 0.0;
		for(size_t p=0; p<dima[0]; p++) {
		for(size_t q=0; q<dima[1]; q++) {
		for(size_t r=0; r<dima[3]; r++) {
			ia[0]=p; ia[1]=q; ia[2]=i; ia[3]=r;
			ib[0]=p; ib[1]=q; ib[2]=j; ib[3]=r;
			cij += dta[dima.abs_index(ia)]*dtb[dimb.abs_index(ib)];
		}
		}
		}
		dtc[dimc.abs_index(ic)] = cij;
	}
	}

	// Invoke the contraction routine

	permutation<4> perma, permb; permutation<2> permc;
	perma.permute(0,2).permute(1,2); // pqir -> ipqr
	permb.permute(0,2).permute(1,2); // pqjr -> jpqr
	const double *dcta = tca.req_const_dataptr();
	const double *dctb = tcb.req_const_dataptr();
	double *dctc = tcc.req_dataptr();
	tod_contract2_impl<1,1,3>::contract(dctc, dimc, permc, dcta, dima,
		perma, dctb, dimb, permb);
	tca.ret_dataptr(dcta); dcta = NULL;
	tcb.ret_dataptr(dctb); dctb = NULL;

	// Compare against the reference

	bool ok = true;
	size_t ielem;
	double dfail_ref, dfail_act;
	for(ielem=0; ielem<szc; ielem++) {
		if(fabs(dctc[ielem]-dtc[ielem])>fabs(dtc[ielem])*5e-15) {
			dfail_ref = dtc[ielem]; dfail_act = dctc[ielem];
			ok=false; break;
		}
	}

	tcc.ret_dataptr(dctc); dctc = NULL;

	delete [] dtc; delete [] dtb; delete [] dta;

	if(!ok) {
		char method[1024], msg[1024];
		snprintf(method, 1024, "tod_contract2_impl_113_test::"
			"test_ij_pqir_pqjr(%lu,%lu,%lu,%lu,%lu)",
			ni, nj, np, nq, nr);
		snprintf(msg, 1024, "contract() result does not match "
			"reference at element %lu: %lg(ref) vs %lg(act), "
			"%lg(diff)", ielem, dfail_ref, dfail_act,
			dfail_act-dfail_ref);
		fail_test(method, __FILE__, __LINE__, msg);
	}
}

void tod_contract2_impl_113_test::test_ij_pqir_pqjr_a(size_t ni, size_t nj,
	size_t np, size_t nq, size_t nr, double d)	
	throw(libtest::test_exception) {

	// c_{ij} = c_{ij} + d \sum_{pqr} a_{pqir} b_{pqjr}

	index<4> ia1, ia2; ia2[0]=np-1; ia2[1]=nq-1; ia2[2]=ni-1; ia2[3]=nr-1;
	index<4> ib1, ib2; ib2[0]=np-1; ib2[1]=nq-1; ib2[2]=nj-1; ib2[3]=nr-1;
	index<2> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1;
	index_range<4> ira(ia1,ia2); dimensions<4> dima(ira);
	index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
	index_range<2> irc(ic1,ic2); dimensions<2> dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<4,double,allocator> ta(dima); tensor_ctrl<4,double> tca(ta);
	tensor<4,double,allocator> tb(dimb); tensor_ctrl<4,double> tcb(tb);
	tensor<2,double,allocator> tc(dimc); tensor_ctrl<2,double> tcc(tc);
	double *dta = new double[sza];
	double *dtb = new double[szb];
	double *dtc = new double[szc];

	// Fill in random input

	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();

	double *ptr = tca.req_dataptr();
	for(size_t i=0; i<sza; i++) ptr[i]=dta[i];
	tca.ret_dataptr(ptr);
	ptr = tcb.req_dataptr();
	for(size_t i=0; i<szb; i++) ptr[i]=dtb[i];
	tcb.ret_dataptr(ptr);
	ta.set_immutable(); tb.set_immutable();

	// Generate reference data

	index<4> ia, ib; index<2> ic;
	for(size_t i=0; i<dimc[0]; i++) {
	for(size_t j=0; j<dimc[1]; j++) {
		ic[0]=i; ic[1]=j;
		double cij = 0.0;
		for(size_t p=0; p<dima[0]; p++) {
		for(size_t q=0; q<dima[1]; q++) {
		for(size_t r=0; r<dima[3]; r++) {
			ia[0]=p; ia[1]=q; ia[2]=i; ia[3]=r;
			ib[0]=p; ib[1]=q; ib[2]=j; ib[3]=r;
			cij += dta[dima.abs_index(ia)]*dtb[dimb.abs_index(ib)];
		}
		}
		}
		dtc[dimc.abs_index(ic)] += d*cij;
	}
	}

	// Invoke the contraction routine

	permutation<4> perma, permb; permutation<2> permc;
	perma.permute(0,2).permute(1,2); // pqir -> ipqr
	permb.permute(0,2).permute(1,2); // pqjr -> jpqr
	const double *dcta = tca.req_const_dataptr();
	const double *dctb = tcb.req_const_dataptr();
	double *dctc = tcc.req_dataptr();
	tod_contract2_impl<1,1,3>::contract(dctc, dimc, permc, dcta, dima,
		perma, dctb, dimb, permb);
	tca.ret_dataptr(dcta); dcta = NULL;
	tcb.ret_dataptr(dctb); dctb = NULL;

	// Compare against the reference

	bool ok = true;
	size_t ielem;
	double dfail_ref, dfail_act;
	for(ielem=0; ielem<szc; ielem++) {
		if(fabs(dctc[ielem]-dtc[ielem])>fabs(dtc[ielem])*5e-15) {
			dfail_ref = dtc[ielem]; dfail_act = dctc[ielem];
			ok=false; break;
		}
	}

	tcc.ret_dataptr(dctc); dctc = NULL;

	delete [] dtc; delete [] dtb; delete [] dta;

	if(!ok) {
		char method[1024], msg[1024];
		snprintf(method, 1024, "tod_contract2_impl_113_test::"
			"test_ij_pqir_pqjr_a(%lu,%lu,%lu,%lu,%lu,%lg)",
			ni, nj, np, nq, nr, d);
		snprintf(msg, 1024, "contract() result does not match "
			"reference at element %lu: %lg(ref) vs %lg(act), "
			"%lg(diff)", ielem, dfail_ref, dfail_act,
			dfail_act-dfail_ref);
		fail_test(method, __FILE__, __LINE__, msg);
	}
}

} // namespace libtensor

