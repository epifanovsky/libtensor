#include <cmath>
#include <cstdlib>
#include <ctime>
#include "tod_contract2_impl_221_test.h"
#include "tensor.h"
#include "tensor_ctrl.h"

namespace libtensor {

typedef libvmm::std_allocator<double> allocator;

void tod_contract2_impl_221_test::perform() throw(libtest::test_exception) {
	srand48(time(NULL));

	test_ijkl_ijp_klp(3, 4, 5, 6, 7);
	test_ijkl_ijp_klp(5, 6, 3, 4, 7);
	test_ijkl_ijp_klp(1, 100, 1, 100, 100);
	test_ijkl_ijp_klp_a(3, 4, 5, 6, 7, -2.0);
}

void tod_contract2_impl_221_test::test_ijkl_ijp_klp(size_t ni, size_t nj,
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

	tensor<3,double,allocator> ta(dima); tensor_ctrl<3,double> tca(ta);
	tensor<3,double,allocator> tb(dimb); tensor_ctrl<3,double> tcb(tb);
	tensor<4,double,allocator> tc(dimc); tensor_ctrl<4,double> tcc(tc);
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

	index<3> ia, ib; index<4> ic;
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
		dtc[dimc.abs_index(ic)] = cij;
	}
	}
	}
	}

	// Invoke the contraction routine

	permutation<3> perma, permb; permutation<4> permc;
	const double *dcta = tca.req_const_dataptr();
	const double *dctb = tcb.req_const_dataptr();
	double *dctc = tcc.req_dataptr();
	tod_contract2_impl<2,2,1>::contract(dctc, dimc, permc, dcta, dima,
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
		snprintf(method, 1024, "tod_contract2_impl_221_test::"
			"test_ijkl_ijp_klp(%lu,%lu,%lu,%lu,%lu)",
			ni, nj, nk, nl, np);
		snprintf(msg, 1024, "contract() result does not match "
			"reference at element %lu: %lg(ref) vs %lg(act), "
			"%lg(diff)", ielem, dfail_ref, dfail_act,
			dfail_act-dfail_ref);
		fail_test(method, __FILE__, __LINE__, msg);
	}
}

void tod_contract2_impl_221_test::test_ijkl_ijp_klp_a(size_t ni, size_t nj,
	size_t nk, size_t nl, size_t np, double d)
	throw(libtest::test_exception) {

	// c_{ijkl} = c_{ijkl} + d \sum_{p} a_{ijp} b_{klp}

	index<3> ia1, ia2; ia2[0]=ni-1; ia2[1]=nj-1; ia2[2]=np-1;
	index<3> ib1, ib2; ib2[0]=nk-1; ib2[1]=nl-1; ib2[2]=np-1;
	index<4> ic1, ic2; ic2[0]=ni-1; ic2[1]=nj-1; ic2[2]=nk-1; ic2[3]=nl-1;
	index_range<3> ira(ia1,ia2); dimensions<3> dima(ira);
	index_range<3> irb(ib1,ib2); dimensions<3> dimb(irb);
	index_range<4> irc(ic1,ic2); dimensions<4> dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<3,double,allocator> ta(dima); tensor_ctrl<3,double> tca(ta);
	tensor<3,double,allocator> tb(dimb); tensor_ctrl<3,double> tcb(tb);
	tensor<4,double,allocator> tc(dimc); tensor_ctrl<4,double> tcc(tc);
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

	index<3> ia, ib; index<4> ic;
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
		dtc[dimc.abs_index(ic)] += d*cij;
	}
	}
	}
	}

	// Invoke the contraction routine

	permutation<3> perma, permb; permutation<4> permc;
	const double *dcta = tca.req_const_dataptr();
	const double *dctb = tcb.req_const_dataptr();
	double *dctc = tcc.req_dataptr();
	tod_contract2_impl<2,2,1>::contract(dctc, dimc, permc, dcta, dima,
		perma, dctb, dimb, permb, d);
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
		snprintf(method, 1024, "tod_contract2_impl_221_test::"
			"test_ijkl_ijp_klp_a(%lu,%lu,%lu,%lu,%lu,%lg)",
			ni, nj, nk, nl, np, d);
		snprintf(msg, 1024, "contract() result does not match "
			"reference at element %lu: %lg(ref) vs %lg(act), "
			"%lg(diff)", ielem, dfail_ref, dfail_act,
			dfail_act-dfail_ref);
		fail_test(method, __FILE__, __LINE__, msg);
	}
}

} // namespace libtensor

