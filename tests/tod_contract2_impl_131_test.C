#include <cmath>
#include <cstdlib>
#include <ctime>
#include "tod_contract2_impl_131_test.h"
#include "tensor.h"
#include "tensor_ctrl.h"

namespace libtensor {

typedef libvmm::std_allocator<double> allocator;

void tod_contract2_impl_131_test::perform() throw(libtest::test_exception) {
	srand48(time(NULL));

	test_jikl_pi_jpkl(1, 4, 5, 6, 2);
	test_jikl_pi_jpkl(3, 4, 5, 6, 7);
	test_jikl_pi_jpkl_a(3, 4, 5, 6, 7, 0.0);
	test_jikl_pi_jpkl_a(3, 4, 5, 6, 7, -2.0);

}

void tod_contract2_impl_131_test::test_jikl_pi_jpkl(size_t ni, size_t nj,
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

	tensor<2,double,allocator> ta(dima); tensor_ctrl<2,double> tca(ta);
	tensor<4,double,allocator> tb(dimb); tensor_ctrl<4,double> tcb(tb);
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

	index<2> ia; index<4> ib, ic;
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
		dtc[dimc.abs_index(ic)] = cjikl;
	}
	}
	}
	}

	// Invoke the contraction routine

	permutation<2> perma; permutation<4> permb, permc;
	perma.permute(0,1); // pi -> ip
	permb.permute(1,3).permute(1,2); // jpkl -> jlkp -> jklp
	permc.permute(0,1); // ijkl -> jikl
	const double *dcta = tca.req_const_dataptr();
	const double *dctb = tcb.req_const_dataptr();
	double *dctc = tcc.req_dataptr();
	tod_contract2_impl<1,3,1>::contract(dctc, dimc, permc, dcta, dima,
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
		snprintf(method, 1024, "tod_contract2_impl_131_test::"
			"test_jikl_pi_jpkl(%lu,%lu,%lu,%lu,%lu)",
			ni, nj, nk, nl, np);
		snprintf(msg, 1024, "contract() result does not match "
			"reference at element %lu: %lg(ref) vs %lg(act), "
			"%lg(diff)", ielem, dfail_ref, dfail_act,
			dfail_act-dfail_ref);
		fail_test(method, __FILE__, __LINE__, msg);
	}
}

void tod_contract2_impl_131_test::test_jikl_pi_jpkl_a(size_t ni, size_t nj,
	size_t nk, size_t nl, size_t np, double d)
	throw(libtest::test_exception) {

	// c_{jikl} = c_{jikl} + d \sum_p a_{pi} b_{jpkl}

	index<2> ia1, ia2; ia2[0]=np-1; ia2[1]=ni-1;
	index<4> ib1, ib2; ib2[0]=nj-1; ib2[1]=np-1; ib2[2]=nk-1; ib2[3]=nl-1;
	index<4> ic1, ic2; ic2[0]=nj-1; ic2[1]=ni-1; ic2[2]=nk-1; ic2[3]=nl-1;

	index_range<2> ira(ia1,ia2); dimensions<2> dima(ira);
	index_range<4> irb(ib1,ib2); dimensions<4> dimb(irb);
	index_range<4> irc(ic1,ic2); dimensions<4> dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();

	tensor<2,double,allocator> ta(dima); tensor_ctrl<2,double> tca(ta);
	tensor<4,double,allocator> tb(dimb); tensor_ctrl<4,double> tcb(tb);
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

	index<2> ia; index<4> ib, ic;
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
		dtc[dimc.abs_index(ic)] += d*cjikl;
	}
	}
	}
	}

	// Invoke the contraction routine

	permutation<2> perma; permutation<4> permb, permc;
	perma.permute(0,1); // pi -> ip
	permb.permute(1,3).permute(1,2); // jpkl -> jlkp -> jklp
	permc.permute(0,1); // ijkl -> jikl
	const double *dcta = tca.req_const_dataptr();
	const double *dctb = tcb.req_const_dataptr();
	double *dctc = tcc.req_dataptr();
	tod_contract2_impl<1,3,1>::contract(dctc, dimc, permc, dcta, dima,
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
		snprintf(method, 1024, "tod_contract2_impl_131_test::"
			"test_jikl_pi_jpkl(%lu,%lu,%lu,%lu,%lu,%lg)",
			ni, nj, nk, nl, np, d);
		snprintf(msg, 1024, "contract() result does not match "
			"reference at element %lu: %lg(ref) vs %lg(act), "
			"%lg(diff)", ielem, dfail_ref, dfail_act,
			dfail_act-dfail_ref);
		fail_test(method, __FILE__, __LINE__, msg);
	}
}

} // namespace libtensor

