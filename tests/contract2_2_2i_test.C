#include <cmath>
#include <cstdlib>
#include <ctime>
#include "contract2_2_2i_test.h"
#include "tensor.h"
#include "tensor_ctrl.h"

namespace libtensor {

using libvmm::std_allocator;

void contract2_2_2i_test::perform() throw(libtest::test_exception) {
	test_1();
}

void contract2_2_2i_test::test_1() throw(libtest::test_exception) {
	index<2> ia1, ia2;
	index<4> ib1, ib2;
	index<2> ic1, ic2;
	ia2[0]=2; ia2[1]=2;
	ib2[0]=2; ib2[1]=2; ib2[2]=2; ib2[3]=2;
	ic2[0]=2; ic2[1]=2;
	index_range<2> ira(ia1,ia2), irc(ic1,ic2);
	index_range<4> irb(ib1,ib2);
	dimensions<2> dima(ira), dimc(irc);
	dimensions<4> dimb(irb);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();
	tensor< 2,double,std_allocator<double> > ta(dima), tc(dimc);
	tensor< 4,double,std_allocator<double> > tb(dimb);
	double *dta = new double[sza];
	double *dtb = new double[szb];
	double *dtc = new double[szc];

	srand48(time(NULL));
	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();

	index<2> ia, ic; index<4> ib;
	for(size_t i=0; i<dimc[0]; i++) {
		for(size_t j=0; j<dimc[1]; j++) {
			ic[0]=i; ic[1]=j;
			double cij = 0.0;
			for(size_t k=0; k<dima[0]; k++) {
				for(size_t l=0; l<dima[1]; l++) {
					ia[0]=k; ia[1]=l;
					ib[0]=i; ib[1]=j; ib[2]=k; ib[3]=l;
					cij += dta[dima.abs_index(ia)]*
						dtb[dimb.abs_index(ib)];
				}
			}
			dtc[dimc.abs_index(ic)] = cij;
		}
	}

	tensor_ctrl<2,double> tca(ta), tcc(tc);
	tensor_ctrl<4,double> tcb(tb);
	permutation<2> perma, permc;
	permutation<4> permb;
	double *ptr = tca.req_dataptr();
	for(size_t i=0; i<sza; i++) ptr[i]=dta[i];
	tca.ret_dataptr(ptr);
	ptr = tcb.req_dataptr();
	for(size_t i=0; i<szb; i++) ptr[i]=dtb[i];
	tcb.ret_dataptr(ptr);
	ta.set_immutable(); tb.set_immutable();
	const double *dcta = tca.req_const_dataptr();
	const double *dctb = tcb.req_const_dataptr();
	double *dctc = tcc.req_dataptr();
	contract2_2_2i::contract(dctc, dimc, permc, dcta, dima, perma,
		dctb, dimb, permb);
	tca.ret_dataptr(dcta); dcta = NULL;
	tcb.ret_dataptr(dctb); dctb = NULL;

	bool ok = true;
	size_t ielem;
	double dfail_ref, dfail_act;
	for(ielem=0; ielem<szc; ielem++) {
		if(fabs(dctc[ielem]-dtc[ielem])>1e-15) {
			dfail_ref = dtc[ielem]; dfail_act = dctc[ielem];
			ok=false; break;
		}
	}

	tcc.ret_dataptr(dctc); dctc = NULL;

	delete [] dtc; delete [] dtb; delete [] dta;

	if(!ok) {
		char msg[1024];
		snprintf(msg, 1024, "contract() result does not match "
			"reference at element %lu: %lg(ref) vs %lg(act), "
			"%lg(diff)", ielem, dfail_ref, dfail_act,
			dfail_act-dfail_ref);
		fail_test("contract2_2_2i_test::test_1()", __FILE__, __LINE__,
			msg);
	}
}

} // namespace libtensor

