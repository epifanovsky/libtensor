#include <cmath>
#include <cstdlib>
#include <ctime>
#include "contract2_2_3i_test.h"
#include "tensor.h"
#include "tensor_ctrl.h"

namespace libtensor {

using libvmm::std_allocator;

void contract2_2_3i_test::perform() throw(libtest::test_exception) {
	test_ij_klm_klim_kljm(3, 4, 5, 6, 7);
	test_ij_klm_klim_kljm(3, 3, 3, 3, 3);
}

void contract2_2_3i_test::test_ij_klm_klim_kljm(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t nm) throw(libtest::test_exception) {
	index ia1(4), ia2(4), ib1(4), ib2(4), ic1(2), ic2(2);
	ia2[0]=nk-1; ia2[1]=nl-1; ia2[2]=ni-1; ia2[3]=nm-1;
	ib2[0]=nk-1; ib2[1]=nl-1; ib2[2]=nj-1; ib2[3]=nm-1;
	ic2[0]=ni-1; ic2[1]=nj-1;
	index_range ira(ia1,ia2), irb(ib1,ib2), irc(ic1,ic2);
	dimensions dima(ira), dimb(irb), dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();
	tensor< double,std_allocator<double> > ta(dima), tb(dimb), tc(dimc);
	double *dta = new double[sza];
	double *dtb = new double[szb];
	double *dtc = new double[szc];

	srand48(time(NULL));
	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();

	index ia(4), ib(4), ic(2);
	for(size_t i=0; i<dimc[0]; i++) {
	for(size_t j=0; j<dimc[1]; j++) {
		ic[0]=i; ic[1]=j;
		double cij = 0.0;
		for(size_t k=0; k<dima[0]; k++) {
		for(size_t l=0; l<dima[1]; l++) {
		for(size_t m=0; m<dima[3]; m++) {
			ia[0]=k; ia[1]=l; ia[2]=i; ia[3]=m;
			ib[0]=k; ib[1]=l; ib[2]=j; ib[3]=m;
			cij += dta[dima.abs_index(ia)]*dtb[dimb.abs_index(ib)];
		}
		}
		}
		dtc[dimc.abs_index(ic)] = cij;
	}
	}

	tensor_ctrl<double> tca(ta), tcb(tb), tcc(tc);
	permutation perma(4), permb(4), permc(2);
	perma.permute(0,2).permute(1,2); // klim -> iklm
	permb.permute(0,2).permute(1,2); // kljm -> jklm
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
	contract2_2_3i::contract(dctc, dimc, permc, dcta, dima, perma,
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
		char method[1024], msg[1024];
		snprintf(method, 1024, "contract2_2_3i_test::"
			"test_ij_klm_klim_kljm(%lu,%lu,%lu,%lu,%lu)",
			ni, nj, nk, nl, nm);
		snprintf(msg, 1024, "contract() result does not match "
			"reference at element %lu: %lg(ref) vs %lg(act), "
			"%lg(diff)", ielem, dfail_ref, dfail_act,
			dfail_act-dfail_ref);
		fail_test(method, __FILE__, __LINE__, msg);
	}
}

} // namespace libtensor

