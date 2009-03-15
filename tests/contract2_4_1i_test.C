#include <cmath>
#include <cstdlib>
#include <ctime>
#include "contract2_4_1i_test.h"
#include "tensor.h"
#include "tensor_ctrl.h"

namespace libtensor {

using libvmm::std_allocator;

void contract2_4_1i_test::perform() throw(libtest::test_exception) {
	srand48(time(NULL));

	test_jikl_m_mi_jmkl(1, 4, 5, 6, 2);
	test_jikl_m_mi_jmkl(3, 4, 5, 6, 7);

}

void contract2_4_1i_test::test_jikl_m_mi_jmkl(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t nm) throw(libtest::test_exception) {

	index ia1(2), ia2(2), ib1(4), ib2(4), ic1(4), ic2(4);
	ia2[0]=nm-1; ia2[1]=ni-1;
	ib2[0]=nj-1; ib2[1]=nm-1; ib2[2]=nk-1; ib2[3]=nl-1;
	ic2[0]=nj-1; ic2[1]=ni-1; ic2[2]=nk-1; ic2[3]=nl-1;

	index_range ira(ia1,ia2), irb(ib1,ib2), irc(ic1,ic2);
	dimensions dima(ira), dimb(irb), dimc(irc);
	size_t sza = dima.get_size(), szb = dimb.get_size(),
		szc = dimc.get_size();
	tensor< double,std_allocator<double> > ta(dima), tb(dimb), tc(dimc);
	double *dta = new double[sza];
	double *dtb = new double[szb];
	double *dtc = new double[szc];

	for(size_t i=0; i<sza; i++) dta[i]=drand48();
	for(size_t i=0; i<szb; i++) dtb[i]=drand48();

	index ia(2), ib(4), ic(4);
	for(size_t j=0; j<dimc[0]; j++) {
	for(size_t i=0; i<dimc[1]; i++) {
	for(size_t k=0; k<dimc[2]; k++) {
	for(size_t l=0; l<dimc[3]; l++) {
		ic[0]=j; ic[1]=i; ic[2]=k; ic[3]=l;
		double cjikl = 0.0;
		for(size_t m=0; m<dima[0]; m++) {
			ia[0]=m; ia[1]=i;
			ib[0]=j; ib[1]=m; ib[2]=k; ib[3]=l;
			cjikl += dta[dima.abs_index(ia)]*
				dtb[dimb.abs_index(ib)];
		}
		dtc[dimc.abs_index(ic)] = cjikl;
	}
	}
	}
	}

	tensor_ctrl<double> tca(ta), tcb(tb), tcc(tc);
	permutation perma(2), permb(4), permc(4);
	perma.permute(0,1); // mi -> im
	permb.permute(1,3).permute(1,2); // jmkl -> jlkm -> jklm
	permc.permute(0,1); // ijkl -> jikl
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
	contract2_4_1i::contract(dctc, dimc, permc, dcta, dima, perma,
		dctb, dimb, permb);
	tca.ret_dataptr(dcta); dcta = NULL;
	tcb.ret_dataptr(dctb); dctb = NULL;

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
		snprintf(method, 1024, "contract2_4_1i_test::"
			"test_jikl_m_mi_jmkl(%lu,%lu,%lu,%lu,%lu)",
			ni, nj, nk, nl, nm);
		snprintf(msg, 1024, "contract() result does not match "
			"reference at element %lu: %lg(ref) vs %lg(act), "
			"%lg(diff)", ielem, dfail_ref, dfail_act,
			dfail_act-dfail_ref);
		fail_test(method, __FILE__, __LINE__, msg);
	}
}

} // namespace libtensor

