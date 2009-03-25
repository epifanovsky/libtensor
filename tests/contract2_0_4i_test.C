
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "contract2_0_4i_test.h"
#include "tensor.h"
#include "tensor_ctrl.h"

namespace libtensor {

using libvmm::std_allocator;

void contract2_0_4i_test::perform() throw(libtest::test_exception) {
	srand48(time(NULL));

	test_ijkl_ijkl(3, 4, 5, 6);

	test_ijkl_ijkl(3, 4, 5, 6);
	test_ijkl_ijkl(3, 3, 3, 3);
	test_ijkl_ijkl(3, 1, 3, 1);
	test_ijkl_ijkl(3, 3, 1, 1);
}

void contract2_0_4i_test::test_ijkl_ijkl(size_t ni, size_t nj, size_t nk,
		size_t nl) throw(libtest::test_exception) {

		//set initial and final indexes
		index<4> ia1, ia2, ib1, ib2;
		ia2[0]=ni-1; ia2[1]=nj-1; ia2[2]=nk-1; ia2[3]=nl-1;
		ib2[0]=ni-1; ib2[1]=nj-1; ib2[2]=nk-1; ib2[3]=nl-1;

		//set index ranges and dimensions
		index_range<4> ira(ia1,ia2), irb(ib1,ib2);
		dimensions<4> dima(ira), dimb(irb);
		size_t sza = dima.get_size(), szb = dimb.get_size();
		tensor< 4,double,std_allocator<double> > ta(dima), tb(dimb);
		double *dta = new double[sza];
		double *dtb = new double[szb];

		//fill tensors with random numbers
		for(size_t i=0; i<sza; i++) dta[i]=drand48();
		for(size_t i=0; i<szb; i++) dtb[i]=drand48();

		index<4> ia, ib;
		double dc = 0; //final result
		for(size_t i=0; i<dima[0]; i++) {
			for(size_t j=0; j<dima[1]; j++) {
				for(size_t k=0; k<dima[2]; k++) {
					for(size_t l=0; l<dima[3]; l++) {
						ia[0]=i; ia[1]=j; ia[2]=k; ia[3]=l;
						ib[0]=i; ib[1]=j; ib[2]=k; ib[3]=l;
						dc += dta[dima.abs_index(ia)]*dtb[dimb.abs_index(ib)];
					}
				}
			}
		}

		tensor_ctrl<4,double> tca(ta), tcb(tb);
		permutation<4> perma, permb; permutation<2> permc;
		double *ptr = tca.req_dataptr(); //??????
		for(size_t i=0; i<sza; i++) ptr[i]=dta[i]; //copy dta to ptr
		tca.ret_dataptr(ptr);
		ptr = tcb.req_dataptr();
		for(size_t i=0; i<szb; i++) ptr[i]=dtb[i]; //copy dtb to ptr
		tcb.ret_dataptr(ptr);
		ta.set_immutable(); tb.set_immutable();
		const double *dcta = tca.req_const_dataptr();
		const double *dctb = tcb.req_const_dataptr();
		double dcc = 0;

		contract2_0_4i::contract(&dcc, dcta, dima, perma,
			dctb, dimb, permb);
		tca.ret_dataptr(dcta); dcta = NULL;
		tcb.ret_dataptr(dctb); dctb = NULL;

		bool ok = true;
		if( fabs(dcc-dc)>fabs(dc)*5e-15 ) {
			ok=false;
		}

//		tcc.ret_dataptr(dctc);
		dcc = NULL;

		delete [] dtb; delete [] dta;

		if(!ok) {
			char method[1024], msg[1024];
			snprintf(method, 1024, "contract2_0_4i_test::"
				"test_ijkl_ijkl_ijkl(%lu,%lu,%lu,%lu)",
				ni, nj, nk, nl);
			snprintf(msg, 1024, "contract() result does not match "
								"\n dcc = %e, \n dc = %e", dcc, dc);
			fail_test(method, __FILE__, __LINE__, msg);
		}
	}

} // namespace libtensor
