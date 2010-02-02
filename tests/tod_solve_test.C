#include <iostream>
#include <cstdlib>
#include <cmath>
#include <libvmm/std_allocator.h>
#include <libtensor/core/tensor.h>
#include <libtensor/tod/tod_solve.h>
#include "tod_solve_test.h"


namespace libtensor {

using libvmm::std_allocator;

//typedef tensor<4, double, libvmm::std_allocator<double> > tensor4_d;

void tod_solve_test::perform() throw(libtest::test_exception) {
	//ia - matrix,
		//ib,ic - vectors

		int n = 4; //matrix order

		index<2> ia1, ia2; index<1> ib1, ib2, ic1, ic2;
		ia2[0]=n-1; ia2[1]=n-1;
		ib2[0]=n-1;
		ic2[0]=n-1;

		index_range<2> ira(ia1,ia2); index_range<1> irb(ib1,ib2), irc(ic1,ic2);
		dimensions<2> dima(ira); dimensions<1> dimb(irb), dimc(irc);
		size_t sza = dima.get_size(), szb = dimb.get_size(),
			szc = dimc.get_size();
		tensor< 2,double,std_allocator<double> > ta(dima);
		tensor< 1,double,std_allocator<double> > tb(dimb), tc(dimc);

		double *dta = new double[sza];
		double *dtb = new double[szb];
		double *dtc = new double[szc];
		double *answ = new double[szc]; //answ

		for(size_t i=0; i<sza; i++) dta[i]=0;
		for(size_t i=0; i<szb; i++) dtb[i]=0;
		for(size_t i=0; i<szc; i++) answ[i]=0;

		dta[0] = 2; dta[1] = 1;dta[4] = 1; dta[5] = 2;
		dta[10] = 3; dta[11] = 2;dta[14] = 2; dta[15] = 4;
		dtb[0] = 1; dtb[1] = 2;dtb[2] = 3; dtb[3] = 4;
		answ[0] = 0;answ[1] = 1;answ[2] = 0.5;answ[3] = 0.75; //correct answer
//		answ[0] = 0;answ[1] = 1;answ[2] = 0.5;answ[3] = 0.756; //wrong answer

//		dta[4] = 1; dta[5] = 2;

		tensor_ctrl<2,double> tca(ta);
		tensor_ctrl<1,double> tcb(tb), tcc(tc);
		double *ptr = tca.req_dataptr(); //??????
		for(size_t i=0; i<sza; i++) ptr[i]=dta[i]; //copy dta to ptr
		tca.ret_dataptr(ptr);
		ptr = tcb.req_dataptr();
		for(size_t i=0; i<szb; i++) ptr[i]=dtb[i]; //copy dtb to ptr
		tcb.ret_dataptr(ptr);
		ptr=NULL;
//


		tod_solve diag(ta,tb);
		diag.perform(tc);
//		tca.ret_dataptr(dcta);
//		dcta = NULL;
//		tcb.ret_dataptr(dctb);
//		dctb = NULL;
//		cout << "resulting vector: \n";
		tensor_ctrl<1,double> tcc1(tc);
		const double *dctc = tcc1.req_const_dataptr();
		ptr = tca.req_dataptr(); //??????
		bool ok = true;
		for(size_t i=0; i<szc; i++) {
			if( fabs(dctc[i]-answ[i])>fabs(answ[i])*5e-15 ) {
				ok=false;
			}
		}

		if(!ok) {
			char method[1024], msg[1024];
			snprintf(method, 1024, "tod_solve_test::"
				"perform()");
			snprintf(msg, 1024, "perform() result does not match ");
			fail_test(method, __FILE__, __LINE__, msg);
		}

		//print initial vector
//		printf("initial vector: \n");
//		for(size_t i=0; i<sza; i++) {
//			printf( " %f", ptr[i])  ;
//
//		}
		tca.ret_dataptr(ptr);

		//print result vector
//		printf("resulting vector: \n");
//		for(size_t i=0; i<szc; i++) {
//			printf( " %f", dctc[i])  ;
//
//		}
		tcc1.ret_dataptr(dctc); dctc = NULL;
		delete [] dtb; delete [] dta;

}

} // namespace libtensor

