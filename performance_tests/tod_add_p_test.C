#define LIBTENSOR_TIMINGS

#include <libtensor.h>
#include "../timer.h"
#include "tod_add_p_test.h"

namespace libtensor {

const char* 
tod_add_p1_test::k_clazz="tod_add_p1";

void tod_add_p1_test::perform() throw(libtest::test_exception) {
	for ( unsigned int i=0; i<10; i++ ) {
		do_calculate();
	} 
}

void tod_add_p1_test::do_calculate() throw(libtest::test_exception) {
	size_t p=32, q=64, r=52, s=128;
	index<4> i1, i2;
	i2[0]=p; i2[1]=q; i2[2]=r; i2[3]=s;
	index_range<4> ir(i1,i2);
	dimensions<4> dim(ir); 
	tensor<4, double, libvmm::std_allocator<double> > ta(dim), tb(dim);
	tensor_ctrl<4,double> tca(ta), tcb(tb); 
	
	double *ptra=tca.req_dataptr();
	double *ptrb=tcb.req_dataptr();
	for ( size_t i=0; i<dim.get_size(); i++ ) ptra[i]=drand48();
	for ( size_t i=0; i<dim.get_size(); i++ ) ptrb[i]=drand48();
	tca.ret_dataptr(ptra);
	tcb.ret_dataptr(ptrb);
	
	// start tod_add calculation
	permutation<4> perm;
	tod_add<4> add(perm);
	add.add_op(tb,perm,2.0);
	add.prefetch();
	add.perform(ta,1.0);
	
	start_timer(); 
	ptra=tca.req_dataptr();
	const double *cptrb=tcb.req_const_dataptr();
	for ( size_t i=0; i<dim.get_size(); i++ ) {
		ptra[i]+=cptrb[i]*2.0;
	}
	tca.ret_dataptr(ptra);
	tcb.ret_dataptr(cptrb);
	stop_timer();
}

const char* 
tod_add_p2_test::k_clazz="tod_add_p2";

void tod_add_p2_test::perform() throw(libtest::test_exception) {
	for ( unsigned int i=0; i<10; i++ ) {
		do_calculate();
	}
} 
void tod_add_p2_test::do_calculate() throw(libtest::test_exception) {
	size_t p=32, q=64, r=52, s=128;
	index<4> i1, i2;
	i2[0]=p; i2[1]=q; i2[2]=r; i2[3]=s;
	index_range<4> ir(i1,i2);
	dimensions<4> dima(ir), dimb(ir); 
	permutation<4> perma, permb;
	permb.permute(0,3);
	permb.permute(1,2);
	dimb.permute(permb);
	
	tensor<4, double, libvmm::std_allocator<double> > ta(dima), tb(dimb);
	tensor_ctrl<4,double> tca(ta), tcb(tb); 
	
	double *ptra=tca.req_dataptr();
	double *ptrb=tcb.req_dataptr();
	for ( size_t i=0; i<dima.get_size(); i++ ) ptra[i]=drand48();
	for ( size_t i=0; i<dimb.get_size(); i++ ) ptrb[i]=drand48();
	tca.ret_dataptr(ptra);
	tcb.ret_dataptr(ptrb);
	
	// start tod_add calculation
	tod_add<4> add(perma);
	add.add_op(tb,permb,2.0);
	add.prefetch();
	add.perform(ta,1.0);
	
	start_timer(); 
	ptra=tca.req_dataptr();
	const double *cptrb=tcb.req_const_dataptr();
	size_t cnta=0, cntb=0;
	for ( size_t i=0; i<dima.get_dim(0); i++ ) { 
		for ( size_t j=0; j<dima.get_dim(1); j++ ) { 
			for ( size_t k=0; k<dima.get_dim(2); k++ ) { 
				for ( size_t l=0; l<dima.get_dim(3); l++ ) {
					ptra[cnta]+=cptrb[cntb]*2.0;
					cnta++;
					cntb+=dimb.get_increment(0);
				}
				cntb-=(dimb.get_size()-dimb.get_increment(1));
			}
			cntb-=(dimb.get_increment(0)-dimb.get_increment(2));
		}
		cntb-=(dimb.get_increment(1)-dimb.get_increment(3));
	}
			
	tca.ret_dataptr(ptra); 
	tcb.ret_dataptr(cptrb);
	
	stop_timer();
}

const char* 
tod_add_p3_test::k_clazz="tod_add_p3";

void tod_add_p3_test::perform() throw(libtest::test_exception) {
	for ( unsigned int i=0; i<10; i++ ) {
		do_calculate();
	}
} 
void tod_add_p3_test::do_calculate() throw(libtest::test_exception) {
	size_t p=32, q=64, r=52, s=128;
	index<4> i1, i2;
	i2[0]=p; i2[1]=q; i2[2]=r; i2[3]=s;
	index_range<4> ir(i1,i2);
	dimensions<4> dima(ir), dimb(ir); 
	permutation<4> perma, permb;
	permb.permute(0,3);
	dimb.permute(permb);
	
	tensor<4, double, libvmm::std_allocator<double> > ta(dima), tb(dimb);
	tensor_ctrl<4,double> tca(ta), tcb(tb); 
	
	double *ptra=tca.req_dataptr();
	double *ptrb=tcb.req_dataptr();
	for ( size_t i=0; i<dima.get_size(); i++ ) ptra[i]=drand48();
	for ( size_t i=0; i<dimb.get_size(); i++ ) ptrb[i]=drand48();
	tca.ret_dataptr(ptra);
	tcb.ret_dataptr(ptrb);
	
	// start tod_add calculation
	tod_add<4> add(perma);
	add.add_op(tb,permb,2.0);
	add.prefetch();
	add.perform(ta,1.0);
	
	start_timer(); 
	
	ptra=tca.req_dataptr();
	const double *cptrb=tcb.req_const_dataptr();
	size_t cnta=0, cntb=0; 
	for ( size_t i=0; i<dima.get_dim(0); i++ ) { 
		for ( size_t j=0; j<dima.get_dim(1); j++ ) { 
			for ( size_t k=0; k<dima.get_dim(2); k++ ) { 
				for ( size_t l=0; l<dima.get_dim(3); l++ ) {				
					ptra[cnta]+=cptrb[cntb]*2.0;
					cntb+=dimb.get_increment(0);
				}	
				cntb-=(dimb.get_size()-dimb.get_increment(2));				
			}
		} 
		cntb-=(dimb.get_increment(1)-dimb.get_increment(3));
	}
	tca.ret_dataptr(ptra);
	tcb.ret_dataptr(cptrb);

	stop_timer();
	
}

} // namespace libtensor

