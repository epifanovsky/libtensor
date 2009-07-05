#include <cmath>
#include <cstdlib>
#include <libtensor.h>
#include "tod_add_test.h"

namespace libtensor {

typedef tensor<4, double, libvmm::std_allocator<double> > tensor4_d;
typedef tensor_ctrl<4, double> tensor_ctrl4;

void tod_add_test::perform() throw(libtest::test_exception) {
	test_exc();

	test_add_to_self_pqrs(2,3,4,5);
	test_add_two_pqrs_pqrs(2,3,4,5);
	test_add_two_pqrs_qprs(2,3,4,5);
	test_add_two_pqrs_prsq(2,3,4,5);
	test_add_two_pqrs_qpsr(2,3,4,5);
	test_add_mult(3,2,5,4);

}

void tod_add_test::test_exc() throw(libtest::test_exception) {
	index<4> i1, i2;
	i2[0]=2; i2[1]=3; i2[2]=5; i2[3]=4;
	index_range<4> ir(i1, i2);
	dimensions<4> dim(ir);
	permutation<4> p1, p2;
	p1.permute(0,1);

	tensor4_d t1(dim), t2(dim);
	tod_add<4> add(p1);

	bool ok=false;
	try {
		p1.permute(2,3);
		add.add_op(t1,p1,0.5);
		add.add_op(t2,p2,1.0);
	}
	catch(exception e) {
		ok=true;
	}

	if(!ok) {
		fail_test("tod_add_test::test_exc()", __FILE__, __LINE__,
			"Expected an exception due to heterogeneous operands");
	}

	ok=false;
	try {
		add.add_op(t2,p2,1.0);
		add.prefetch();
		add.perform(t1);
	}
	catch(exception e) {
		ok=true;
	}

	if(!ok) {
		fail_test("tod_add_test::test_exc()", __FILE__, __LINE__,
			"Expected an exception due to heterogeneous result tensor");
	}

}

void tod_add_test::test_add_to_self_pqrs( size_t p, size_t q, size_t r, size_t s )
	throw(libtest::test_exception)
{
	index<4> i1, i2;
	i2[0]=p; i2[1]=q; i2[2]=r; i2[3]=s;
	index_range<4> ir(i1, i2);
	dimensions<4> dim(ir);
	tensor4_d t(dim), t1(dim);

	tensor_ctrl4 ctrl1(t1);

	double *ptr1=ctrl1.req_dataptr();
	for ( size_t i=0; i<dim.get_size(); i++ ) ptr1[i]=drand48();
	ctrl1.ret_dataptr(ptr1);

	permutation<4> perm;
	tod_add<4> add(perm);
	add.add_op(t1,perm,2.0);
	add.add_op(t1,perm,0.5);
	add.prefetch();
	add.perform(t);

	tensor_ctrl4 ctrl(t);
	const double* cptr=ctrl.req_const_dataptr();
	const double* cptr1=ctrl1.req_const_dataptr();
	for ( size_t i=0; i<dim.get_size(); i++ ) {
		if ( fabs(cptr[i]-(2.0*cptr1[i]+0.5*cptr1[i])) > 1e-14 ) {
			fail_test("tod_add_test::test_add_to_self_pqrs()", __FILE__, __LINE__,
			"tod_add yielded the wrong result");
		}
	}
	ctrl.ret_dataptr(cptr);
	ctrl1.ret_dataptr(cptr1);
}

void tod_add_test::test_add_two_pqrs_pqrs( size_t p, size_t q, size_t r, size_t s )
	throw(libtest::test_exception)
{
	index<4> i1, i2;
	i2[0]=p; i2[1]=q; i2[2]=r; i2[3]=s;
	index_range<4> ir(i1, i2);
	dimensions<4> dim(ir);
	tensor4_d t1(dim), t2(dim), t1_ref(dim);

	tensor_ctrl4 ctrl1(t1), ctrl2(t2), ctrl1_ref(t1_ref);

	double *ptr1=ctrl1.req_dataptr();
	double *ptr1_ref=ctrl1_ref.req_dataptr();
	double *ptr2=ctrl2.req_dataptr();
	for ( size_t i=0; i<dim.get_size(); i++ ) ptr1[i]=ptr1_ref[i]=drand48();
	for ( size_t i=0; i<dim.get_size(); i++ ) ptr2[i]=drand48();
	ctrl1.ret_dataptr(ptr1);
	ctrl1_ref.ret_dataptr(ptr1_ref);
	ctrl2.ret_dataptr(ptr2);

	permutation<4> perm;
	tod_add<4> add(perm);
	add.add_op(t2,perm,2.0);
	add.prefetch();
	add.perform(t1,1.0);

	const double* cptr1=ctrl1.req_const_dataptr();
	const double* cptr1_ref=ctrl1_ref.req_const_dataptr();
	const double* cptr2=ctrl2.req_const_dataptr();
	for ( size_t i=0; i<dim.get_size(); i++ ) {
		if ( fabs(cptr1[i]-(cptr1_ref[i]+2.0*cptr2[i])) > 1e-14 ) {
			fail_test("tod_add_test::test_add_two_pqrs_pqrs()", __FILE__, __LINE__,
			"tod_add yielded the wrong result");
		}
	}
	ctrl1.ret_dataptr(cptr1);
	ctrl1_ref.ret_dataptr(cptr1_ref);
	ctrl2.ret_dataptr(cptr2);
}

void tod_add_test::test_add_two_pqrs_qprs( size_t p, size_t q, size_t r, size_t s )
	throw(libtest::test_exception)
{
	index<4> i1, i2;
	i2[0]=p; i2[1]=q; i2[2]=r; i2[3]=s;
	index_range<4> ir(i1, i2);
	dimensions<4> dim1(ir), dim2(ir);
	permutation<4> p1, p2;
	p2.permute(0,1);
	dim2.permute(p2);

	tensor4_d t1(dim1), t2(dim2), t1_ref(dim1);

	tensor_ctrl4 ctrl1(t1), ctrl2(t2), ctrl1_ref(t1_ref);

	double *ptr1=ctrl1.req_dataptr();
	double *ptr1_ref=ctrl1_ref.req_dataptr();
	double *ptr2=ctrl2.req_dataptr();

	for (size_t i=0; i<dim1.get_size(); i++) ptr1[i]=ptr1_ref[i]=drand48();
	for (size_t i=0; i<dim2.get_size(); i++) ptr2[i]=drand48();

	ctrl1.ret_dataptr(ptr1);
	ctrl1_ref.ret_dataptr(ptr1_ref);
	ctrl2.ret_dataptr(ptr2);

	tod_add<4> add(p1);
	add.add_op(t2,p2,0.1);
	add.prefetch();
	add.perform(t1,1.0);

	const double* cptr1=ctrl1.req_const_dataptr();
	const double* cptr1_ref=ctrl1_ref.req_const_dataptr();
	const double* cptr2=ctrl2.req_const_dataptr();

	size_t cnt=0;
	for ( size_t i=0; i<dim1[0]; i++ ) 
	for ( size_t j=0; j<dim1[1]; j++ )
	for ( size_t k=0; k<dim1[2]; k++ ) 
	for ( size_t l=0; l<dim1[3]; l++ ) {
		i1[0]=j; i1[1]=i; i1[2]=k; i1[3]=l;
		if ( fabs(cptr1[cnt]-(cptr1_ref[cnt]+0.1*cptr2[dim2.abs_index(i1)])) > 1e-14 ) {
			fail_test("tod_add_test::test_add_two_pqrs_qprs()", __FILE__, __LINE__,
			"tod_add yielded the wrong result");
		}
		cnt++;
	}
	ctrl1.ret_dataptr(cptr1);
	ctrl1_ref.ret_dataptr(cptr1_ref);
	ctrl2.ret_dataptr(cptr2);
}

void tod_add_test::test_add_two_pqrs_prsq( size_t p, size_t q, size_t r, size_t s )
	throw(libtest::test_exception)
{
	index<4> i1, i2;
	i2[0]=p; i2[1]=q; i2[2]=r; i2[3]=s;
	index_range<4> ir(i1, i2);
	dimensions<4> dim1(ir), dim2(ir);
	permutation<4> p1, p2;
	p2.permute(1,2);
	p2.permute(2,3);
	dim2.permute(p2);
	p2.invert();

	tensor4_d t1(dim1), t2(dim2), t1_ref(dim1);

	tensor_ctrl4 ctrl1(t1), ctrl2(t2), ctrl1_ref(t1_ref);

	double *ptr1=ctrl1.req_dataptr();
	double *ptr1_ref=ctrl1_ref.req_dataptr();
	double *ptr2=ctrl2.req_dataptr();

	for ( size_t i=0; i<dim1.get_size(); i++ ) ptr1[i]=ptr1_ref[i]=drand48();
	for ( size_t i=0; i<dim1.get_size(); i++ ) ptr2[i]=drand48();
	ctrl1.ret_dataptr(ptr1);
	ctrl1_ref.ret_dataptr(ptr1_ref);
	ctrl2.ret_dataptr(ptr2);

	tod_add<4> add(p1);
	add.add_op(t2,p2,0.1);
	add.prefetch();
	add.perform(t1,1.0);

	const double* cptr1=ctrl1.req_const_dataptr();
	const double* cptr1_ref=ctrl1_ref.req_const_dataptr();
	const double* cptr2=ctrl2.req_const_dataptr();
	size_t cnt=0;
	for ( size_t i=0; i<dim1[0]; i++ ) for ( size_t j=0; j<dim1[1]; j++ )
	for ( size_t k=0; k<dim1[2]; k++ ) for ( size_t l=0; l<dim1[3]; l++ ) {
		i1[0]=i; i1[1]=k; i1[2]=l; i1[3]=j;
		if ( fabs(cptr1[cnt]-(cptr1_ref[cnt]+0.1*cptr2[dim2.abs_index(i1)])) > 1e-14 ) {
			fail_test("tod_add_test::test_add_two_pqrs_prsq()", __FILE__, __LINE__,
			"tod_add yielded the wrong result");
		}
		cnt++;
	}
	ctrl1.ret_dataptr(cptr1);
	ctrl2.ret_dataptr(cptr2);
	ctrl1_ref.ret_dataptr(cptr1_ref);
}

void tod_add_test::test_add_two_pqrs_qpsr( size_t p, size_t q, size_t r, size_t s )
	throw(libtest::test_exception)
{
	index<4> i1, i2;
	i2[0]=p; i2[1]=q; i2[2]=r; i2[3]=s;
	index_range<4> ir(i1, i2);
	dimensions<4> dim1(ir), dim2(ir);
	permutation<4> p1, p2;
	p2.permute(0,1);
	p2.permute(2,3); 
	dim2.permute(p2);

	tensor4_d t1(dim1), t2(dim2), t1_ref(dim1);

	tensor_ctrl4 ctrl1(t1), ctrl2(t2), ctrl1_ref(t1_ref);

	double *ptr1=ctrl1.req_dataptr();
	double *ptr2=ctrl2.req_dataptr();
	double *ptr1_ref=ctrl1_ref.req_dataptr();

	for (size_t i=0; i<dim1.get_size(); i++) ptr1[i]=ptr1_ref[i]=drand48();
	for (size_t i=0; i<dim2.get_size(); i++) ptr2[i]=drand48();

	ctrl1.ret_dataptr(ptr1);
	ctrl1_ref.ret_dataptr(ptr1_ref);
	ctrl2.ret_dataptr(ptr2);
	
	tod_add<4> add(p1);
	add.add_op(t2,p2,0.1);
	add.prefetch();
	add.perform(t1,1.0);

	const double *cptr1=ctrl1.req_const_dataptr();
	const double *cptr2=ctrl2.req_const_dataptr();
	const double *cptr1_ref=ctrl1_ref.req_const_dataptr();
	
	size_t cnt=0;
	for ( size_t i=0; i<dim1[0]; i++ ) 
	for ( size_t j=0; j<dim1[1]; j++ )
	for ( size_t k=0; k<dim1[2]; k++ ) 
	for ( size_t l=0; l<dim1[3]; l++ ) {
		i1[0]=j; i1[1]=i; i1[2]=l; i1[3]=k;
		if ( fabs(cptr1[cnt]-(cptr1_ref[cnt]+0.1*cptr2[dim2.abs_index(i1)]))>1e-14 ) {
			fail_test("tod_add_test::test_add_two_pqrs_qpsr()", __FILE__, __LINE__,
			"tod_add yielded the wrong result");
		}
		cnt++;
	}
	ctrl1.ret_dataptr(cptr1);
	ctrl2.ret_dataptr(cptr2);
	ctrl1_ref.ret_dataptr(cptr1_ref);
}

void tod_add_test::test_add_mult( size_t p, size_t q, size_t r, size_t s )
	throw(libtest::test_exception)
{
	index<4> i1, i2;
	i2[0]=p; i2[1]=q; i2[2]=r; i2[3]=s;
	index_range<4> ir(i1, i2);
	dimensions<4> dim(ir), dim3(ir);
	permutation<4> perm, p3;
	p3.permute(0,1);
	dim3.permute(p3);
	tensor4_d t1(dim), t2(dim), t3(dim3);

	tensor_ctrl4 ctrl1(t1), ctrl2(t2), ctrl3(t3);

	double *ptr1=ctrl1.req_dataptr();
	double *ptr2=ctrl2.req_dataptr();
	double *ptr3=ctrl3.req_dataptr();
	size_t cnt=0;
	while ( cnt < dim.get_size() ) {
		ptr1[cnt]=double(cnt);
		ptr2[cnt]=0.5*cnt;
		ptr3[cnt]=0.25*cnt;
		cnt++;
	}

	ctrl1.ret_dataptr(ptr1);
	ctrl2.ret_dataptr(ptr2);
	ctrl3.ret_dataptr(ptr3);

	tod_add<4> add(perm);
	add.add_op(t2,perm,1.0);
	add.add_op(t3,p3,-4.0);
	add.prefetch();
	add.perform(t1,0.5);

	const double *ptr=ctrl1.req_const_dataptr();
	cnt=0;
	for ( size_t i=0; i<dim[0]; i++ ) for ( size_t j=0; j<dim[1]; j++ )
	for ( size_t k=0; k<dim[2]; k++ ) for ( size_t l=0; l<dim[3]; l++ ) {
		if ( fabs(ptr[cnt]-(cnt+.25*cnt-0.5*(((j*dim3[1]+i)*dim3[2]+k)*dim3[3]+l))) > 1e-14 ) {
			fail_test("tod_add_test::test_add_mult()", __FILE__, __LINE__,
			"tod_add yielded the wrong result");
		}
		cnt++;
	}
	ctrl1.ret_dataptr(ptr);
}

typedef tensor<2, double, libvmm::std_allocator<double> > tensor2_d;
typedef tensor_ctrl<2, double> tensor_ctrl2;


void tod_add_test::test_add_two_pq_qp( size_t p, size_t q )
	throw(libtest::test_exception)
{
	index<2> i1, i2;
	i2[0]=p; i2[1]=q;
	index_range<2> ir(i1, i2);
	dimensions<2> dim(ir), dim3(ir);
	permutation<2> perm, p3;
	p3.permute(0,1);
	dim3.permute(p3);
	tensor2_d t1(dim), t2(dim), t3(dim3);

	tensor_ctrl2 ctrl1(t1), ctrl2(t2), ctrl3(t3);

	double *ptr1=ctrl1.req_dataptr();
	double *ptr2=ctrl2.req_dataptr();
	double *ptr3=ctrl3.req_dataptr();
	size_t cnt=0;
	while ( cnt < dim.get_size() ) {
		ptr1[cnt]=1.;
		ptr2[cnt]=0.5*cnt*(-1.*(cnt%2));
		ptr3[cnt]=0.5*cnt;
		cnt++;
	}

	ctrl1.ret_dataptr(ptr1);
	ctrl2.ret_dataptr(ptr2);
	ctrl3.ret_dataptr(ptr3);

	tod_add<2> add(perm);
	add.add_op(t2,perm,2.0);
	add.add_op(t3,p3,-1.0);
	add.prefetch();
	add.perform(t1,0.5);

	const double *ptr=ctrl1.req_const_dataptr();
	cnt=0;
	for ( size_t i=0; i<dim[0]; i++ ) for ( size_t j=0; j<dim[1]; j++ ) {
		if ( fabs(ptr[cnt]-(cnt+.5*cnt*(-1.*(cnt%2))-.25*(j*dim3[1]+i))) > 1e-14 ) {
			fail_test("tod_add_test::test_add_mult()", __FILE__, __LINE__,
			"tod_add yielded the wrong result");
		}
		cnt++;
	}
	ctrl1.ret_dataptr(ptr);
}


} // namespace libtensor

