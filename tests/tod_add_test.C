#include <cmath>
#include <sstream>
#include <libtensor.h>
#include "compare_ref.h"
#include "tod_add_test.h"

namespace libtensor {

typedef tensor<2, double, libvmm::std_allocator<double> > tensor2_d;
typedef tensor_ctrl<2, double> tensor_ctrl2;
typedef tensor<4, double, libvmm::std_allocator<double> > tensor4_d;
typedef tensor_ctrl<4, double> tensor_ctrl4;

const double tod_add_test::k_thresh = 1e-14;

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
	permutation<4> p1;
	p1.permute(0,1);

	tensor4_d t1(dim), t2(dim);
	tod_add<4> add(t1,p1,0.4);

	bool ok=false;
	try {
		add.add_op(t2,1.0);
	}
	catch(exception& e) {
		ok=true;
	}

	if(!ok) {
		fail_test("tod_add_test::test_exc()", __FILE__, __LINE__,
			"Expected an exception due to heterogeneous operands");
	}

	ok=false;
	try {
		add.prefetch();
		add.perform(t2);
	}
	catch(exception& e) {
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
	tensor4_d tc(dim), ta(dim), tc_ref(dim);

	tensor_ctrl4 ctrla(ta), ctrlc_ref(tc_ref);

	double *ptra=ctrla.req_dataptr();
	for ( size_t i=0; i<dim.get_size(); i++ ) ptra[i]=drand48();
	ctrla.ret_dataptr(ptra);

	double* ptrc_ref=ctrlc_ref.req_dataptr();
	const double* cptra=ctrla.req_const_dataptr();
	double ta_max=0.0;
	for ( size_t i=0; i<dim.get_size(); i++ ) {
		ptrc_ref[i]=2.0*cptra[i]+0.5*cptra[i];
		if ( fabs(cptra[i]) > ta_max ) ta_max=fabs(cptra[i]);
	}
	ctrla.ret_dataptr(cptra);
	ctrlc_ref.ret_dataptr(ptrc_ref);
	
	tod_add<4> add(ta,2.0);
	add.add_op(ta,0.5);
	add.prefetch();
	add.perform(tc);

	std::ostringstream testname;
	testname << "tod_add_test::test_add_to_self_pqrs(";
	testname << p << "," << q <<","<<r<<","<<s<<")";
	compare_ref<4>::compare(testname.str().c_str(),tc,tc_ref,ta_max*k_thresh);
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
	ctrl2.ret_dataptr(ptr2);

	const double* cptr2=ctrl2.req_const_dataptr();
	double t2_max=0.0;
	for ( size_t i=0; i<dim.get_size(); i++ ) {
		ptr1_ref[i]+=2.0*cptr2[i];
		if ( fabs(cptr2[i]) > t2_max ) t2_max=fabs(cptr2[i]);
	}
	ctrl1_ref.ret_dataptr(ptr1_ref);
	ctrl2.ret_dataptr(cptr2);

	tod_add<4> add(t2,2.0);
	add.prefetch();
	add.perform(t1,1.0);

	std::ostringstream testname;
	testname << "tod_add_test::test_add_to_pqrs_pqrs(";
	testname << p <<"," << q << "," << r << "," << s << ")";
	compare_ref<4>::compare(testname.str().c_str(),t1,t1_ref,t2_max*k_thresh);

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
	ctrl2.ret_dataptr(ptr2);

	const double* cptr2=ctrl2.req_const_dataptr();
	size_t cnt=0;
	double t2_max=0.0;
	for ( size_t i=0; i<dim1[0]; i++ ) 
	for ( size_t j=0; j<dim1[1]; j++ )
	for ( size_t k=0; k<dim1[2]; k++ ) 
	for ( size_t l=0; l<dim1[3]; l++ ) {
		i1[0]=j; i1[1]=i; i1[2]=k; i1[3]=l;
		ptr1_ref[cnt]+=0.1*cptr2[dim2.abs_index(i1)];
		if ( fabs(cptr2[dim2.abs_index(i1)]) > t2_max )
			t2_max=fabs(cptr2[dim2.abs_index(i1)]); 
		cnt++;
	}
	ctrl1_ref.ret_dataptr(ptr1_ref);
	ctrl2.ret_dataptr(cptr2);
	
	tod_add<4> add(t2,p2,0.1);
	add.prefetch();
	add.perform(t1,1.0);

	std::ostringstream testname;
	testname << "tod_add_test::test_add_two_pqrs_qprs(";
	testname << p << "," << q << "," << r << "," << s << ")";
	compare_ref<4>::compare(testname.str().c_str(),t1,t1_ref,t2_max*k_thresh);
}

void tod_add_test::test_add_two_pqrs_prsq( size_t p, size_t q, size_t r, size_t s )
	throw(libtest::test_exception)
{
	index<4> i1, i2;
	i2[0]=p; i2[1]=q; i2[2]=r; i2[3]=s;
	index_range<4> ir(i1, i2);
	dimensions<4> dim1(ir), dim2(ir);
	permutation<4> p2;
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
	ctrl2.ret_dataptr(ptr2);

	const double* cptr2=ctrl2.req_const_dataptr();
	double t2_max=0.0;
	size_t cnt=0;
	for ( size_t i=0; i<dim1[0]; i++ ) 
	for ( size_t j=0; j<dim1[1]; j++ )
	for ( size_t k=0; k<dim1[2]; k++ ) 
	for ( size_t l=0; l<dim1[3]; l++ ) {
		i1[0]=i; i1[1]=k; i1[2]=l; i1[3]=j;
		ptr1_ref[cnt]+=0.1*cptr2[dim2.abs_index(i1)];
		if ( fabs(cptr2[dim2.abs_index(i1)]) > t2_max )
			t2_max=fabs(cptr2[dim2.abs_index(i1)]); 
		cnt++;
	}
	ctrl2.ret_dataptr(cptr2);
	ctrl1_ref.ret_dataptr(ptr1_ref);


	tod_add<4> add(t2,p2,0.1);
	add.prefetch();
	add.perform(t1,1.0);

	std::ostringstream testname;
	testname << "tod_add_test::test_add_two_pqrs_prsq(";
	testname << p << "," << q << "," << r << "," << s << ")";
	compare_ref<4>::compare(testname.str().c_str(),t1,t1_ref,t2_max*k_thresh);
}

void tod_add_test::test_add_two_pqrs_qpsr( size_t p, size_t q, size_t r, size_t s )
	throw(libtest::test_exception)
{
	index<4> i1, i2;
	i2[0]=p; i2[1]=q; i2[2]=r; i2[3]=s;
	index_range<4> ir(i1, i2);
	dimensions<4> dim1(ir), dim2(ir);
	permutation<4> p2;
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
	ctrl2.ret_dataptr(ptr2);
	
	const double *cptr2=ctrl2.req_const_dataptr();
	double t2_max=0.0;
	size_t cnt=0;
	for ( size_t i=0; i<dim1[0]; i++ ) 
	for ( size_t j=0; j<dim1[1]; j++ )
	for ( size_t k=0; k<dim1[2]; k++ ) 
	for ( size_t l=0; l<dim1[3]; l++ ) {
		i1[0]=j; i1[1]=i; i1[2]=l; i1[3]=k;
		ptr1_ref[cnt]+=0.1*cptr2[dim2.abs_index(i1)];
		if ( fabs(cptr2[dim2.abs_index(i1)]) > t2_max )
			t2_max=fabs(cptr2[dim2.abs_index(i1)]); 
		cnt++;
	}
	ctrl2.ret_dataptr(cptr2);
	ctrl1_ref.ret_dataptr(ptr1_ref);

	tod_add<4> add(t2,p2,0.1);
	add.prefetch();
	add.perform(t1,1.0);

	std::ostringstream testname;
	testname << "tod_add_test::test_add_two_pqrs_qpsr(";
	testname << p << "," << q << "," << r << "," << s << ")";
	compare_ref<4>::compare(testname.str().c_str(),t1,t1_ref,t2_max*k_thresh);
}

void tod_add_test::test_add_mult( size_t p, size_t q, size_t r, size_t s )
	throw(libtest::test_exception)
{
	index<4> i1, i2;
	i2[0]=p; i2[1]=q; i2[2]=r; i2[3]=s;
	index_range<4> ir(i1, i2);
	dimensions<4> dim(ir), dim3(ir);
	permutation<4> p3;
	p3.permute(0,1);
	dim3.permute(p3);
	tensor4_d t1(dim), t2(dim), t3(dim3), t4(dim), t1_ref(dim);

	tensor_ctrl4 ctrl1(t1), ctrl2(t2), ctrl3(t3), ctrl4(t4), ctrl1_ref(t1_ref);

	double *ptr1=ctrl1.req_dataptr();
	double *ptr2=ctrl2.req_dataptr();
	double *ptr3=ctrl3.req_dataptr();
	double *ptr4=ctrl4.req_dataptr();
	double *ptr1_ref=ctrl1_ref.req_dataptr();
	for (size_t i=0; i<dim.get_size(); i++) ptr1[i]=ptr1_ref[i]=drand48();
	for (size_t i=0; i<dim.get_size(); i++) ptr2[i]=drand48();
	for (size_t i=0; i<dim.get_size(); i++) ptr3[i]=drand48();
	for (size_t i=0; i<dim.get_size(); i++) ptr4[i]=drand48();
	ctrl1.ret_dataptr(ptr1);
	ctrl2.ret_dataptr(ptr2);
	ctrl3.ret_dataptr(ptr3);
	ctrl4.ret_dataptr(ptr4);

	const double *cptr2=ctrl2.req_const_dataptr();
	const double *cptr3=ctrl3.req_const_dataptr();
	const double *cptr4=ctrl4.req_const_dataptr();
	size_t cnt=0;
	double t_max=0.0;
	for ( size_t i=0; i<dim[0]; i++ ) 
	for ( size_t j=0; j<dim[1]; j++ )
	for ( size_t k=0; k<dim[2]; k++ ) 
	for ( size_t l=0; l<dim[3]; l++ ) {
		i1[0]=j; i1[1]=i; i1[2]=k; i1[3]=l;
		ptr1_ref[cnt]+=0.5*(cptr2[cnt]-4.0*cptr3[dim3.abs_index(i1)]+0.2*cptr4[cnt]);
		if ( fabs(ptr1_ref[cnt]) > t_max ) t_max=fabs(ptr1_ref[cnt]); 
		cnt++;
	}
	ctrl1_ref.ret_dataptr(ptr1_ref);
	ctrl2.ret_dataptr(ptr2);
	ctrl3.ret_dataptr(ptr3);
	ctrl4.ret_dataptr(ptr4);
	
	tod_add<4> add(t2,1.0);
	add.add_op(t3,p3,-4.0);
	add.add_op(t4,0.2);
	add.prefetch();
	add.perform(t1,0.5);

	std::ostringstream testname;
	testname << "tod_add_test::test_add_mult(";
	testname << p << "," << q << "," << r << "," << s << ")";
	compare_ref<4>::compare(testname.str().c_str(),t1,t1_ref,t_max*k_thresh);
}



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
	tensor2_d t1(dim), t2(dim), t3(dim3), t1_ref(dim);

	tensor_ctrl2 ctrl1(t1), ctrl2(t2), ctrl3(t3), ctrl1_ref(t1_ref);

	double *ptr1=ctrl1.req_dataptr();
	double *ptr1_ref=ctrl1_ref.req_dataptr();
	double *ptr2=ctrl2.req_dataptr();
	double *ptr3=ctrl3.req_dataptr();
	for (size_t i=0; i<dim.get_size(); i++) ptr1[i]=ptr1_ref[i]=drand48();
	for (size_t i=0; i<dim.get_size(); i++) ptr2[i]=drand48();
	for (size_t i=0; i<dim.get_size(); i++) ptr3[i]=drand48();
	ctrl1.ret_dataptr(ptr1);
	ctrl2.ret_dataptr(ptr2);
	ctrl3.ret_dataptr(ptr3);

	const double *cptr2=ctrl2.req_const_dataptr();
	const double *cptr3=ctrl3.req_const_dataptr();
	size_t cnt=0;
	double t_max=0.0;
	for ( size_t i=0; i<dim[0]; i++ ) 
	for ( size_t j=0; j<dim[1]; j++ ) {
		ptr1_ref[cnt]+=0.5*(2.0*cptr2[cnt]-cptr3[j*dim3[1]+i]);	
		if ( fabs(ptr1_ref[cnt]) > t_max ) t_max=fabs(ptr1_ref[cnt]);
		cnt++;
	} 		
	ctrl1_ref.ret_dataptr(ptr1_ref);
	ctrl2.ret_dataptr(ptr2);
	ctrl3.ret_dataptr(ptr3);

	tod_add<2> add(t2,2.0);
	add.add_op(t3,p3,-1.0);
	add.prefetch();
	add.perform(t1,0.5);

	std::ostringstream testname;
	testname << "tod_add_test::test_add_two_pq_qp(";
	testname << p << "," << q << ")";
	compare_ref<2>::compare(testname.str().c_str(),t1,t1_ref,t_max*k_thresh);
}


} // namespace libtensor

