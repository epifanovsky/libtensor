#ifndef LIBTENSOR_TOD_ADD_H
#define LIBTENSOR_TOD_ADD_H

#include "defs.h"
#include "exception.h"
#include "tensor_i.h"
#include "tensor_ctrl.h"
#include "tod_additive.h"

namespace libtensor {

/**	\brief Adds two or more tensors

	Tensor addition of n tensors:
	\f[ B = c_B \mathcal{P_B} \left( c_1 \mathcal{P}_1 A_1 + c_2 \mathcal{P}_2 A_2 + \cdots +
		c_n \mathcal{P}_n A_n \right) \f]

	Each operand must have the same dimensions as the result in order
	for the operation to be successful. 

	\ingroup libtensor_tod
**/
template<size_t N>
class tod_add : public tod_additive<N> {
private:
	struct operand {
		tensor_i<N,double> &m_t;
		const double m_c;
		const permutation<N> m_p;
		struct operand* m_next;
		operand(tensor_i<N,double> &t, const permutation<N> &p, double c)
			: m_t(t), m_p(p), m_c(c), m_next(NULL) {}
	};

	struct operand* m_head;	
	struct operand* m_tail; 

	dimensions<N>* m_dim;  //!< dimensions of the output tensor
	permutation<N> m_p; //!< permutation to be applied to all tensors

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the operation

		\param pb Permutation of the resulting %tensor b
	**/
	tod_add(const permutation<N> &pb) throw(exception);

	/**	\brief Virtual destructor
	**/
	virtual ~tod_add();

	//@}
	
	
	/**	\brief Adds an operand
		\param t Tensor.
		\param p Permutation of %tensor elements.
		\param c Coefficient.
	**/
	void add_op(tensor_i<N,double> &t, const permutation<N> &p,
		const double c) throw(exception);

	//!	\name Implementation of direct_tensor_operation<T>
	//@{
	virtual void prefetch() throw(exception);
	//@}

	//!	\name Implementation of tod_additive
	//@{
	virtual void perform(tensor_i<N,double> &t) throw(exception);
	virtual void perform(tensor_i<N,double> &t, double c)
		throw(exception);
	//@}
private:
	/**	\brief Add one tensor to another 

		\f[ A = A + c_B \mathcal{P}_B B \f] 

		\param a tensor data of result
		\param da dimensions of tensor A
		\param b tensor data to add
		\param db dimensions of tensor B
		\param pb permutation to be applied to B before addition
		\param cb coefficient B is multiplied with
	**/
	void add_to(double *a, const dimensions<N> &da, 
		const double *b, const dimensions<N> &db, 
		const permutation<N> &pb, double cb);
};

template<size_t N>
tod_add<N>::tod_add(const permutation<N> &p) throw(exception)
	: m_p(p), m_dim(NULL), m_head(NULL), m_tail(NULL)
{ }

template<size_t N>
tod_add<N>::~tod_add() 
{
	if ( m_dim != NULL ) delete m_dim;

	struct operand* node=m_head;
	while ( node != NULL ) {
		struct operand* next=node->m_next;
		delete next;
		node=next;
	}
}


template<size_t N>
void tod_add<N>::add_op(tensor_i<N,double> &t, const permutation<N> &p,
	const double c) throw(exception) 
{
	// don nothing if coefficient is zero
	if ( c==0. ) return;

	// modify permutation with pb
	permutation<N> new_p(p);
	new_p.permute(m_p);

	// this is the first operand added
	if ( m_head == NULL ) {
		// set dimensions of the output tensor
		m_dim=new dimensions<N>(t.get_dims());
		m_dim->permute(new_p);

		m_head=new struct operand(t,new_p,c);
		m_tail=m_head;
  	}
	// there are already operands added
	else {
		// first check whether the new operand tensor has the right dimensions
		dimensions<N> dim(t.get_dims());
		dim.permute(new_p);
		if ( dim != *m_dim ) 
			throw_exc("tod_add<N>", 
			"add_op(tensor_i<N,double>&,const permutation<N>&,const double)",
			"The tensor operands have different dimensions");

		m_tail->m_next=new operand(t,new_p,c);
		m_tail=m_tail->m_next;
	}
}

template<size_t N>
void tod_add<N>::prefetch() throw(exception) {
	struct operand* node=m_head;
	while ( node != NULL ) {
		tensor_ctrl<N,double> ctrlo(node->m_t);
		ctrlo.req_prefetch();
		node=node->m_next;
	}
}

template<size_t N>
void tod_add<N>::perform(tensor_i<N,double> &t) 
	throw(exception) 
{
	// first check whether the output tensor has the right dimensions
	if ( *m_dim != t.get_dims() )
		throw_exc("tod_add<N>", 
			"perform(tensor_i<N,double>&)",
			"The output tensor has incompatible dimensions");

	tensor_ctrl<N,double> ctrlt(t);
	double* tptr=ctrlt.req_dataptr();
	// set all elements of t to zero 
	memset(tptr,0,m_dim->get_size()*sizeof(double));
	
	struct operand* node=m_head;
	while ( node != NULL ) {
		tensor_ctrl<N,double> ctrlo(node->m_t);
		const double* optr=ctrlo.req_const_dataptr();
		add_to( tptr, *m_dim, optr, node->m_t.get_dims(), node->m_p, node->m_c );
		ctrlo.ret_dataptr(optr);	
		node=node->m_next;		
	}	

	ctrlt.ret_dataptr(tptr);
}

template<size_t N>
void tod_add<N>::perform(tensor_i<N,double> &t, double c) throw(exception) {
	// first check whether the output tensor has the right dimensions
	if ( *m_dim != t.get_dims() )
		throw_exc("tod_add<N>", 
			"perform(tensor_i<N,double>&)",
			"The output tensor has incompatible dimensions");

	if ( c==0. ) return; 

	tensor_ctrl<N,double> ctrlt(t);
	double* tptr=ctrlt.req_dataptr();
	
	struct operand* node=m_head;
	while ( node != NULL ) {
		if ( &t == &(node->m_t) ) {
			throw_exc("tod_add<N>",
				"perform(tensor_i<N,double>&)",
				"Result tensor cannot be in the operand list");
		}

		tensor_ctrl<N,double> ctrlo(node->m_t);
		const double* optr=ctrlo.req_const_dataptr();
		add_to( tptr, *m_dim, optr, node->m_t.get_dims(), node->m_p, c*node->m_c );
		ctrlo.ret_dataptr(optr);			
		node=node->m_next;		
	}	

	ctrlt.ret_dataptr(tptr);
}

template<size_t N>
void tod_add<N>::add_to(double *a, const dimensions<N> &da, 
		const double *b, const dimensions<N> &db, const permutation<N> &pb, double cb)
{
	// simplest case: just call cblas_daxpy
	if ( pb.is_identity() ) {
		cblas_daxpy(da.get_size(),cb,b,1,a,1);
	}
	else {
		// XXX this is can be reduced by sorting everything in groups

		// determine common indexes
		index<N> pbindx; // permuted index of tensor B
		for ( size_t i=0; i<N; i++ ) pbindx[i]=i;
		pbindx.permute(pb);

		// find the last permuted index
		size_t ia=N-1; 
		while ( (ia>0) && (pbindx[ia]==ia) ) ia--; 
#ifdef LIBTENSOR_DEBUG 
		// OK, this should never happen, since this is the case that m_pb is the identity
		if ( pbindx[ia]==ia )
			throw_exc("tod_add_noperm<N>", 
				"perform(tensor_i<N,double>&)",
				"Permutation pb is not identity but no index is permuted");
#endif // LIBTENSOR_DEBUG	
		// XXX

		size_t incb;
		// OK, the last index is permuted
		if ( ia == (N-1) ) {
			// position where index ia can be found in the permuted index
			size_t ib=pbindx[ia]; 
			incb=db.get_increment(ib);

			// do we have something like 0123 -> 0(23)1
			while ( (ib>0) && (pbindx[ia]==ib) ) { ia--; ib--; }

		}
		else {
			incb=1;
		}

		size_t size=da.get_increment(ia);
		

		double *aptr=a;
		const double *bptr=b;
		// reset pbindx and use it as index to keep track of the steps in tensor A
		for ( size_t i=0; i<N; i++ ) pbindx[i]=0;
		// to convert pbindx to bindx (see below)
		permutation<N> inv_pb(pb);
		inv_pb.invert();
		// loop until aptr is at the end of the array
		while ( aptr != a+da.get_size() ) {
			cblas_daxpy(size,cb,bptr,incb,aptr,1);
			aptr+=size;
								
			// now increase bptr appropriately
			size_t cnt=ia;
			pbindx[ia]++;
			while ( (cnt>0) && (pbindx[cnt]>=da[cnt]) ) {
				pbindx[cnt--]=0;
				pbindx[cnt]++;
			}
			// are we not yet at the end?
			if ( pbindx[0] < da[0] ) {
				index<N> bindx(pbindx);
				bindx.permute(inv_pb);
				bptr=b+db.abs_index(bindx);
			}
		}
	}
}

} // namespace libtensor

#endif // LIBTENSOR_TOD_ADD_H

