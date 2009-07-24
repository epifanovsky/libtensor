#ifndef LIBTENSOR_TOD_ADD_H
#define LIBTENSOR_TOD_ADD_H

#include "defs.h"
#include "exception.h"
#include "core/tensor_i.h"
#include "core/tensor_ctrl.h"
#include "tod_additive.h"

#include <list>

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

	struct registers {
		double m_ca; //!< scaling factor
		const double* m_ptra; //!< tensor to add
		double* m_ptrb; //!< result tensor
	};

	struct loop_list_node;
	typedef std::list<loop_list_node> loop_list_t;
	typedef processor<loop_list_t, registers> processor_t;
	typedef processor_op_i<loop_list_t, registers> processor_op_i_t;

	struct loop_list_node {
		processor_op_i_t* m_op;
		size_t m_len, m_inca, m_incb;

		loop_list_node() : m_op(NULL) { }
		loop_list_node( size_t len, size_t inca, size_t incb ) :
			m_len(len), m_inca(inca), m_incb(incb), m_op(NULL) { }
		processor_op_i_t *op() const { return m_op; }
	};

	class op_loop : public processor_op_i_t {
	private:
		size_t m_len, m_inca, m_incb;
	public:
		op_loop(size_t len, size_t inca, size_t incb) :
			m_len(len), m_inca(inca), m_incb(incb) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

	//!	b_j += m_ca * a_{i_j}
	class op_daxpy : public processor_op_i_t {
	private:
		size_t m_len, m_inca, m_incb;
	public:
		op_daxpy(size_t len , size_t inca, size_t incb) :
			m_inca(inca), m_incb(incb), m_len(len) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
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

	/**	\brief Build operations list for addition of two tensors

		The resulting operations list is appropriate to add a tensor with
		dimensions da and permutation pa to tensor with dimensions db
	**/
	void build_list( loop_list_t &list, const dimensions<N> &da,
		const permutation<N> &pa, const dimensions<N> &db );
	//!	Delete all operations in list
	void clean_list( loop_list_t &list );


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
		loop_list_t list;
		build_list(list,node->m_t.get_dims(),node->m_p,t.get_dims());
		try {
			registers regs;
			regs.m_ptra=optr;
			regs.m_ca=node->m_c;
			regs.m_ptrb=tptr;
			processor_t(list,regs).process_next();
		}
		catch ( exception e ) {
			clean_list(list);
			throw;
		}

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
		loop_list_t list;
		build_list(list,node->m_t.get_dims(),node->m_p,t.get_dims());
		try {
			registers regs;
			regs.m_ptra=optr;
			regs.m_ca=c*node->m_c;
			regs.m_ptrb=tptr;
			processor_t(list,regs).process_next();
		}
		catch ( exception e ) {
			clean_list(list);
			throw;
		}

		clean_list(list);
		ctrlo.ret_dataptr(optr);
		node=node->m_next;
	}

	ctrlt.ret_dataptr(tptr);
}

template<size_t N>
void tod_add<N>::build_list( loop_list_t &list, const dimensions<N> &da,
	const permutation<N> &pa, const dimensions<N> &db )
{
	size_t ia[N];
	for (size_t i=0; i<N; i++) ia[i]=i;
	pa.apply(N,ia);

	// loop over all indices and build the list
	size_t pos=0, max_len=0;
	while ( pos < N ) {
		size_t len=1;
		size_t iapos=ia[pos];
		while (pos<N) {
			len*=da.get_dim(iapos);
			pos++; iapos++;
			if ( ia[pos]!=iapos ) break;
		}

		size_t inca=da.get_increment(iapos-1);
		size_t incb=db.get_increment(pos-1);

		list.push_back(loop_list_node(len,inca,incb));

		if ( (inca==1) || (incb==1) )
			if ( len >= max_len ) max_len=len;
	}

	// fill the list with processor_op_t
	for (typename loop_list_t::iterator it=list.begin(); it!=list.end(); it++ ) {
		if ( (it->m_len==max_len) && ((it->m_inca==1)||(it->m_incb==1)) ) {
			it->m_op=new op_daxpy(it->m_len,it->m_inca,it->m_incb);
			list.splice(list.end(),list,it);
			break;
		}
		else {
			it->m_op=new op_loop(it->m_len,it->m_inca,it->m_incb);
		}
	}
	for (typename loop_list_t::iterator it=list.begin(); it!=list.end(); it++ ) {
		if ( it->m_op == NULL ) {
			it->m_op=new op_loop(it->m_len,it->m_inca,it->m_incb);
		}
	}
}

template<size_t N>
void tod_add<N>::clean_list( loop_list_t &list )
{
	for (typename loop_list_t::iterator it=list.begin(); it != list.end(); it++) {
		delete it->m_op;
		it->m_op=NULL;
	}
}

template<size_t N>
void tod_add<N>::op_loop::exec( processor_t &proc, registers &regs)
	throw(exception)
{
	const double *ptra = regs.m_ptra;
	double *ptrb = regs.m_ptrb;

	for(size_t i=0; i<m_len; i++) {
		regs.m_ptra = ptra;
		regs.m_ptrb = ptrb;
		proc.process_next();
		ptra += m_inca;
		ptrb += m_incb;
	}

}

template<size_t N>
void tod_add<N>::op_daxpy::exec( processor_t &proc, registers &regs)
	throw(exception)
{
	cblas_daxpy(m_len,regs.m_ca,regs.m_ptra,m_inca,regs.m_ptrb,m_incb);
}

} // namespace libtensor

#endif // LIBTENSOR_TOD_ADD_H

