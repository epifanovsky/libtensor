#ifndef LIBTENSOR_BTOD_ADD_H
#define LIBTENSOR_BTOD_ADD_H

#include "defs.h"
#include "exception.h"
#include "block_tensor_i.h"
#include "block_tensor_ctrl.h"
#include "btod_additive.h"
#include "tod_add.h"

namespace libtensor {

/**	\brief Adds two or more block tensors

	Similar to tod_add for tensors btod_add performs the addition of n block tensors:
	\f[ B = c_B \mathcal{P_B} \left( c_1 \mathcal{P}_1 A_1 + c_2 \mathcal{P}_2 A_2 + \cdots +
		c_n \mathcal{P}_n A_n \right) \f]

	Each operand must have the same dimensions and the same block structure as the result in order
	for the operation to be successful.

	\ingroup libtensor
 **/
template<size_t N>
class btod_add : public btod_additive<N> {
private:
	struct operand {
		block_tensor_i<N, double> &m_bt;
		double m_c;
		permutation<N> m_p;
		struct operand* m_next;
		operand(block_tensor_i<N,double> &bt, const permutation<N> &p, double c)
			: m_bt(bt), m_p(p), m_c(c), m_next(NULL) {}
	};

	struct operand* m_head;
	struct operand* m_tail;

	dimensions<N>* m_dim;  //!< dimensions of the output tensor
	permutation<N> m_pb;

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Default constructor
	 **/
	btod_add();

	/**	\brief Initializes the contraction operation
		\param pb Permutation of result tensor.
	 **/
	btod_add(const permutation<N> &pb);

	/**	\brief Virtual destructor
	 **/
	virtual ~btod_add();

	//@}

	//!	\name Implementation of libtensor::btod_additive<N>
	//@{
	virtual void perform(block_tensor_i<N, double> &bt, double cb)
		throw(exception);
	//@}

	//!	\name Implementation of
	//		libtensor::direct_block_tensor_operation<N, double>
	//@{
	virtual const block_index_space_i<N> &get_bis() const;
	virtual void perform(block_tensor_i<N, double> &bt)
		throw(exception);
	//@}

	/**	\brief Adds an operand
		\param bt Tensor.
		\param p Permutation of %tensor elements.
		\param c Coefficient.
	**/
	void add_op(block_tensor_i<N, double> &bt, const permutation<N> &p,
		double c) throw(exception);

};

template<size_t N>
btod_add<N>::btod_add() : m_dim(NULL), m_head(NULL), m_tail(NULL) {

}

template<size_t N>
btod_add<N>::btod_add(const permutation<N> &pb)
	: m_pb(pb), m_dim(NULL), m_head(NULL), m_tail(NULL)
{ }

template<size_t N>
btod_add<N>::~btod_add()
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
void btod_add<N>::add_op(block_tensor_i<N,double> &bt, const permutation<N> &p,
		double c) throw(exception)
{
	// do nothing if coefficient is zero
	if ( c==0. ) return;

	// modify permutation with m_pb
	permutation<N> new_p(p);
	new_p.permute(m_pb);

	// first check whether the new operand tensor has the right dimensions
	if ( m_head == NULL ) {
		// set dimensions of the output tensor
		m_dim=new dimensions<N>(bt.get_dims());
		m_dim->permute(new_p);
	}
	else {
		dimensions<N> dim(bt.get_dims());
		dim.permute(new_p);
		if ( dim != *m_dim )
			throw_exc("btod_add<N>",
			"add_op(block_tensor_i<N,double>&,const permutation<N>&,const double)",
			"The block tensor operands have different dimensions");
	}


	struct operand* node=new struct operand(bt,new_p,c);
	if ( m_head == NULL ) {
		m_head=node;
		m_tail=m_head;
  	}
	else {
		m_tail->m_next=node;
		m_tail=node;
	}
}

template<size_t N>
void btod_add<N>::perform(block_tensor_i<N, double> &bt,
	double cb) throw(exception)
{
	// first check whether the output tensor has the right dimensions
	if ( *m_dim != bt.get_dims() )
		throw_exc("btod_add<N>",
			"perform(block_tensor_i<N,double>&)",
			"The output tensor has incompatible dimensions");

	block_tensor_ctrl<N,double> ctrlbt(bt);

	index<N> idx;
	// setup tod_add object to perform the operation on the blocks
	tod_add<N> addition(m_pb);

	struct operand* node=m_head;
	while ( node != NULL ) {
		block_tensor_ctrl<N,double> ctrlbto(node->m_bt);

		// do a prefetch here? probably not!
		// ctrlbto.req_prefetch();
		addition.add_op(ctrlbto.req_block(idx),node->m_p,node->m_c);
		node=node->m_next;
	}
	addition.prefetch();
	addition.perform(ctrlbt.req_block(idx),cb);
}

template<size_t N>
const block_index_space_i<N> &btod_add<N>::get_bis() const {
	throw_exc("btod_add<N>", "get_bis()", "Not implemented");
}

template<size_t N>
void btod_add<N>::perform(block_tensor_i<N, double> &bt)
	throw(exception)
{
	// first check whether the output tensor has the right dimensions
	if ( *m_dim != bt.get_dims() )
		throw_exc("btod_add<N>",
			"perform(block_tensor_i<N,double>&)",
			"The output tensor has incompatible dimensions");

	block_tensor_ctrl<N,double> ctrlbt(bt);

	index<N> idx;
	// setup tod_add object to perform the operation on the blocks
	tod_add<N> addition(m_pb);

	struct operand* node=m_head;
	while ( node != NULL ) {
		block_tensor_ctrl<N,double> ctrlbto(node->m_bt);

		// do a prefetch here? probably not!
		// ctrlbto.req_prefetch();
		addition.add_op(ctrlbto.req_block(idx),node->m_p,node->m_c);
		node=node->m_next;
	}

	addition.prefetch();
	addition.perform(ctrlbt.req_block(idx));
}

} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT2_H
