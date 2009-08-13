#ifndef LIBTENSOR_BTOD_ADD_H
#define LIBTENSOR_BTOD_ADD_H

#include <new>
#include <vector>
#include "defs.h"
#include "exception.h"
#include "core/block_tensor_i.h"
#include "core/block_tensor_ctrl.h"
#include "tod/tod_add.h"
#include "btod_additive.h"
#include "btod_so_copy.h"

namespace libtensor {


/**	\brief Addition of multiple block tensors

	This block %tensor operation performs the addition of block tensors:
	\f[ B = c_1 \mathcal{P}_1 A_1 + c_2 \mathcal{P}_2 A_2 + \cdots \f]

	The operation must have at least one operand provided at the time of
	construction. Other operands are added afterwards and must agree in
	the dimensions and the block structure.

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_add : public btod_additive<N> {
public:
	static const char *k_clazz; //!< Class name

private:
	typedef struct operand {
		block_tensor_i<N, double> &m_bt; //!< Block %tensor
		permutation<N> m_perm; //!< Permutation
		double m_c; //!< Scaling coefficient
		operand(block_tensor_i<N, double> &bt,
			const permutation<N> &perm, double c)
		: m_bt(bt), m_perm(perm), m_c(c) { };
	} operand_t;

	block_index_space<N> m_bis; //!< Block %index space of the result
	dimensions<N> m_bidims; //!< Block %index dimensions
	symmetry<N, double> m_symmetry; //!< Symmetry of the result
	std::vector<operand_t*> m_ops; //!< Operand list

public:
	//!	\name Construction, destruction, initialization
	//@{

	/**	\brief Initializes the addition operation
		\param bt First block %tensor in the series.
		\param c Scaling coefficient.
	 **/
	btod_add(block_tensor_i<N, double> &bt, double c = 1.0);

	/**	\brief Initializes the addition operation
		\param bt First block %tensor in the series.
		\param pb Permutation of the first %tensor.
		\param c Scaling coefficient.
	 **/
	btod_add(block_tensor_i<N, double> &bt, const permutation<N> &p,
		double c = 1.0);

	/**	\brief Virtual destructor
	 **/
	virtual ~btod_add();

	/**	\brief Adds an operand (block %tensor in the series)
		\param bt Block %tensor.
		\param c Scaling coefficient.
		\throw bad_parameter If the block %tensor has incompatible
			%dimensions or block structure.
		\throw out_of_memory If memory allocation fails.
	 **/
	void add_op(block_tensor_i<N, double> &bt, double c = 1.0)
		throw(bad_parameter, out_of_memory);

	/**	\brief Adds an operand (block %tensor in the series)
		\param bt Block %tensor.
		\param perm Permutation of the block %tensor.
		\param c Scaling coefficient.
		\throw bad_parameter If the block %tensor has incompatible
			%dimensions or block structure.
		\throw out_of_memory If memory allocation fails.
	 **/
	void add_op(block_tensor_i<N, double> &bt, const permutation<N> &perm,
		double c = 1.0) throw(bad_parameter, out_of_memory);

	//@}

	//!	\name Implementation of
	//		libtensor::direct_block_tensor_operation<N, double>
	//@{
	virtual const block_index_space<N> &get_bis() const;
	virtual const symmetry<N, double> &get_symmetry() const;
	virtual void perform(block_tensor_i<N, double> &bt)
		throw(exception);
	//@}

	//!	\name Implementation of libtensor::btod_additive<N>
	//@{
	virtual void perform(block_tensor_i<N, double> &bt, double cb)
		throw(exception);
	//@}

private:
	void add_operand(block_tensor_i<N, double> &bt,
		const permutation<N> &perm, double c) throw(out_of_memory);
};


template<size_t N>
const char *btod_add<N>::k_clazz = "btod_add<N>";


template<size_t N>
btod_add<N>::btod_add(block_tensor_i<N, double> &bt, double c)
: m_bis(bt.get_bis()), m_bidims(m_bis.get_block_index_dims()),
	m_symmetry(m_bidims) {

	add_operand(bt, permutation<N>(), c);
}


template<size_t N>
btod_add<N>::btod_add(
	block_tensor_i<N, double> &bt, const permutation<N> &perm, double c)
: m_bis(bt.get_bis()), m_bidims(m_bis.get_block_index_dims().permute(perm)),
	m_symmetry(m_bidims) {

	m_bis.permute(perm);
	add_operand(bt, perm, c);
}


template<size_t N>
btod_add<N>::~btod_add() {

	typename std::vector<operand_t*>::iterator i = m_ops.begin();
	while(i != m_ops.end()) {
		delete (*i);
		*i = NULL;
		i++;
	}
}


template<size_t N>
void btod_add<N>::add_op(block_tensor_i<N, double> &bt, double c)
	throw(bad_parameter, out_of_memory) {

	static const char *method =
		"add_op(block_tensor_i<N, double>&, double)";

	if(c == 0.0) return;

	if(!m_bis.equals(bt.get_bis())) {
		throw bad_parameter("libtensor", k_clazz, method, __FILE__,
			__LINE__, "Incompatible block index space.");
	}

	add_operand(bt, permutation<N>(), c);

	// Reduce symmetry if necessary
/*
	block_tensor_ctrl<N, double> bt_ctrl(bt);
	for(size_t i = 0; i < m_symmetry.get_num_elements(); i++) {
		if(!bt_ctrl.req_sym_contains_element(
			m_symmetry.get_element(i))) {


		}
	}*/
}


template<size_t N>
void btod_add<N>::add_op(block_tensor_i<N, double> &bt,
	const permutation<N> &perm, double c)
	throw(bad_parameter, out_of_memory) {

	static const char *method = "add_op(block_tensor_i<N, double>&, "
		"const permutation<N>&, double)";

	if(c == 0.0) return;

	block_index_space<N> bis(bt.get_bis());
	bis.permute(perm);
	if(!m_bis.equals(bis)) {
		throw bad_parameter("libtensor", k_clazz, method, __FILE__,
			__LINE__, "Incompatible block index space.");
	}

	add_operand(bt, perm, c);

	// Reduce symmetry if necessary
}


template<size_t N>
const block_index_space<N> &btod_add<N>::get_bis() const {

	return m_bis;
}


template<size_t N>
const symmetry<N, double> &btod_add<N>::get_symmetry() const {

	return m_symmetry;
}


template<size_t N>
void btod_add<N>::perform(block_tensor_i<N, double> &bt) throw(exception) {

	static const char *method = "perform(block_tensor_i<N, double>&)";

	if(!m_bis.equals(bt.get_bis())) {
		throw bad_parameter("libtensor", k_clazz, method, __FILE__,
			__LINE__, "Incompatible block index space.");
	}

	btod_so_copy<N> symcopy(m_symmetry);
	symcopy.perform(bt);

	block_tensor_ctrl<N, double> dst_ctrl(bt);
	std::vector< block_tensor_ctrl<N, double>* > src_ctrl(
		m_ops.size(), NULL);
	dimensions<N> bidims = m_bis.get_block_index_dims();

	for(size_t iop = 0; iop < m_ops.size(); iop++) {
		src_ctrl[iop] =
			new block_tensor_ctrl<N, double>(m_ops[iop]->m_bt);
	}

	size_t norbits = dst_ctrl.req_sym_num_orbits();
	for(size_t iorbit = 0; iorbit < norbits; iorbit++) {

		orbit<N, double> orb = dst_ctrl.req_sym_orbit(iorbit);
		index<N> dst_blk_idx;
		bidims.abs_index(orb.get_abs_index(), dst_blk_idx);

		permutation<N> perm0;
		tod_add<N> todadd(perm0);
		std::vector< index<N> > src_blk_idx(m_ops.size());

		for(size_t iop = 0; iop < m_ops.size(); iop++) {
			operand_t &op = *(m_ops[iop]);
			permutation<N> invperm(op.m_perm);
			invperm.invert();
			src_blk_idx[iop] = dst_blk_idx;
			src_blk_idx[iop].permute(invperm);
			if(src_ctrl[iop]->req_is_zero_block(src_blk_idx[iop]))
				continue;
			tensor_i<N, double> &src_blk =
				src_ctrl[iop]->req_block(src_blk_idx[iop]);
			todadd.add_op(src_blk, op.m_perm, op.m_c);
		}

		tensor_i<N, double> &dst_blk = dst_ctrl.req_block(dst_blk_idx);
		todadd.perform(dst_blk);

		for(size_t iop = 0; iop < m_ops.size(); iop++) {
			src_ctrl[iop]->ret_block(src_blk_idx[iop]);
		}
		dst_ctrl.ret_block(dst_blk_idx);

	}

	for(size_t iop = 0; iop < m_ops.size(); iop++) {
		delete src_ctrl[iop];
		src_ctrl[iop] = NULL;
	}
/*
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
	addition.perform(ctrlbt.req_block(idx));*/
}


template<size_t N>
void btod_add<N>::perform(block_tensor_i<N, double> &bt,
	double cb) throw(exception)
{
	/*
	// first check whether the output tensor has the right dimensions
	if ( *m_dim != bt.get_bis().get_dims() )
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
	addition.perform(ctrlbt.req_block(idx),cb);*/
}


template<size_t N>
void btod_add<N>::add_operand(block_tensor_i<N,double> &bt,
	const permutation<N> &perm, double c) throw(out_of_memory) {

	static const char *method = "add_operand(block_tensor_i<N,double>&, "
		"const permutation<N>&, double)";

	try {
		operand_t *op = new operand(bt, perm, c);
		m_ops.push_back(op);
	} catch(std::bad_alloc &e) {
		throw out_of_memory("libtensor", k_clazz, method, __FILE__,
			__LINE__, "op");
	}
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_ADD_H
