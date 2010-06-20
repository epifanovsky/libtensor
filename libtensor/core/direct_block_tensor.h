#ifndef LIBTENSOR_DIRECT_BLOCK_TENSOR_H
#define LIBTENSOR_DIRECT_BLOCK_TENSOR_H

#include "../defs.h"
#include "../exception.h"
#include "abs_index.h"
#include "block_map.h"
#include "direct_block_tensor_base.h"
#include "direct_block_tensor_operation.h"

namespace libtensor {

/**	\brief Direct block %tensor
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam Alloc Memory allocator type.

	\ingroup libtensor_core
 **/
template<size_t N, typename T, typename Alloc>
class direct_block_tensor : public direct_block_tensor_base<N, T> {
public:
	static const char *k_clazz; //!< Class name

public:
	typedef direct_block_tensor_base<N,T> base_t; //!< Base class type
	typedef T element_t; //!< Tensor element type
	typedef std::map<size_t,unsigned char> map_t;
	typedef std::pair<size_t,unsigned char> pair_t;

private:
	block_map<N, T, Alloc> m_map; //!< Block map
	std::map<size_t, size_t> m_count; //!< Block count
	dimensions<N> m_bidims; //!< Block %index dims

public:
	//!	\name Construction and destruction
	//@{

	direct_block_tensor(direct_block_tensor_operation<N, T> &op);
	virtual ~direct_block_tensor() { }

	//@}

	using direct_block_tensor_base<N, T>::get_bis;

protected:
	//!	\name Implementation of libtensor::block_tensor_i<N, T>
	//@{

	virtual bool on_req_is_zero_block(const index<N> &idx)
		throw(exception);
	virtual tensor_i<N, T> &on_req_block(const index<N> &idx)
		throw(exception);
	virtual void on_ret_block(const index<N> &idx) throw(exception);
	virtual tensor_i<N, T> &on_req_aux_block(const index<N> &idx)
		throw(exception);
	virtual void on_ret_aux_block(const index<N> &idx) throw(exception);

	//@}

	using direct_block_tensor_base<N, T>::get_op;

private:
	//! \brief Performs calculation of the given block
	void perform(const index<N>& idx) throw(exception);
};


template<size_t N, typename T, typename Alloc>
const char *direct_block_tensor<N, T, Alloc>::k_clazz =
	"direct_block_tensor<N, T, Alloc>";


template<size_t N, typename T, typename Alloc>
direct_block_tensor<N, T, Alloc>::direct_block_tensor(
	direct_block_tensor_operation<N, T> &op) :

	direct_block_tensor_base<N, T>(op),
	m_bidims(get_bis().get_block_index_dims()) {

}


template<size_t N, typename T, typename Alloc>
bool direct_block_tensor<N, T, Alloc>::on_req_is_zero_block(const index<N> &idx)
	throw(exception) {

	return !get_op().get_schedule().contains(idx);
}


template<size_t N, typename T, typename Alloc>
tensor_i<N, T> &direct_block_tensor<N, T, Alloc>::on_req_block(
	const index<N> &idx) throw(exception) {

	static const char *method = "on_req_block(const index<N>&)";

#ifdef LIBTENSOR_DEBUG
	if(!get_op().get_schedule().contains(idx)) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"idx");
	}
#endif // LIBTENSOR_DEBUG

	abs_index<N> aidx(idx, m_bidims);
	typename std::map<size_t, size_t>::iterator icnt =
		m_count.insert(std::pair<size_t, size_t>(
			aidx.get_abs_index(), 0)).first;
	bool newblock = icnt->second++ == 0;

	if(newblock) {
		dimensions<N> blkdims = get_op().get_bis().get_block_dims(idx);
		m_map.create(aidx.get_abs_index(), blkdims);
	}

	tensor_i<N, T> &blk = m_map.get(aidx.get_abs_index());

	if(newblock) get_op().compute_block(blk, idx);

	return blk;
}


template<size_t N, typename T, typename Alloc>
void direct_block_tensor<N, T, Alloc>::on_ret_block(const index<N> &idx)
	throw(exception) {

	static const char *method = "on_ret_block(const index<N>&)";

	abs_index<N> aidx(idx, m_bidims);
	typename std::map<size_t, size_t>::iterator icnt =
		m_count.find(aidx.get_abs_index());
	if(icnt == m_count.end()) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"idx");
	}

	if(--icnt->second == 0) {
		m_map.remove(aidx.get_abs_index());
		m_count.erase(icnt);
	}
}


template<size_t N, typename T, typename Alloc>
tensor_i<N, T> &direct_block_tensor<N, T, Alloc>::on_req_aux_block(
	const index<N> &idx) throw(exception) {

	static const char *method = "on_req_aux_block(const index<N>&)";

	throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
		"invalid_req");
}


template<size_t N, typename T, typename Alloc>
void direct_block_tensor<N, T, Alloc>::on_ret_aux_block(const index<N> &idx)
	throw(exception) {

	static const char *method = "on_ret_aux_block(const index<N>&)";

	throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
		"invalid_req");
}


} // namespace libtensor

#endif // LIBTENSOR_DIRECT_BLOCK_TENSOR_H
