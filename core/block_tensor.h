#ifndef LIBTENSOR_BLOCK_TENSOR_H
#define LIBTENSOR_BLOCK_TENSOR_H

#include "defs.h"
#include "exception.h"
#include "block_index_space.h"
#include "block_iterator.h"
#include "block_map.h"
#include "block_tensor_i.h"
#include "immutable.h"
#include "orbit_iterator.h"
#include "symmetry_i.h"
#include "tensor.h"
#include "tensor_ctrl.h"

namespace libtensor {

/**	\brief Block %tensor
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam Symmetry Block %tensor symmetry.
	\tparam Alloc Memory allocator.

	\ingroup libtensor_core
 **/
template<size_t N, typename T, typename Symmetry, typename Alloc>
class block_tensor : public block_tensor_i<N, T>, public immutable {
private:
	static const char *k_clazz; //!< Class name

private:
	class bi_handler : public block_iterator_handler_i<N, T> {
	public:
		virtual ~bi_handler() { };
		virtual void on_begin(index<N> &idx, block_symop<N, T> &symop,
			const index<N> &orbit, const dimensions<N> &dims) const;
		virtual bool on_end(const index<N> &idx, const index<N> &orbit,
			const dimensions<N> &dims) const = 0;
		virtual void on_next(index<N> &idx, block_symop<N, T> &symop,
			const index<N> &orbit, const dimensions<N> &dims) const;
	};

	class oi_handler : public orbit_iterator_handler_i<N, T> {
	public:
		virtual ~oi_handler() { };
		virtual void on_begin(index<N> &idx,
			const dimensions<N> &dims) const;
		virtual bool on_end(const index<N> &idx,
			const dimensions<N> &dims) const;
		virtual void on_next(index<N> &idx,
			const dimensions<N> &dims) const;
	};

public:
	block_index_space<N> m_bis; //!< Block %index space
	dimensions<N> m_bidims; //!< Block %index %dimensions
	Symmetry m_symmetry; //!< Block %tensor symmetry
	block_map<N, T, Alloc> m_map; //!< Block map
	tensor<N, T, Alloc> m_t;
	tensor_ctrl<N, T> m_ctrl;

public:
	//!	\name Construction and destruction
	//@{
	block_tensor(const block_index_space<N> &bis);
	//block_tensor(const block_tensor_i<N, T> &bt);
	block_tensor(const block_tensor<N, T, Symmetry, Alloc> &bt);
	virtual ~block_tensor();
	//@}

	//!	\name Implementation of libtensor::tensor_i<N, T>
	//@{
	virtual const dimensions<N> &get_dims() const;
	//@}

	//!	\name Implementation of libtensor::block_tensor_i<N, T>
	//@{
	virtual const block_index_space<N> &get_bis() const;
	//@}

protected:
	//!	\name Implementation of libtensor::block_tensor_i<N, T>
	//@{
	virtual const symmetry_i<N, T> &on_req_symmetry() throw(exception);
	virtual void on_req_symmetry_operation(symmetry_operation_i<N, T> &op)
		throw(exception);
	virtual orbit_iterator<N, T> on_req_orbits() throw(exception);
	virtual tensor_i<N, T> &on_req_block(const index<N> &idx)
		throw(exception);
	virtual void on_ret_block(const index<N> &idx) throw(exception);
	//@}

	//!	\name Implementation of libtensor::tensor_i<N, T>
	//@{
	virtual void on_req_prefetch() throw(exception);
	virtual T *on_req_dataptr() throw(exception);
	virtual const T *on_req_const_dataptr() throw(exception);
	virtual void on_ret_dataptr(const T *p) throw(exception);
	//@}

	//!	\name Implementation of libtensor::immutable
	//@{
	virtual void on_set_immutable();
	//@}
};


template<size_t N, typename T, typename Symmetry, typename Alloc>
const char *block_tensor<N, T, Symmetry, Alloc>::k_clazz =
	"block_tensor<N, T, Symmetry, Alloc>";


template<size_t N, typename T, typename Symmetry, typename Alloc>
block_tensor<N, T, Symmetry, Alloc>::block_tensor(
	const block_index_space<N> &bis)
: m_bis(bis), m_bidims(bis.get_block_index_dims()), m_symmetry(m_bidims),
	m_t(m_bis.get_dims()), m_ctrl(m_t) {

}
/*
template<size_t N, typename T, typename Alloc>
block_tensor<N, T, Alloc>::block_tensor(const block_tensor_i<N, T> &bt)
: m_bis(bt.get_bis()), m_bi_dims(m_bis.get_block_index_dims()),
	m_t(m_bis.get_dims()), m_ctrl(m_t) {

}
*/
template<size_t N, typename T, typename Symmetry, typename Alloc>
block_tensor<N, T, Symmetry, Alloc>::block_tensor(
	const block_tensor<N, T, Symmetry, Alloc> &bt)
: m_bis(bt.get_bis()), m_bidims(bt.m_bidims), m_symmetry(bt.m_symmetry),
	m_t(m_bis.get_dims()), m_ctrl(m_t) {

}

template<size_t N, typename T, typename Symmetry, typename Alloc>
block_tensor<N, T, Symmetry, Alloc>::~block_tensor() {

}

template<size_t N, typename T, typename Symmetry, typename Alloc>
const dimensions<N> &block_tensor<N, T, Symmetry, Alloc>::get_dims() const {
	return m_t.get_dims();
}

template<size_t N, typename T, typename Symmetry, typename Alloc>
const block_index_space<N> &block_tensor<N, T, Symmetry, Alloc>::get_bis()
	const {

	return m_bis;
}

template<size_t N, typename T, typename Symmetry, typename Alloc>
const symmetry_i<N, T> &block_tensor<N, T, Symmetry, Alloc>::on_req_symmetry()
	throw(exception) {
	throw_exc("block_tensor<N, T, Alloc>", "on_req_symmetry()", "NIY");
}

template<size_t N, typename T, typename Symmetry, typename Alloc>
void block_tensor<N, T, Symmetry, Alloc>::on_req_symmetry_operation(
	symmetry_operation_i<N, T> &op) throw(exception) {
	throw_exc("block_tensor<N, T, Alloc>", "on_req_symmetry_operation()", "NIY");
}

template<size_t N, typename T, typename Symmetry, typename Alloc>
orbit_iterator<N, T> block_tensor<N, T, Symmetry, Alloc>::on_req_orbits() throw(exception) {
	throw_exc("block_tensor<N, T, Alloc>", "on_req_orbits()", "NIY");
}

template<size_t N, typename T, typename Symmetry, typename Alloc>
tensor_i<N, T> &block_tensor<N, T, Symmetry, Alloc>::on_req_block(const index<N> &idx)
	throw(exception) {
	return m_t;
}

template<size_t N, typename T, typename Symmetry, typename Alloc>
void block_tensor<N, T, Symmetry, Alloc>::on_ret_block(const index<N> &idx)
	throw(exception) {
	throw_exc("block_tensor<N, T, Alloc>", "on_ret_block()", "NIY");
}

template<size_t N, typename T, typename Symmetry, typename Alloc>
void block_tensor<N, T, Symmetry, Alloc>::on_req_prefetch() throw(exception) {
	m_ctrl.req_prefetch();
}

template<size_t N, typename T, typename Symmetry, typename Alloc>
T *block_tensor<N, T, Symmetry, Alloc>::on_req_dataptr() throw(exception) {
	return m_ctrl.req_dataptr();
}

template<size_t N, typename T, typename Symmetry, typename Alloc>
const T *block_tensor<N, T, Symmetry, Alloc>::on_req_const_dataptr() throw(exception) {
	return m_ctrl.req_const_dataptr();
}

template<size_t N, typename T, typename Symmetry, typename Alloc>
void block_tensor<N, T, Symmetry, Alloc>::on_ret_dataptr(const T *p) throw(exception) {
	m_ctrl.ret_dataptr(p);
}

template<size_t N, typename T, typename Symmetry, typename Alloc>
void block_tensor<N, T, Symmetry, Alloc>::on_set_immutable() {
	m_t.set_immutable();
}

} // namespace libtensor

#endif // LIBTENSOR_BLOCK_TENSOR_H
