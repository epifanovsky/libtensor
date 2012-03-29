#ifndef LIBTENSOR_TENSOR_TRANSF_H
#define LIBTENSOR_TENSOR_TRANSF_H

#include "../defs.h"
#include "../exception.h"
#include "index.h"
#include "permutation.h"
#include "scalar_transf.h"

namespace libtensor {

/**	\brief Describes how the canonical block needs to be transformed to
		 obtain a replica
	\tparam N Tensor order.
	\tparam T Tensor element type.

    This element consists of a transformation to be applied to the tensor
    elements and a permutation of the block index.

	\ingroup libtensor_core
 **/
template<size_t N, typename T>
class tensor_transf {
    scalar_transf<T> m_st; //!< Scalar transformation
    permutation<N> m_perm; //!< Permutation

public:
    /** \brief Constructor
     **/
    tensor_transf(const permutation<N> &p = permutation<N>(),
            const scalar_transf<T> &st = scalar_transf<T>()) :
        m_perm(p), m_st(st) { }

    //! \name Manipulators
    //@{
    /** \brief Reset tensor transformation to identity transformation
     **/
	void reset() {
	    m_st.reset(); m_perm.reset();
	}

	/** \brief Apply permutation perm to this transformation
	 **/
	void permute(const permutation<N> &perm) { m_perm.permute(perm); }

	/** \brief Apply scalar transformation st to this transformation
	 **/
	void transform(const scalar_transf<T> &st) { m_st.transform(st); }

	/** \brief Apply tr to this transformation
	 **/
	void transform(const tensor_transf<N, T> &tr) {
	    m_st.transform(tr.m_st);
	    m_perm.permute(tr.m_perm);
	}

	/** \brief Invert this transformation
	 **/
	void invert() { m_st.invert(); m_perm.invert(); }

	/** \brief Apply transformation to an tensor block index
	 **/
	void apply(index<N> &idx) const { idx.permute(m_perm); }

	//@}

	//! \name Member access functions
	//@{

	const scalar_transf<T> &get_scalar_tr() const { return m_st; }
	const permutation<N> &get_perm() const { return m_perm; }

	//@}

	/** \brief Check if tensor transformation is identity transformation.
	 **/
	bool is_identity() const {
	    return m_st.is_identity() && m_perm.is_identity();
	}

	//! \name Comparison operators
	//@{

	bool operator==(const tensor_transf<N, T> &tr) const {
	    return ((m_st == tr.m_st) && (m_perm.equals(tr.m_perm)));
	}

	bool operator!=(const tensor_transf<N, T> &tr) const {
	    return (! operator==(tr));
	}

	//@}
};


} // namespace libtensor

#endif // LIBTENSOR_TENSOR_TRANSF_H
