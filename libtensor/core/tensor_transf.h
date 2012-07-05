#ifndef LIBTENSOR_TENSOR_TRANSF_H
#define LIBTENSOR_TENSOR_TRANSF_H

#include "../defs.h"
#include "../exception.h"
#include "index.h"
#include "permutation.h"
#include "scalar_transf.h"

namespace libtensor {

/** \brief Generalized transformation of a tensor.
    \tparam N Tensor order.
    \tparam T Tensor element type.

    The transformation of a tensor is described by a permutation of tensor
    indexes and a element-wise transformation applied to all tensor elements
    individually.

    \ingroup libtensor_core
 **/
template<size_t N, typename T>
class tensor_transf {
    scalar_transf<T> m_st; //!< Element-wise transformation
    permutation<N> m_perm; //!< Permutation

public:
    /** \brief Default constructor
        \param p Permutation (default: identity permutation).
        \param st Element-wise transformation (default: identity).
     **/
    tensor_transf(const permutation<N> &p = permutation<N>(),
            const scalar_transf<T> &st = scalar_transf<T>()) :
        m_perm(p), m_st(st) { }

    /** \brief Copy constructor
        @param tr Other transformation
        @param inverse Flag to obtain the inverse of tr (default: false)
     **/
    tensor_transf(const tensor_transf<N, double> &tr, bool inverse = false) :
        m_perm(tr.m_perm, inverse), m_st(tr.m_st) {

        if (inverse) m_st.invert();
    }

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

    /** \brief Apply transformation to a tensor index
     **/
    void apply(index<N> &idx) const { idx.permute(m_perm); }

    /** \brief Apply transformation to a tensor element
     **/
    void apply(T &el) const { m_st.apply(el); }

    //@}

    //! \name Member access functions
    //@{

    /** \brief Access to scalar transformation
     **/
    scalar_transf<T> &get_scalar_tr() { return m_st; }

    /** \brief Access to scalar transformation (const)
     **/
    const scalar_transf<T> &get_scalar_tr() const { return m_st; }

    /** \brief Access to permutation transformation
     **/
    permutation<N> &get_perm() { return m_perm; }

    /** \brief Access to permutation transformation (const)
     **/
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
