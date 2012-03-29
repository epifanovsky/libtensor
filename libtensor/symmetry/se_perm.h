#ifndef LIBTENSOR_SE_PERM_H
#define LIBTENSOR_SE_PERM_H

#include <libtensor/core/symmetry_element_i.h>

namespace libtensor {

/**	\brief Permutational %symmetry element
	\tparam N Tensor order.
	\tparam T Tensor element type.

	Permutation %symmetry elements establish relationships among block
	%tensor blocks based on permuting the blocks' multi-indexes.
	An appropriate transformation needs to be applied to the elements in
	the block as well.

	The element is initialized with a %permutation of %tensor indexes,
	and a element-wise (scalar) transformation of %tensor elements. The
	%permutation and the scalar transformation must agree, i.e. if the
	n-th power of the permutation yields the identity permutation, the
	n-th power of the scalar transformation also needs to be the identity
	transformation. The agreement is tested upon the creation of the
	element. If it is not satisfied, an exception is thrown.

	\ingroup libtensor_symmetry
 **/
template< size_t N, typename T>
class se_perm : public symmetry_element_i<N, T> {
public:
    static const char *k_clazz; //!< Class name
    static const char *k_sym_type; //!< Symmetry type

private:
    tensor_transf<N, T> m_transf; //!< Block transformation
    size_t m_orderp; //!< Order of permutation
    size_t m_orderc; //!< Order of scalar transformation

public:
    //!	\name Construction and destruction
    //@{

    /**	\brief Initializes the %symmetry element
		\param perm Permutation.
		\param tr Scalar transformation.
		\throw bad_symmetry If the permutation and the flag are
			inconsistent.
     **/
    se_perm(const permutation<N> &perm, const scalar_transf<T> &tr);

    /**	\brief Copy constructor
     **/
    se_perm(const se_perm<N, T> &elem) : m_transf(elem.m_transf),
            m_orderp(elem.m_orderp), m_orderc(elem.m_orderc) { }

    /**	\brief Virtual destructor
     **/
    virtual ~se_perm() { }

    //@}


    //!	\name Permutational %symmetry
    //@{

    const permutation<N> &get_perm() const {
        return m_transf.get_perm();
    }

    const scalar_transf<T> &get_transf() const {
        return m_transf.get_scalar_tr();
    }

    size_t get_orderp() const { return m_orderp; }

    size_t get_orderc() const { return m_orderc; }

    //@}

    //!	\name Implementation of symmetry_element_i<N, T>
    //@{

    /**	\copydoc symmetry_element_i<N, T>::get_type()
     **/
    virtual const char *get_type() const {
        return k_sym_type;
    }

    /**	\copydoc symmetry_element_i<N, T>::clone()
     **/
    virtual symmetry_element_i<N, T> *clone() const {
        return new se_perm<N, T>(*this);
    }

    /**	\copydoc symmetry_element_i<N, T>::is_valid_bis
     **/
    virtual bool is_valid_bis(const block_index_space<N> &bis) const;

    /**	\copydoc symmetry_element_i<N, T>::is_allowed
     **/
    virtual bool is_allowed(const index<N> &idx) const {

        return true;
    }

    /**	\copydoc symmetry_element_i<N, T>::apply(index<N>&)
     **/
    virtual void apply(index<N> &idx) const;

    /**	\copydoc symmetry_element_i<N, T>::apply(
			index<N>&, transf<N, T>&)
     **/
    virtual void apply(index<N> &idx, tensor_transf<N, T> &tr) const;

    //@}
};

template<size_t N, typename T>
inline
bool se_perm<N, T>::is_valid_bis(const block_index_space<N> &bis) const {

    block_index_space<N> bis2(bis);
    bis2.permute(m_transf.get_perm());
    return bis2.equals(bis);
}

template<size_t N, typename T>
inline
void se_perm<N, T>::apply(index<N> &idx) const {

    m_transf.apply(idx);
}

template<size_t N, typename T>
inline
void se_perm<N, T>::apply(index<N> &idx, tensor_transf<N, T> &tr) const {

    m_transf.apply(idx);
    tr.transform(m_transf);
}



} // namespace libtensor

#endif // LIBTENSOR_SE_PERM_H
