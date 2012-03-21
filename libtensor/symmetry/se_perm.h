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
	which also specifies a transformation of %tensor elements. The
	symmetric/anti-symmetric flag yields the scalar coefficient in
	the transformation. The permutation and the flag must agree: when
	the block transformation is applied onto itself multiple times such
	that the original permutation is recovered, the scalar coefficient
	must turn unity. The agreement is tested upon the creation of the
	element. If it is not satisfied, an exception is thrown.

    TODO:
    - replace permutation and even / symm by tensor_transf

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class se_perm : public symmetry_element_i<N, T> {
public:
    static const char *k_clazz; //!< Class name
    static const char *k_sym_type; //!< Symmetry type

private:
    permutation<N> m_perm; //!< Permutation
    bool m_even; //!< Even/odd %permutation
    bool m_symm; //!< Symmetric/anti-symmetric
    transf<N, T> m_transf; //!< Block transformation

public:
    //!	\name Construction and destruction
    //@{

    /**	\brief Initializes the %symmetry element
		\param perm Permutation.
		\param symm Symmetric/anti-symmetric.
		\throw bad_symmetry If the permutation and the flag are
			inconsistent.
a     **/
    se_perm(const permutation<N> &perm, bool symm);

    /**	\brief Copy constructor
     **/
    se_perm(const se_perm<N, T> &elem);

    /**	\brief Virtual destructor
     **/
    virtual ~se_perm() { }

    //@}


    //!	\name Permutational %symmetry
    //@{

    const permutation<N> &get_perm() const {
        return m_perm;
    }

    bool is_symm() const {
        return m_symm;
    }

    const transf<N, T> &get_transf() const {
        return m_transf;
    }

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
    virtual void apply(index<N> &idx, transf<N, T> &tr) const;

    //@}
};

} // namespace libtensor

#endif // LIBTENSOR_SE_PERM_H
