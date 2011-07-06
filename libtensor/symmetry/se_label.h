#ifndef LIBTENSOR_SE_LABEL_H
#define LIBTENSOR_SE_LABEL_H

#include "../defs.h"
#include "../core/block_index_space.h"
#include "../core/mask.h"
#include "../core/symmetry_element_i.h"
#include "bad_symmetry.h"
#include "label/label_set.h"
#include "label/product_table_container.h"
#include "label/product_table_i.h"

namespace libtensor {

/**	\brief Symmetry element for labels assigned to %tensor blocks
	\tparam N Symmetry cardinality (%tensor order).
	\tparam T Tensor element type.

	This %symmetry elements establishes a set of allowed blocks via labeling
	of block dimensions, masks and product tables.

    The main component of the symmetry element are so-called label sets
    (\sa label_set<N>). Each label set is defined with respect to a product
    table and a subset of "active" dimensions. Each dimension of a symmetry
    element can only be active in one label set. New label sets
    are added to the symmetry element via the function
    \code
    create_subset(const mask<N> &, const std::string &)
    \endcode
    which creates an empty label set with the active dimensions given by
    mask. The label set then needs to be set up using the respective member
    functions.

    With the label sets allowed blocks are determined as follows:
    - Any non-active dimensions are ignored, i.e.
        - no label sets == all blocks forbidden
        - the active dimensions determine the allowed blocks
    - If any label set determines a block to be allowed, the block is allowed.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class se_label : public symmetry_element_i<N, T> {
public:
	static const char *k_clazz; //!< Class name
	static const char *k_sym_type; //!< Symmetry type

	typedef label_set<N> set_t; //!< Label set
	typedef std::list<set_t *> set_list_t; //!< List of label sets

    typedef typename set_list_t::iterator iterator; //!< List iterator
	typedef typename set_list_t::const_iterator const_iterator; //!< List iterator

private:
	dimensions<N> m_bidims; //!< Block index dimensions
	mask<N> m_total_msk; //!< Total mask

	std::list<set_t *> m_sets; //!< List of label sets

public:
	//!	\name Construction and destruction
	//@{
    /** \brief Initializes the %symmetry element
        \param bidims Block %index dimensions.
     **/
    se_label(const dimensions<N> &bidims);

	/**	\brief Copy constructor
	 **/
	se_label(const se_label<N, T> &elem);

	/**	\brief Virtual destructor
	 **/
	virtual ~se_label() { clear(); }
	//@}

	//!	\name Manipulating function
	//@{

	/** \brief A label subset using a mask and a product table id

	    \param msk Mask of "active" dimensions
	    \param id Product table id
	    \return Newly created subset
	 **/
	set_t &create_subset(const mask<N> &msk, const std::string &id);

	/** \brief Clear all subsets
	 **/
	void clear();
	//@}

	//! \name STL-like iterators over label sets
	//@{
    iterator begin() { return m_sets.begin(); }
    const_iterator begin() const { return m_sets.begin(); }

    iterator end() { return m_sets.end(); }
    const_iterator end() const { return m_sets.end(); }

	set_t &get_subset(iterator it);
	const set_t &get_subset(const_iterator it) const;
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
		return new se_label<N, T>(*this);
	}

	/**	\copydoc symmetry_element_i<N, T>::permute
	 **/
	void permute(const permutation<N> &perm);

	/**	\copydoc symmetry_element_i<N, T>::is_valid_bis
	 **/
	virtual bool is_valid_bis(const block_index_space<N> &bis) const;

	/**	\copydoc symmetry_element_i<N, T>::is_allowed
	 **/
	virtual bool is_allowed(const index<N> &idx) const;

	/**	\copydoc symmetry_element_i<N, T>::apply(index<N>&)
	 **/
	virtual void apply(index<N> &idx) const { }

	/**	\copydoc symmetry_element_i<N, T>::apply(
			index<N>&, transf<N, T>&)
	 **/
	virtual void apply(index<N> &idx, transf<N, T> &tr) const { }
	//@}
};

template<size_t N, typename T>
const char *se_label<N, T>::k_clazz = "se_label<N, T>";

template<size_t N, typename T>
const char *se_label<N, T>::k_sym_type = "se_label";

template<size_t N, typename T>
se_label<N, T>::se_label(const dimensions<N> &bidims) : m_bidims(bidims) { }

template<size_t N, typename T>
se_label<N, T>::se_label(const se_label<N, T> &el) :
	m_bidims(el.m_bidims), m_total_msk(el.m_total_msk),
	m_sets(0) {

    for (const_iterator it2 = el.begin(); it2 != el.end(); it2++) {

        set_t *set = new set_t(el.get_subset(it2));
        m_sets.push_back(set);
    }
}

template<size_t N, typename T>
typename se_label<N, T>::set_t &se_label<N, T>::create_subset(
        const mask<N> &msk, const std::string &id) {

#ifdef LIBTENSOR_DEBUG
    static const char *method =
            "create_subset(const mask<N> &, const std::string &)";

    for (size_t i = 0; i < N; i++) {
        if (m_total_msk[i] && msk[i])
            throw bad_parameter(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "msk");
    }
#endif

    set_t *set = new set_t(m_bidims, msk, id);
    m_sets.push_back(set);
    m_total_msk |= msk;

    return *set;
}

template<size_t N, typename T>
void se_label<N, T>::clear() {

    for (iterator it = m_sets.begin(); it != m_sets.end(); it++) {
        delete *it; *it = 0;
    }

    for (register size_t i = 0; i < N; i++) m_total_msk[i] = false;
}

template<size_t N, typename T>
typename se_label<N, T>::set_t &se_label<N, T>::get_subset(iterator it) {

#ifdef LIBTENSOR_DEBUG
    static const char *method = "get_subset(iterator)";

    if (it == m_sets.end())
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "it");
#endif

    return *(*it);
}

template<size_t N, typename T>
const typename se_label<N, T>::set_t &se_label<N, T>::get_subset(
        const_iterator it) const {

#ifdef LIBTENSOR_DEBUG
    static const char *method = "get_subset(const_iterator)";

    if (it == m_sets.end())
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "it");
#endif

    return *(*it);
}

template<size_t N, typename T>
void se_label<N, T>::permute(const permutation<N> &p) {

    m_bidims.permute(p);
    m_total_msk.permute(p);
    for (iterator it = m_sets.begin(); it != m_sets.end(); it++)
        (*it)->permute(p);
}

template<size_t N, typename T>
bool se_label<N, T>::is_valid_bis(const block_index_space<N> &bis) const {

	const dimensions<N> &bidims = bis.get_block_index_dims();

	return (bidims == m_bidims);
}

template<size_t N, typename T>
bool se_label<N, T>::is_allowed(const index<N> &idx) const {

	static const char *method = "is_allowed(const index<N> &)";

#ifdef LIBTENSOR_DEBUG
	// Test, if index is valid block index
	for (size_t i = 0; i < N; i++) {
	    if (idx[i] >= m_bidims[i]) {
	        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
	                "idx.");
	    }
	}
#endif

	// If no label sets exist all blocks are forbidden
    if (m_sets.size() == 0) return false;

    // Loop over label sets
    for (const_iterator it = begin(); it != end(); it++) {

        if ((*it)->is_allowed(idx)) return true;
    }

	return false;
}

} // namespace libtensor

#endif // LIBTENSOR_SE_LABEL_H

