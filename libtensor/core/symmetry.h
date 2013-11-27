#ifndef LIBTENSOR_SYMMETRY_H
#define LIBTENSOR_SYMMETRY_H

#include <algorithm>
#include <list>
#include "../defs.h"
#include "../exception.h"
#include "block_index_space.h"
#include "symmetry_element_set.h"

namespace libtensor {

/** \brief Tensor symmetry
    \tparam N Tensor order.
    \tparam T Tensor element type.

	The class represents the %symmetry of a (block) %tensor by storing
	a list of %symmetry elements.

	TODO Move to folder symmetry.

    \ingroup libtensor_core
 **/
template<size_t N, typename T>
class symmetry {
public:
    static const char *k_clazz; //!< Class name

public:
    //!    Read-only iterator
    typedef typename std::list<
        symmetry_element_set<N, T>*>::const_iterator iterator;

private:
    block_index_space<N> m_bis; //!< Block %index space
    std::list<symmetry_element_set<N, T>*> m_subsets; //!< Symmetry subsets

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Creates %symmetry using a given block %index space
        \param bis Block %index space.
     **/
    symmetry(const block_index_space<N> &bis) : m_bis(bis) { }

    /** \brief Destructor
     **/
    ~symmetry();

    //@}


    //!    \name Manipulations
    //@{

    const block_index_space<N> &get_bis() const { return m_bis; }

    /** \brief Inserts an element in the group
     **/
    void insert(const symmetry_element_i<N, T> &e);

    /** \brief Returns the begin iterator
     **/
    iterator begin() const {
        return m_subsets.begin();
    }

    /** \brief Returns the end iterator
     **/
    iterator end() const {
        return m_subsets.end();
    }

    /** \brief Returns the subset pointed at by an iterator
     **/
    const symmetry_element_set<N, T> &get_subset(iterator &i) const {
        return **i;
    }

    /** \brief Removes all %symmetry elements
     **/
    void clear() {
        remove_all();
    }

    //@}

private:
    void remove_all();

private:
    symmetry(const symmetry<N, T>&);
    const symmetry<N, T> &operator=(const symmetry<N, T>&);

};


template<size_t N, typename T>
const char *symmetry<N, T>::k_clazz = "symmetry<N, T>";


template<size_t N, typename T>
symmetry<N, T>::~symmetry() {

    remove_all();
}


template<size_t N, typename T>
void symmetry<N, T>::insert(const symmetry_element_i<N, T> &e) {

    typename std::list<symmetry_element_set<N, T>*>::iterator i =
        m_subsets.begin();
    while(i != m_subsets.end() &&
        (*i)->get_id().compare(e.get_type()) != 0) i++;
    if(i == m_subsets.end()) {
        i = m_subsets.insert(m_subsets.end(),
            new symmetry_element_set<N, T>(e.get_type()));
    }
    (*i)->insert(e);
}


template<size_t N, typename T>
void symmetry<N, T>::remove_all() {

    if(m_subsets.empty()) return;
    for(typename std::list<symmetry_element_set<N, T>*>::iterator i =
        m_subsets.begin(); i != m_subsets.end(); i++) delete *i;
    m_subsets.clear();
}


} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_H
