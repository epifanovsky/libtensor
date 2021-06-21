#ifndef LIBTENSOR_SYMMETRY_ELEMENT_SET_H
#define LIBTENSOR_SYMMETRY_ELEMENT_SET_H

#include <list>
#include <string>
#include "symmetry_element_i.h"

namespace libtensor {


/** \brief Collection of same-type %symmetry elements

	This class collects %symmetry elements of a specific type given by
	the \c id string passed to the constructor. It provides STL-style
	iterator functions to traverse the list of symmetry elements.

	TODO: add debug check for id when adding symmetry elements
	TODO: move to directory \c symmetry

    \ingroup libtensor_core
 **/
template<size_t N, typename T>
class symmetry_element_set {
public:
    static const char *k_clazz; //!< Class name

private:
    //!    Symmetry element interface type
    typedef symmetry_element_i<N, T> symmetry_element_t;

    //!    Container type
    typedef std::list<symmetry_element_t*> container_t;

public:
    //!    Collection iterator type
    typedef typename container_t::iterator iterator;

    //!    Collection constant iterator type
    typedef typename container_t::const_iterator const_iterator;

private:
    std::string m_id; //!< Symmetry type id
    container_t m_set; //!< Container

public:
    /** \brief Initializes the set with a type id
     **/
    symmetry_element_set(const char *id) : m_id(id) { }


    /** \brief Initializes the set with a type id
     **/
    symmetry_element_set(const std::string &id) : m_id(id) { }


    /** \brief Destroys the set
     **/
    ~symmetry_element_set();


    /** \brief Returns the symmetry type id
     **/
    const std::string &get_id() const {
        return m_id;
    }


    /** \brief Returns true if the set is empty
     **/
    bool is_empty() const {
        return m_set.empty();
    }


    /** \brief Removes all elements from the set
     **/
    void clear() {
        remove_all();
    }


    /** \brief Returns an iterator pointing at the first element in
            the set
     **/
    iterator begin() {
        return m_set.begin();
    }


    /** \brief Returns an iterator pointing at the first element in
            the set (const version)
     **/
    const_iterator begin() const {
        return m_set.begin();
    }


    /** \brief Returns an iterator pointing at the last element in
            the set
     **/
    iterator end() {
        return m_set.end();
    }


    /** \brief Returns an iterator pointing at the last element in
            the set (const version)
     **/
    const_iterator end() const {
        return m_set.end();
    }


    /** \brief Returns a %symmetry element specified by an iterator
     **/
    symmetry_element_i<N, T> &get_elem(iterator &i) {
        return **i;
    }


    /** \brief Returns a %symmetry element specified by an iterator
            (const version)
     **/
    const symmetry_element_i<N, T> &get_elem(const_iterator &i) const {
        return **i;
    }


    /** \brief Inserts a %symmetry element to the set
     **/
    void insert(const symmetry_element_i<N, T> &elem) {
        m_set.push_back(elem.clone());
    }


    /** \brief Removes a %symmetry element from the set
     **/
    void remove(iterator &i) {
        delete *i;
        m_set.erase(i);
    }

private:
    void remove_all() noexcept;

};


template<size_t N, typename T>
const char *symmetry_element_set<N, T>::k_clazz = "symmetry_element_set<N, T>";


template<size_t N, typename T>
symmetry_element_set<N, T>::~symmetry_element_set() {

    remove_all();
}


template<size_t N, typename T>
void symmetry_element_set<N, T>::remove_all() noexcept {

    for(iterator i = m_set.begin(); i != m_set.end(); i++) {
        delete *i;
    }
    m_set.clear();
}


} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_ELEMENT_SET_H
