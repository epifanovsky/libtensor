#ifndef LIBTENSOR_SEQUENCE_H
#define LIBTENSOR_SEQUENCE_H

#include "out_of_bounds.h"

namespace libtensor {


/** \brief Fixed-length array of objects of given type
    \tparam N Sequence length.
    \tparam T Object type.

    Simple container for a fixed-length array of objects. This class provides
    methods for accessing the array elements with or without doing boundary
    checks: at() and at_nothrow(). In addition, the overloaded operator[]
    only performs boundary checks when LIBTENSOR_DEBUG is enabled.

    The template parameter N specifies the length of the array, the parameter T
    specifies the type of objects in the array. T can be any POD type or any
    structure or class that defines the default constructor and assignment
    operation (operator=).

    \sa sequence<0, T>, sequence_test

    \ingroup libtensor_core
 **/
template<size_t N, typename T>
class sequence {
public:
    static const char *k_clazz; //!< Class name

private:
    T m_seq[N]; //!< Array of objects

public:
    //! \name Construction and destruction
    //@{

    /** \brief Default constructor
        \param t Value to be used as the elements of the sequence.
     **/
    sequence(const T &t = T()) {
        for(size_t i = 0; i < N; i++) m_seq[i] = t;
    }

    /** \brief Copy constructor
        \param seq Another sequence.
     **/
    sequence(const sequence<N, T> &seq) {
        for(size_t i = 0; i < N; i++) m_seq[i] = seq.m_seq[i];
    }

    //@}


    //! \name Access to %sequence elements, manipulations
    //@{

    /** \brief Returns the reference to the element at the given position
            (l-value)
        \param pos Position (not to exceed N).
        \throw out_of_bounds If the position exceeds N.
     **/
    T &at(size_t pos) {
        check_bounds(pos);
        return m_seq[pos];
    }

    /** \brief Returns the element at the given position (r-value)
        \param pos Position (not to exceed N).
        \throw out_of_bounds If the position exceeds N.
     **/
    const T &at(size_t pos) const {
        check_bounds(pos);
        return m_seq[pos];
    }

    /** \brief Returns the reference to the element at the given position
            (l-value). Does not check the position
        \param pos Position.
     **/
    T &at_nothrow(size_t pos) {
        return m_seq[pos];
    }

    /** \brief Returns the element at the given position (r-value). Does not
            check the position
        \param pos Position.
     **/
    const T &at_nothrow(size_t pos) const {
        return m_seq[pos];
    }

    //@}


    //!	\name Overloaded operators
    //@{

    /** \brief Returns the reference to the element at the given position
            (l-value)
        \param pos Position (not to exceed N).
        \throw out_of_bounds If the position exceeds N (LIBTENSOR_DEBUG only).
     **/
    T &operator[](size_t pos) {
#ifdef LIBTENSOR_DEBUG
        return at(pos);
#else // LIBTENSOR_DEBUG
        return at_nothrow(pos);
#endif // LIBTENSOR_DEBUG
    }

    /**	\brief Returns the element at the given position (r-value)
        \param pos Position (not to exceed N).
        \throw out_of_bounds If the position exceeds N (LIBTENSOR_DEBUG only).
     **/
    const T &operator[](size_t pos) const {
#ifdef LIBTENSOR_DEBUG
        return at(pos);
#else // LIBTENSOR_DEBUG
        return at_nothrow(pos);
#endif // LIBTENSOR_DEBUG
    }

    //@}

private:
    void check_bounds(size_t pos) const {
        if(pos >= N) {
            throw out_of_bounds(g_ns, k_clazz, "check_bounds(size_t)",
                __FILE__, __LINE__, "pos");
        }
    }

};


/**	\brief Fixed-length array of objects of given type (specialized for zero
        length)
    \tparam T Object type.

    This zero-length sequence is a stub that contains no objects. It provides
    the same interface as the full sequence, but the access methods raise
    out_of_bounds upon attempts go get the elements of the array.

    \sa sequence<N, T>, sequence_test

    \ingroup libtensor_core
 **/
template<typename T>
class sequence<0, T> {
public:
    static const char *k_clazz; //!< Class name

public:
    //!	\name Construction and destruction
    //@{

    /** \copydoc sequence<N,T>::sequence(const T&)
     **/
    sequence(const T &t = T()) {

    }

    /** \copydoc sequence<N,T>::sequence(const sequence<N, T>&)
     **/
    sequence(const sequence<0, T> &seq) {

    }

    //@}


    //! \name Access to %sequence elements, manipulations
    //@{

    /** \copydoc sequence<N,T>::at(size_t)
     **/
    T &at(size_t pos) {
        throw_out_of_bounds();
    }

    /** \copydoc sequence<N,T>::at(size_t) const
     **/
    const T &at(size_t pos) const {
        throw_out_of_bounds();
    }

    /** \copydoc sequence<N,T>::at_nothrow(size_t)
     **/
    T &at_nothrow(size_t pos) {

    }

    /** \copydoc sequence<N,T>::at_nothrow(size_t) const
     **/
    const T &at_nothrow(size_t pos) const {

    }

    //@}


    //! \name Overloaded operators
    //@{

    /** \copydoc sequence<N,T>::operator[](size_t)
     **/
    T &operator[](size_t pos) {
#ifdef LIBTENSOR_DEBUG
        return at(pos);
#else // LIBTENSOR_DEBUG
        return at_nothrow(pos);
#endif // LIBTENSOR_DEBUG
    }

    /** \copydoc sequence<N,T>::operator[](size_t) const
     **/
    const T &operator[](size_t pos) const {
#ifdef LIBTENSOR_DEBUG
        return at(pos);
#else // LIBTENSOR_DEBUG
        return at_nothrow(pos);
#endif // LIBTENSOR_DEBUG
    }

    //@}

private:
    void throw_out_of_bounds() const {
        throw out_of_bounds(g_ns, k_clazz, "throw_out_of_bounds()",
            __FILE__, __LINE__, "pos");
    }

};


template<size_t N, typename T>
const char *sequence<N, T>::k_clazz = "sequence<N, T>";


template<typename T>
const char *sequence<0, T>::k_clazz = "sequence<0, T>";


} // namespace libtensor

#endif // LIBTENSOR_SEQUENCE_H
