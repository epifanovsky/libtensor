#ifndef LIBTENSOR_INDEX_H
#define LIBTENSOR_INDEX_H

#include <iostream>
#include "../defs.h"
#include "../exception.h"
#include "permutation.h"
#include "sequence.h"

namespace libtensor {

template<size_t N>
class index;

/** \brief Prints out an index to an output stream

    \ingroup libtensor_core
 **/
template<size_t N>
std::ostream &operator<<(std::ostream &os, const index<N> &i);

/** \brief Prints out an index to an output stream (specialization for N=0)

    \ingroup libtensor_core
 **/
template<>
std::ostream &operator<<(std::ostream &os, const index<0> &i);


/** \brief Index of a single %tensor element
    \tparam N Index order.

    An %index is a sequence of integers that identifies a single %tensor
    element. The number of integers in the sequence (the order of the
    %index) agrees with the number of %tensor %dimensions. Each integer
    of the sequence gives the position of the element along each dimension.


    <b>Creation of indexes</b>

    A newly created %index object points at the first %tensor element, i.e.
    has zeros along each dimension. To modify the %index, at() or
    operator[] can be used:
    \code
    index<2> i, j;
    i[0] = 2; i[1] = 3;
    j.at(0) = 2; j.at(1) = 3; // operator[] and at() are identical
    \endcode


    <b>Comparison methods</b>

    Two %index objects can be compared using equals(), which returns true
    two indexes identify the same element in a %tensor, and false otherwise:
    \code
    index<2> i, j;
    bool b;
    i[0] = 2; i[1] = 3;
    j[0] = 2; j[1] = 3;
    b = i.equals(j); // b == true
    j[0] = 3; j[1] = 2;
    b = i.equals(j); // b == false
    \endcode

    For convenience, operator== and operator!= are overloaded for indexes.
    Continuing the above code example,
    \code
    j[0] = 2; j[1] = 3;
    if(i == j) {
        // code here will be executed
    }
    if(i != j) {
        // code here will not be executed
    }
    \endcode

    Two non-equal indexes can be put in an ordered sequence using the
    defined comparison operation: each %index element is compared according
    to its seniority, the first element being the most senior, and the
    last element being junior. The comparison is performed with the less()
    method or overloaded operator< and operator>.
    \code
    i[0] = 2; i[1] = 3;
    j[0] = 2; j[1] = 4;
    if(i.less(j)) {
        // code here will be executed
    }
    j[0] = 1; j[1] = 4;
    if(i.less(j)) {
        // code here will not be executed
    }
    \endcode


    <b>Output to a stream</b>

    To print the current value of the %index to an output stream, the
    overloaded operator<< can be used.


    <b>Exceptions</b>

    Methods and operators that require an input position or %index may
    throw an out_of_bounds %exception if supplied input falls out of
    allowable range.


    \ingroup libtensor_core
**/
template<size_t N>
class index : public sequence<N, size_t> {
    friend std::ostream &operator<<
        <N>(std::ostream &os, const index<N> &i);

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Default constructor creates an %index with all zeros
     **/
    index();

    /** \brief Copies the %index from another instance
        \param idx Another %index.
     **/
    index(const index<N> &idx);

    //@}


    //!    \name Access to %sequence elements, manipulations
    //@{

    /** \brief Permutes the %index
        \param p Permutation.
     **/
    index<N> &permute(const permutation<N> &perm);

    //@}


    //!    \name Comparison
    //@{

    /** \brief Checks if two indices are equal

        Returns true if the indices are equal, false otherwise.
     **/
    bool equals(const index<N> &idx) const;

    /** \brief Compares two indexes
        \return true if this %index is smaller than the other one,
            false otherwise.
     **/
    bool less(const index<N> &idx) const;

    //@}


    //!    \name Overloaded operators
    //@{

    /** \brief Compares two indexes (see less())
     **/
    bool operator<(const index<N> &idx) const;

    //@}

};


template<size_t N>
inline index<N>::index() : sequence<N, size_t>(0) {

}

template<size_t N>
inline index<N>::index(const index<N> &idx) : sequence<N, size_t>(idx) {

}

template<size_t N>
inline index<N> &index<N>::permute(const permutation<N> &perm) {

    perm.apply(*this);
    return *this;
}

template<size_t N>
inline bool index<N>::equals(const index<N> &idx) const {

    for(register size_t i = 0; i < N; i++)
        if(sequence<N, size_t>::at_nothrow(i) !=
            idx.sequence<N, size_t>::at_nothrow(i)) return false;
    return true;
}

template<size_t N>
inline bool index<N>::less(const index<N> &idx) const {

    for(register size_t i = 0; i < N; i++) {
        if(sequence<N, size_t>::at_nothrow(i) <
            idx.sequence<N, size_t>::at_nothrow(i)) return true;
        if(sequence<N, size_t>::at_nothrow(i) >
            idx.sequence<N, size_t>::at_nothrow(i)) return false;
    }
    return false;
}

template<size_t N>
inline bool index<N>::operator<(const index<N> &idx) const {

    return less(idx);
}


/** \brief Prints out the index to an output stream

    \ingroup libtensor
 **/
template<size_t N>
std::ostream &operator<<(std::ostream &os, const index<N> &i) {

    os << "[";
    for(size_t j = 0; j != N - 1; j++)
        os << i.sequence<N, size_t>::at_nothrow(j) << ",";
    os << i.sequence<N, size_t>::at_nothrow(N - 1);
    os << "]";
    return os;
}

/** \brief Prints out an index<0> to an output stream

    \ingroup libtensor
 **/
template<>
inline std::ostream &operator<<(std::ostream &os, const index<0> &i) {
    os << "[*]";
    return os;
}

template<size_t N>
inline bool operator==(const index<N> &i1, const index<N> &i2) {
     return i1.equals(i2);
}

template<size_t N>
inline bool operator!=(const index<N> &i1, const index<N> &i2) {
    return ! i1.equals(i2);
}

template<size_t N>
inline bool operator>=(const index<N> &i1, const index<N> &i2) {
    return ! i1.less(i2);
}

template<size_t N>
inline bool operator>(const index<N> &i1, const index<N> &i2) {
    return i2.less(i1);
}

template<size_t N>
inline bool operator<=(const index<N> &i1, const index<N> &i2) {
    return ! i2.less(i1);
}


} // namespace libtensor

#endif // LIBTENSOR_INDEX_H

