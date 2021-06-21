#ifndef LIBTENSOR_PERMUTATION_H
#define LIBTENSOR_PERMUTATION_H

#include <cstdio>
#include <iostream>
#include <string>
#include "../defs.h"
#include "../exception.h"
#include "sequence.h"
#include "out_of_bounds.h"

namespace libtensor {


template<size_t N> class mask;


/** \brief Permutation of a set

    <b>Overview</b>

    A permutation object contains a permuted order or %tensor indices
    starting with 0. The identity %permutation doesn't change the order
    of the indexes.

    For example, a fourth-order %tensor %index \e ijkl and an identity
    %permutation (0123). The %permutation elements are indexes of the
    permuted entity in the unpermuted entity. The identity %permutation
    therefore doesn't change the order of the indexes. If we swap the first
    two indices, the %permutation becomes (1023), and the %tensor %index
    becomes \e jikl. If we then switch the first two pairs of indices, the
    %permutation becomes (2310), and the %index is \e klji.

    This example shows the above operations and prints out the resulting
    %index.
    \code
    char idx[5], perm_idx[5];
    strcpy(idx, "ijkl");
    strcpy(perm_idx, "ijkl");
    permutation p(4);

    // Operations on the permutation p
    p.permute(0,1); // jikl
    p.permute(0,2).permute(1,3); // klji

    p.apply(4, idx, perm_idx);
    printf("%s -> %s", idx, perm_idx); // prints ijkl -> klji
    \endcode

    <b>Inverse permutations</b>

    The inverse %permutation applied to the original %permutation yields
    the identity %permutation. The inverse of the identity %permutation or
    an elementary %permutation (%permutation of just two indexes) is the
    %permutation itself.
    \f[ \mathcal{P} \mathcal{P}^{-1} = 1 \f]

    To obtain the inverse %permutation, the copy constructor or the
    invert() method can be used:
    \code
    permutation p(4);

    // Operations on the permutation p
    p.permute(0,1); // ijkl -> jikl
    p.permute(0,2).permute(1,3); // jikl -> klji

    permutation pc(p); // Plain copy
    permutation pi(p, true); // Inverse copy

    pc.invert(); // Now contains inverse permutation

    bool b_equals = pc.equals(pi); // true

    p.permute(pi); // p is now a unit permutation
    bool b_unit = p.is_identity(); // true
    \endcode

    \ingroup libtensor_core
 **/
template<size_t N>
class permutation {
public:
    static const char *k_clazz; //!< Class name

private:
    size_t m_idx[N]; //!< Permuted indices

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Creates a unit %permutation of a specified order
     **/
    permutation();

    /** \brief Creates a copy or an inverted copy of a %permutation

        Creates a copy or an inverted copy of a %permutation.
        \param p Another %permutation.
        \param b_inverse Create inverse %permutation.
     **/
    permutation(const permutation<N> &p, bool b_inverse = false);

    //@}

    //!    \name Manipulations with the %permutation
    //@{

    /** \brief Accumulates %permutation

        Permutes this %permutation with another %permutation.

        \param p Another %permutation.
        \return Reference to this %permutation.
     **/
    permutation<N> &permute(const permutation<N> &p);

    /** \brief Permutes two indexes
        \param i First %index.
        \param j Second %index.
        \return Reference to this %permutation.
        \throw out_of_bounds If either of the indices is out of bounds.
     **/
    permutation<N> &permute(size_t i, size_t j);

    /** \brief Inverts the %permutation
        \return Reference to this %permutation.
     **/
    permutation<N> &invert();

    /** \brief Resets %permutation (makes it an identity %permutation)
     **/
    void reset();

    /** \brief Applies a %mask to the permutation

        The %permutation is adjusted such that the elements of a set
        that correspond to set indexes in the mask will not change
        their positions upon the action of this %permutation on the set.
     **/
    void apply_mask(const mask<N> &msk);

    //@}

    //! \name Access functions
    //@{

    /** \brief Access the i-th element of the permutation sequence
     **/
    const size_t &operator[](size_t i) const;

    //@}

    //!    \name Special comparisons
    //@{

    /** \brief Checks if the %permutation is an identity %permutation

        Checks if the %permutation is an identity %permutation.
        An identity %permutation doesn't change the order in
        a sequence.

        \return true if this is an identity %permutation, false
        otherwise.
     **/
    bool is_identity() const;

    /** \brief Checks if two permutations are identical
        \return true if the two permutations are equal, false otherwise.
     **/
    bool equals(const permutation<N> &p) const;

    //@}


    //!    \name Application of the %permutation
    //@{

    /** \brief Permutes a sequence of objects
        \param seq Sequence of objects.
     **/
    template<typename T>
    void apply(sequence<N, T> &seq) const;

    //@}


    //!    \name Overloaded operators
    //@{

    bool operator==(const permutation<N> &p) const;
    bool operator!=(const permutation<N> &p) const;
    bool operator<(const permutation<N> &p) const;

    //@}

};


template<>
class permutation<0> {
public:
    permutation() { }
    permutation(const permutation<0> &p, bool inverse = false) { }
};


template<size_t N>
const char *permutation<N>::k_clazz = "permutation<N>";


template<size_t N>
inline permutation<N>::permutation() {

    for(size_t i = 0; i < N; i++) m_idx[i] = i;
}


template<size_t N>
inline permutation<N>::permutation(const permutation<N> &p, bool b_inverse) {

    if(b_inverse) {
        for(size_t i = 0; i < N; i++) m_idx[p.m_idx[i]] = i;
    } else {
        for(size_t i = 0; i < N; i++) m_idx[i] = p.m_idx[i];
    }
}


template<size_t N>
inline permutation<N> &permutation<N>::permute(const permutation<N> &p) {

    size_t idx_cp[N];
    for(size_t i = 0; i < N; i++) idx_cp[i] = m_idx[i];
    for(size_t i = 0; i < N; i++) m_idx[i] = idx_cp[p.m_idx[i]];
    return *this;
}


template<size_t N>
inline permutation<N> &permutation<N>::permute(size_t i, size_t j) {

#ifdef LIBTENSOR_DEBUG
    if(i >= N || j >= N) {
        throw out_of_bounds(g_ns, k_clazz, "permute(size_t, size_t)",
            __FILE__, __LINE__, "Index out of range.");
    }
#endif // LIBTENSOR_DEBUG
    if(i == j) return *this;
    size_t i_cp = m_idx[i];
    m_idx[i] = m_idx[j];
    m_idx[j] = i_cp;
    return *this;
}


template<size_t N>
inline permutation<N> &permutation<N>::invert() {

    size_t idx_cp[N];
    for(size_t i = 0; i < N; i++) idx_cp[i] = m_idx[i];
    for(size_t i = 0; i < N; i++) m_idx[idx_cp[i]] = i;
    return *this;
}


template<size_t N>
inline void permutation<N>::reset() {

    for(size_t i = 0; i < N; i++) m_idx[i] = i;
}


template<size_t N>
inline void permutation<N>::apply_mask(const mask<N> &msk) {

    size_t i = 0;
    while(i < N) {
        if(i != m_idx[i] && msk[i]) {
            permute(i, m_idx[i]);
            i = 0;
        } else {
            i++;
        }
    }
}

template<size_t N>
inline const size_t &permutation<N>::operator[](size_t i) const {
#ifdef LIBTENSOR_DEBUG
    if(i >= N) {
        throw out_of_bounds(g_ns, k_clazz, "operator[](size_t)",
            __FILE__, __LINE__, "Index out of range.");
    }
#endif // LIBTENSOR_DEBUG
    return m_idx[i];
}


template<size_t N>
inline bool permutation<N>::is_identity() const {

    for(size_t i = 0; i < N; i++)
        if(m_idx[i] != i) return false;
    return true;
}


template<size_t N>
inline bool permutation<N>::equals(const permutation<N> &p) const {

    if(&p == this) return true;
    for(size_t i = 0; i < N; i++)
        if(m_idx[i] != p.m_idx[i]) return false;
    return true;
}


template<size_t N> template<typename T>
void permutation<N>::apply(sequence<N, T> &seq) const {

    sequence<N, T> buf(seq);
    for(size_t i = 0; i < N; i++) seq[i] = buf[m_idx[i]];
}


template<size_t N>
inline bool permutation<N>::operator==(const permutation<N> &p) const {

    return equals(p);
}


template<size_t N>
inline bool permutation<N>::operator!=(const permutation<N> &p) const {

    return !equals(p);
}


template<size_t N>
inline bool permutation<N>::operator<(const permutation<N> &p) const {

    for(size_t i = 0; i < N; i++) {
        if(m_idx[i] != p.m_idx[i]) return m_idx[i] < p.m_idx[i];
    }
    return false;
}


/** \brief Prints out a permutation to an output stream

    \ingroup libtensor_core
**/
template<size_t N>
std::ostream &operator<<(std::ostream &os, const permutation<N> &p) {

    static const char *alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    sequence<N, char> seq1('\0'), seq2('\0');
    for(size_t i = 0; i < N; i++) seq1[i] = seq2[i] = alphabet[i];
    p.apply(seq2);
    std::string s1(N, ' '), s2(N, ' ');
    for(size_t i = 0; i < N; i++) {
        s1[i] = seq1[i];
        s2[i] = seq2[i];
    }
    os << "[" << s1 << "->" << s2 << "]";
    return os;
}


} // namespace libtensor

#endif // LIBTENSOR_PERMUTATION_H

