#ifndef LIBTENSOR_PERMUTATION_GENERATOR_H
#define LIBTENSOR_PERMUTATION_GENERATOR_H


#include <vector>
#include "mask.h"


namespace libtensor {


/** \brief Generator for permutations of N items.
    \tparam N Number of items.

    This class implements a slighty modified version of the algorithm by
    H. F. Trotter ("Alg. 115: Perm", Commun. of the ACM 5 (8) 434-435,
    doi:10.1145/368637.368660) to iteratively generate all possible
    permutations of a sequence of N items.

    A masked can be passed to the constructor of the class to restrict the
    items which are permuted. The m items (m<=N) for which the mask is set
    to \c true stay fixed. The number of permutations generated is then
    \f$ (N-m)!. \f$

    The generated permutations in subsequent steps only differ by one pair
    permutation of successive items.

    \ingroup libtensor_symmetry
 **/
template<size_t N>
class permutation_generator {
private:
    size_t m_n; //!< Length of subsequence
    sequence<N, size_t> m_map; //! Map of indexes to be permuted
    sequence<N, size_t> m_p; //!< Helper array
    mask<N> m_d; //!< Helper array
    permutation<N> m_perm; //!< Current permutation of [0 .. (m_n - 1)]
    bool m_done; //!< All permutations done

public:
    /** \brief Default constructor
     **/
    permutation_generator();

    /** \brief Constructor
        \param seq Sequence to permute.
        \param msk Mask to restrict the indexes to be permuted

        Only indexes for which msk is false will be permuted.
     **/
    permutation_generator(const mask<N> &msk);

    /** \brief Obtain the current permutation
     **/
    const permutation<N> &get_perm() const { return m_perm; }

    /** \brief Compute next permutation.
        \return True, if this is not the last permutation.
     **/
    bool next();

    /** \brief Is this the last permutations.
     **/
    bool is_last() const { return m_done; }
};


template<size_t N>
permutation_generator<N>::permutation_generator() :
    m_map(0), m_p(0), m_done(false), m_n(N) {

    for (register size_t i = 0; i < N; i++) {
        m_map[i] = i;
        m_d[i] = true;
    }
}


template<size_t N>
permutation_generator<N>::permutation_generator(const mask<N> &msk) :
    m_map(N), m_p(0), m_done(false) {

    register size_t j = 0;
    for (register size_t i = 0; i < N; i++) {
        if (! msk[i]) m_map[j++] = i;
        m_d[i] = true;
    }
    m_n = j;
}


template<size_t N>
bool permutation_generator<N>::next() {

    if (m_done) return false;

    register size_t i = m_n - 1, k = 0;
    for (; i > 0; i--) {
        if (m_d[i]) { m_p[i]++; } else { m_p[i]--; }
        if (m_p[i] == (i + 1)) {
            m_d[i] = false;
            continue;
        }
        if (m_p[i] != 0) break;
        m_d[i] = true;
        k++;
    }
    if (i == 0) m_done = true;

    size_t q = (m_done ? k : m_p[i] - 1 + k);
    m_perm.permute(m_map[q], m_map[q+1]);

    return (! m_done);
}


} // namespace libtensor

#endif // LIBTENSOR_PERMUTATION_GENERATOR_H
