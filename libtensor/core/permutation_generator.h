#ifndef LIBTENSOR_PERMUTATION_GENERATOR_H
#define LIBTENSOR_PERMUTATION_GENERATOR_H

#include <vector>
#include "mask.h"

namespace libtensor {

/** \brief Generator for all possible permutations of parts of a sequence.
    \tparam N Length of sequence.
    \tparam T Sequence value type.

    This class implements a slighty modified version of the algorithm by
    H. F. Trotter ("Alg. 115: Perm", Commun. of the ACM 5 (8) 434-435,
    doi:10.1145/368637.368660) to iteratively generate all possible
    permutations of a sequence of values.

    \ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class permutation_generator {
private:
    sequence<N, T> m_seq; //!< Sequence to be permuted
    size_t m_n; //!< Number of indexes to permute (m_n <= N)

    std::vector<size_t> m_map; //!< Map of indexes to be permuted
    std::vector<size_t> m_p; //!< Helper array
    std::vector<bool> m_d; //!< Helper array
    bool m_done; //!< All permutations done

public:
    /** \brief Constructor
        \param seq Sequence to permute.
        \param msk Mask of elements to be permuted.
     **/
    permutation_generator(const sequence<N, T> &seq, const mask<N> &msk) :
        m_seq(seq), m_n(determine_n(msk)), m_map(m_n),
        m_p(m_n, 0), m_d(m_n, true), m_done(false) {

        for (register size_t i = 0, j = 0; i < N; i++) {
            if (! msk[i]) continue;
            m_map[j++] = i;
        }
    }

    /** \brief Obtain the current permutation of the sequence
     **/
    const sequence<N, T> &get_sequence() const { return m_seq; }

    /** \brief Compute next permutation.
        \return True, if this is not the last permutation.
     **/
    bool next() {
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
        std::swap(m_seq[m_map[q]], m_seq[m_map[q + 1]]);

        return (! m_done);
    }

    /** \brief Is this the last permutations.
     **/
    bool is_last() const { return m_done; }

private:
    static size_t determine_n(const mask<N> &msk) {
        size_t n = 0;
        for (register size_t i = 0; i < N; i++) { if (msk[i]) n++; }
        return n;
    }
};


} // namespace libtensor

#endif // LIBTENSOR_PERMUTATION_GENERATOR_H
