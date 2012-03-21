#ifndef LIBTENSOR_PERMUTATION_GENERATOR_H
#define LIBTENSOR_PERMUTATION_GENERATOR_H

#include <vector>
#include "mask.h"

namespace libtensor {

/** \brief Generator for all possible permutations of a sequence.

    This class implements a slighty modified version of the algorithm by
    H. F. Trotter ("Alg. 115: Perm", Commun. of the ACM 5 (8) 434-435,
    doi:10.1145/368637.368660) to iteratively generate all possible
    permutations of a sequence of values.

    \ingroup libtensor_symmetry
 **/
class permutation_generator {
private:
    std::vector<size_t> m_perm; //!< Current permutation of [0 .. (m_n - 1)]
    std::vector<size_t> m_p; //!< Helper array
    std::vector<bool> m_d; //!< Helper array
    bool m_done; //!< All permutations done

public:
    /** \brief Constructor
        \param seq Sequence to permute.
        \param msk Mask of elements to be permuted.
     **/
    permutation_generator(size_t n) :
        m_perm(n, 0), m_p(n, 0), m_d(n, true), m_done(false) {

        for (register size_t i = 0; i < m_perm.size(); i++) {
            m_perm[i] = i;
        }
    }

    /** \brief Obtain the current permutation of the sequence
     **/
    const size_t &operator[](size_t i) const { return m_perm[i]; }

    /** \brief Compute next permutation.
        \return True, if this is not the last permutation.
     **/
    bool next() {
        if (m_done) return false;

        register size_t i = m_perm.size() - 1, k = 0;
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
        std::swap(m_perm[q], m_perm[q + 1]);

        return (! m_done);
    }

    /** \brief Is this the last permutations.
     **/
    bool is_last() const { return m_done; }
};


} // namespace libtensor

#endif // LIBTENSOR_PERMUTATION_GENERATOR_H
