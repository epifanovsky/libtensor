#ifndef LIBTENSOR_SEQUENCE_GENERATOR_H
#define LIBTENSOR_SEQUENCE_GENERATOR_H

#include <vector>
#include <libtensor/exception.h>

namespace libtensor {

/** \brief Generates all ordered sequences of n numbers in the interval
        \f$ \left[0:m\right[ \f$ .

    Constructs all possible sequences of n ordered non-identical numbers
    from the interval \f$ \left[0:m\right[ \f$ starting with
    \f$ \left[0,1,2, ..., n\right] \f$ .

    \ingroup libtensor_core
 **/
class sequence_generator {
private:
    std::vector<size_t> m_seq; //!< Sequence of numbers
    size_t m_nmax; //!< Upper limit of interval
    bool m_done; //!< Flag if done.

public:
    /** \brief Constructor
        @param n Number of elements in the sequence
        @param m Upper limit of values
     **/
    sequence_generator(size_t n, size_t m) :
        m_seq(n, 0), m_nmax(m), m_done(false) {

#ifdef LIBTENSOR_DEBUG
        if (n > m)
            throw bad_parameter(g_ns, "sequence_generator",
                    "sequence_generator(size_t, size_t)",
                    __FILE__, __LINE__, "n > m");
#endif // LIBTENSOR_DEBUG

        for (size_t i = 0; i < n; i++) m_seq[i] = i;
        if (n == m || n == 0) m_done = true;
    }

    /** \brief Return current sequence
     **/
    const std::vector<size_t> &get_seq() const { return m_seq; }

    /** \brief Determine next sequence
        @return True if successful, false if last possible combination reached
     **/
    bool next() {

        if (m_done) return false;

        register size_t j = 1;
        for (; j < m_seq.size(); j++) {
            if (m_seq[j] - m_seq[j - 1] > 1) break;
        }
        j--;
        m_seq[j]++;

        if (j == m_seq.size() - 1 && m_seq[j] >= m_nmax) {
            m_done = true;
            m_seq[j]--;
            return false;
        }

        for (register size_t i = 0; i < j; i++) m_seq[i] = i;
        return true;
    }

    /** \brief Checks if all sequences have been constructed
     **/
    bool is_last() { return m_done; }
};


} // namespace libtensor


#endif // LIBTENSOR_SEQUENCE_GENERATOR_H

