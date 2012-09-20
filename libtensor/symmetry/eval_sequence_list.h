#ifndef LIBTENSOR_EVAL_SEQUENCE_LIST_H
#define LIBTENSOR_EVAL_SEQUENCE_LIST_H

#include <vector>
#include <libtensor/core/sequence.h>
#include <libtensor/core/out_of_bounds.h>


namespace libtensor {


template<size_t N>
class eval_sequence_list {
public:
    static const char *k_clazz;

public:
    typedef sequence<N, size_t> eval_sequence_t;

private:
    std::vector<eval_sequence_t> m_list; // List of sequences

public:
    /** \brief Add sequence to list
        \return Returns the index of the sequence in list
     **/
    size_t add(const eval_sequence_t &seq);

    /** \brief Current size of the list
     **/
    size_t size() const { return m_list.size(); }

    /** \brief Checks if the sequence already exists in list
     **/
    bool has_sequence(const eval_sequence_t &seq) const;

    /** \brief Returns the index of the sequence in list
            (returns -1 if not found)
     **/
    size_t get_position(const eval_sequence_t &seq) const;

    /** \brief Index access
     **/
    eval_sequence_t &operator[](size_t n);

    /** \brief Const index access
     **/
    const eval_sequence_t &operator[](size_t n) const;

    /** \brief Clears contents
     **/
    void clear() { m_list.clear(); }
};


template<size_t N>
const char *eval_sequence_list<N>::k_clazz = "eval_sequence_list<N>";


template<size_t N>
size_t eval_sequence_list<N>::add(const eval_sequence_t &seq) {

    size_t i = get_position(seq);
    if (i == (size_t) -1) {
        m_list.push_back(seq);
        return m_list.size() - 1;
    }
    else {
        return i;
    }
}

template<size_t N>
bool eval_sequence_list<N>::has_sequence(const eval_sequence_t &seq) const {

    return get_position(seq) != (size_t) -1;
}


template<size_t N>
size_t eval_sequence_list<N>::get_position(const eval_sequence_t &seq) const {

    size_t seqno = 0;
    for (; seqno < m_list.size(); seqno++) {
        const eval_sequence_t &ref = m_list[seqno];

        register size_t i = 0;
        for (; i < N; i++) {
            if (seq[i] != ref[i]) break;
        }
        if (i == N) return seqno;
    }

    return (size_t) -1;
}


template<size_t N>
sequence<N, size_t> &eval_sequence_list<N>::operator[](size_t n) {

#ifdef LIBTENSOR_DEBUG
    if (n >= m_list.size()) {
        throw out_of_bounds(g_ns, k_clazz, "operator[](size_t)",
                __FILE__, __LINE__, "n");
    }
#endif

    return m_list[n];
}


template<size_t N>
const sequence<N, size_t> &eval_sequence_list<N>::operator[](size_t n) const {

#ifdef LIBTENSOR_DEBUG
    if (n >= m_list.size()) {
        throw out_of_bounds(g_ns, k_clazz, "operator[](size_t)",
                __FILE__, __LINE__, "n");
    }
#endif

    return m_list[n];
}


} // namespace libtensor


#endif // LIBTENSOR_EVAL_SEQUENCE_LIST_H
