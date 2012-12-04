#ifndef LIBTENSOR_BLOCK_LIST_H
#define LIBTENSOR_BLOCK_LIST_H

#include <cstring> // for size_t
#include <algorithm>
#include <vector>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/dimensions.h>
#include <libtensor/core/index.h>

namespace libtensor {


/** \brief Unique list of block in a block tensor

    XXX: In future: move to core part of library? base orbit_list on block_list?

    \ingroup libtensor_gen_bto
 **/
template<size_t N>
class block_list {
public:
    static const char *k_clazz; //!< Class name

public:
    typedef typename std::vector<size_t>::const_iterator iterator;

private:
    dimensions<N> m_dims; //!< Block index dimensions
    mutable std::vector<size_t> m_blks; //!< List of blocks
    mutable bool m_sorted; //!< Whether the list is sorted

public:
    /** \brief Initializes an empty list
        \param bidims Block index dimensions.
     **/
    block_list(const dimensions<N> &bidims) :
        m_dims(bidims), m_sorted(true)
    { }

    /** \brief Initializes and fills the list
        \param bidims Block index dimensions.
        \param blst Vector of block indexes.
     **/
    block_list(const dimensions<N> &bidims, const std::vector<size_t> &blst) :
        m_dims(bidims), m_blks(blst), m_sorted(false)
    { }

    /** \brief Add a block to the list
        \param absidx Absolute index of block
     **/
    void add(size_t aidx);

    /** \brief Add a block to the list
        \param idx Index of block
     **/
    void add(const index<N> &idx) {
        add(abs_index<N>::get_abs_index(idx, m_dims));
    }

    /** \brief Empties the list
     **/
    void clear() {
        m_blks.clear();
    }

    /** \brief Return the current number of blocks in list
     **/
    size_t get_size() const { return m_blks.size(); }

    /** \brief Return the block index dimensions
     **/
    const dimensions<N> &get_dims() const {
        return m_dims;
    }

    /** \brief Does the list contain a given block
        \param absidx Absolute index of block to find in list
     **/
    bool contains(size_t aidx) const;

    /** \brief Does the list contain a certain block
        \param idx Index of block to find in list
     **/
    bool contains(const index<N> &idx) const {
        return contains(abs_index<N>::get_abs_index(idx, m_dims));
    }

    /** \brief Iterator to first block in list
     **/
    iterator begin() const {
        return m_blks.begin();
    }

    /** \brief Iterator to end of list
     **/
    iterator end() const {
        return m_blks.end();
    }

    /** \brief Obtain absolute index of current block
     **/
    size_t get_abs_index(iterator &i) const {
        return *i;
    }

    /** \brief Obtain index of current block
        \param i Iterator
        \param[out] idx Index of block iterator points to
     **/
    void get_index(iterator &i, index<N> &idx) const {
        abs_index<N>::get_index(*i, m_dims, idx);
    }

    void sort() {
        std::sort(m_blks.begin(), m_blks.end());
        m_sorted = true;
    }

};


template<size_t N>
const char *block_list<N>::k_clazz = "block_list<N>";


template<size_t N>
inline void block_list<N>::add(size_t aidx) {

#ifdef LIBTENSOR_DEBUG
    if (aidx >= m_dims.get_size()) {
        throw out_of_bounds(g_ns, k_clazz, "add(size_t)", __FILE__, __LINE__,
                "absidx");
    }
#endif

    m_blks.push_back(aidx);
    if(m_sorted && m_blks.size() > 1) {
        size_t n = m_blks.size();
        m_sorted = (m_blks[n - 2] < m_blks[n - 1]);
    }
}


template<size_t N>
inline bool block_list<N>::contains(size_t aidx) const {

    if(!m_sorted) {
        std::sort(m_blks.begin(), m_blks.end());
        m_sorted = true;
    }
    return std::binary_search(m_blks.begin(), m_blks.end(), aidx);
}


} // namespace libtensor

#endif // LIBTENSOR_BLOCK_LIST_H
