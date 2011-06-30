#ifndef LIBTENSOR_LABEL_TARGET_H
#define LIBTENSOR_LABEL_TARGET_H

#include "../../core/sequence.h"
#include "../../exception.h"
#include "../bad_symmetry.h"
#include "product_table_container.h"
#include "product_table_i.h"

namespace libtensor {

/** \brief Subset of a se_label<N, T> %symmetry element.

    A label subset defines allowed blocks with respect to a product table and
    a set of "active" dimensions.

    To achieve this it uses
    - block labels along the active dimensions,
    - a set of intrinsic labels, and
    - an evaluation order
    which can all be set and accessed using the respective member functions.

    The evaluation order defines the order in which a product of
    labels is evaluated to determine if a block is allowed. It consists of
    a sequence of unique values from the interval [0, N] where N represents
    the set of intrinsic labels, while smaller values refer to dimension
    numbers. Thus, only values for "active" dimensions are allowed in the
    evaluation order sequence.

    Allowed blocks are determined as follows:
    - Dimensions not part of the evaluation order are ignored. Thus,
      if the evaluation order sequence is empty, all blocks are allowed.
    - Use the evaluation order to build a sequence of labels from block labels
      and one of the intrinsic labels.
    - If the product of the labels in the sequence contains the label 0, the
      block is allowed.
    - Repeat the last two steps with all other intrinsic labels.

    \ingroup libtensor_symmetry
 **/
template<size_t N>
class label_set {
public:
    static const char *k_clazz; //!< Class name

    typedef product_table_i::label_t label_t; //!< Label type
    typedef product_table_i::label_group label_group; //!< Group of labels
    typedef label_group::const_iterator iterator; //!< Iterator for label_group

private:
    dimensions<N> m_bidims; //!< Block index dimensions
    mask<N> m_msk; //!< Mask of "active" dimensions
    const product_table_i &m_pt; //!< Associated product table

    sequence<N, size_t> m_type; //!< Dimension types
    sequence<N, label_group*> m_blk_labels; //!< Block labels of dimension type
    label_group m_intr_labels; //!< Group of intrinsic labels

    sequence<N + 1, size_t> m_eval_order; //!< Evaluation order
    size_t m_eval_size; //!< Number of evaluation indexes

public:
    //! \name Constructor and destructors
    //@{
    /** \brief Constructor
        \param bidims Block index dimensions
        \param msk Mask of active dimensions
        \param id Product table id.
     **/
    label_set(const dimensions<N> &bidims, const mask<N> &msk,
            const std::string &id);

    /** \brief Copy constructor
     **/
    label_set(const label_set<N> &set);

    /** \brief Destructor
     **/
    ~label_set();
    //@}

    //! \name Manipulating functions
    //@{
    /** \brief Assign label to block indexes given by %mask and position
        \param msk Dimension mask
        \param blk Block position
        \param label Block label
     **/
    void assign(const mask<N> &msk, size_t blk, label_t label);

    /** \brief Add label to the list of intrinsic labels
     **/
    void add_intrinsic(label_t l);

    /** \brief Initializes the order sequence with the default ordering

        The default ordering is \f$ {i_0, i_1, .. i_X, N} \f$ where
        \f$ i_0 < i_1 < ...\f$ refer to the active dimension numbers.
     **/
    void set_default_order();

    /** \brief Append an index to the evaluation order sequence.
        \param idx Index to be added.

        The index must not yet exist in the evalutation order.
        Valid indexes idx are from the interval [0, N] where \c idx==N refers
        to the intrinsic label.
     **/
    void append(size_t idx);

    /** \brief Permute the indexes
        \param p Permutation
     **/
    void permute(const permutation<N> &p);

    /** \brief Clear the block labels, intrinsic labels and evaluation order.
     **/
    void clear();

    /** \brief Clear the list of intrinsic labels
     **/
    void clear_intrinsic() { m_intr_labels.clear(); }

    /** \brief Clear the evaluation order.
     **/
    void clear_order() { m_eval_size = 0; }
    //@}

    //! \name Access functions
    //@{
    /** \brief Get the block index dimensions
     **/
    const dimensions<N> &get_block_index_dims() const { return m_bidims; }

    /** \brief Get the index mask
     **/
    const mask<N> &get_mask() const { return m_msk; }

    /** \brief Get the ID of the associated product table
     **/
    const std::string &get_table_id() const { return m_pt.get_id(); }

    /** \brief Get the type ID of a dimension
        \param dim Dimension
        \return Type ID
     **/
    size_t get_dim_type(size_t dim) const;

    /** \brief Get the block label given by type ID and block number
        \param type Type ID
        \param blk Block number
        \return Block label
     **/
    label_t get_label(size_t type, size_t blk) const;

    /** \brief Get the size of the evaluation order sequence
     **/
    size_t get_order_size() const { return m_eval_size; }

    /** \brief Get the i-th element in the evaluation order sequence
     **/
    size_t operator[](size_t i) const;
    //@}

    //! \name STL-like iterator over intrinsic labels
    //@{
    iterator begin() { m_intr_labels.begin(); }
    iterator end() { m_intr_labels.end(); }

    label_t get_intrinsic(iterator it) const { return *it; }
    //@}

    /** \brief Tests, if the two sets overlap
     **/
    bool has_overlap(const label_set<N> &set) const;

    /** \brief Tests, if the block is allowed w.r.t to this label set
     **/
    bool is_allowed(const index<N> &bidx) const;
};

template<size_t N>
const char *label_set<N>::k_clazz = "label_set<N>";

template<size_t N>
label_set<N>::label_set(const dimensions<N> &bidims, const mask<N> &msk,
        const std::string &id) : m_bidims(bidims), m_msk(msk),
    m_pt(product_table_container::get_instance().req_const_table(id)),
    m_type(0), m_blk_labels(0), m_intr_labels(0),
    m_eval_order(0), m_eval_size(0) {

    size_t cur_type = 0;
    for (register size_t i = 0; i < N; i++) {
        if (! m_msk[i]) continue;

        m_type[i] = cur_type;
        m_blk_labels[cur_type] = new label_group(m_bidims[i], m_pt.invalid());

        for (register size_t j = i + 1; j < N; j++) {
            if (! m_msk[i]) continue;

            if (m_bidims[i] == m_bidims[j]) {
                m_type[j] = cur_type;
            }
        }
        cur_type++;
    }
}

template<size_t N>
label_set<N>::label_set(const label_set<N> &set) : m_bidims(set.bidims),
    m_msk(set.m_msk), m_type(set.m_type), m_blk_labels(0),
    m_intr_labels(set.m_intr_labels), m_eval_order(set.m_eval_order),
    m_eval_size(set.m_eval_size),
    m_pt(product_table_container::get_instance().req_const_table(
            set.m_pt.get_id())) {

    for (register size_t i = 0; i < N && set.m_blk_labels[i] != 0; i++) {

        m_blk_labels[i] = new label_group(*(set.m_blk_labels[i]));
    }
}

template<size_t N>
label_set<N>::~label_set() {

    clear();
    product_table_container::get_instance().ret_table(m_pt.get_id());
}

template<size_t N>
void label_set<N>::assign(const mask<N> &msk, size_t blk, label_t l) {

    static const char *method = "assign(const mask<N> &, size_t, label_t)";

#ifdef LIBTENSOR_DEBUG
    if ((msk & m_msk) != msk) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "msk");
    }
#endif

    register size_t i = 0;
    for (; i < N; i++)  if (msk[i]) break;
    if (i == N) return; // mask has no true component

    size_t type = m_type[i];

#ifdef LIBTENSOR_DEBUG
    // Test if position is out of bounds
    if (blk >= m_bidims[i]) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__, "blk");
    }

    // Test if all masked indexes are of the same type
    for (register size_t j = i + 1; j < N; j++) {
        if (! msk[j]) continue;
        if (m_type[j] == type) continue;

        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "msk");
    }

    // Test if the label is valid for the product table
    if (! m_pt.is_valid(l)) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__, "l");
    }
#endif

    // Test if there are dimensions included in the type that are not part
    // of the mask
    bool adjust = false;
    for(i = 0; i < N; i++) {
        if (msk[i]) continue;
        if (m_type[i] == type) { adjust = true; break; }
    }

    // If yes, split dimension type into two
    size_t cur_type = type;
    if (adjust) {
        for (i = 0; i < N; i++) if (m_blk_labels[i] == 0) break;
        cur_type = i;
        m_blk_labels[cur_type] =
                new label_group(*(m_blk_labels[type]));

        // Assign all masked indexes to the new type.
        for (i = 0; i < N; i++) {
            if (msk[i]) m_type[i] = cur_type;
        }
    }

    // Set the new block label
    m_blk_labels[cur_type]->at(blk) = l;
}

template<size_t N>
void label_set<N>::add_intrinsic(label_t l) {

#ifdef LIBTENSOR_DEBUG
    if (! m_pt.is_valid(l)) {
        throw bad_parameter(g_ns, k_clazz, "add_intrinsic(label_t)",
                __FILE__, __LINE__, "l.");
    }
#endif

    label_group::iterator it = m_intr_labels.begin();
    for (; it != m_intr_labels.end() && l < *it; it++) { }

    if (it == m_intr_labels.end()) {
        m_intr_labels.push_back(l);
        return;
    }
    if (*it == l) return;

    m_intr_labels.insert(it, l);
}

template<size_t N>
void label_set<N>::set_default_order() {

    size_t ii = 0;
    for (size_t i = 0; i < N; i++) {
        if (! m_msk[i]) continue;
        m_eval_order[ii++] = i;
    }
    m_eval_order[ii++] = N;
    m_eval_size = ii;
}

template<size_t N>
void label_set<N>::append(size_t idx) {

#ifdef LIBTENSOR_DEBUG
    static const char *method = "append(size_t)";

    if (idx > N || ! m_msk[idx]) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "idx.");
    }

    for (size_t i = 0; i < m_eval_size; i++) {
        if (m_eval_order[i] == idx) {
            throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "Duplicate index.");
        }
    }

    size_t max_size = 0;
    for (size_t i = 0; i < N; i++) if (m_msk[i]) max_size++;
    max_size++;

    if (m_eval_size >= max_size) {
        throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Evaluation order complete.");
    }
#endif

    m_eval_order[m_eval_size] = idx;
    m_eval_size++;
}

template<size_t N>
void label_set<N>::permute(const permutation<N> &p) {

    sequence<N, size_t> seqa;
    for (size_t i = 0; i < N; i++) seqa[i] = i;
    p.permute(seqa);

    bool affected = false;
    for (size_t i = 0; i < N; i++) {
        if (m_msk[i] && seqa[i] != i) { affected = true; break; }
    }
    if (! affected) return;

    m_msk.permute(p);
    m_bidims.permute(p);
    m_type.permute(p);
    for (size_t i = 0; i < m_eval_size; i++) {
        if (m_eval_order[i] == N) continue;
        m_eval_order[i] = seqa[m_eval_order[i]];
    }
}

template<size_t N>
void label_set<N>::clear() {

    clear_intrinsic();
    clear_order();

    for (register size_t i = 0; i < N && m_blk_labels[i] != 0; i++) {
        delete m_blk_labels[i]; m_blk_labels[i] = 0;
    }
}

template<size_t N>
size_t label_set<N>::get_dim_type(size_t dim) const {

#ifdef LIBTENSOR_DEBUG
    static const char *method = "get_dim_type(size_t)";
    if (dim > N || ! m_msk[dim]) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "dim");
    }
#endif

    return m_type[dim];
}

template<size_t N>
typename label_set<N>::label_t label_set<N>::get_label(size_t type,
        size_t blk) const {

#ifdef LIBTENSOR_DEBUG
    static const char *method = "get_label(size_t, size_t)";

    if (type > N || m_blk_labels[type] == 0) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "dim");
    }
    if (m_blk_labels->size() >= blk) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "blk");
    }
#endif

    return m_blk_labels[type]->at(blk);
}

template<size_t N>
size_t label_set<N>::operator[](size_t i) const {

#ifdef LIBTENSOR_DEBUG
    static const char *method = "operator[](size_t)";

    if (i >= m_eval_size) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "i");
    }
#endif

    return m_eval_order[i];
}

template<size_t N>
bool label_set<N>::has_overlap(const label_set<N> &set) const {

    for (size_t i = 0; i < N; i++) {
        if (m_msk[i] && set.m_msk[i]) return true;
    }
    return false;
}

template<size_t N>
bool label_set<N>::is_allowed(const index<N> &idx) const {

#ifdef LIBTENSOR_DEBUG
    static const char *method = "is_allowed(const index<N> &)";

    // Test, if index is valid block index
    for (size_t i = 0; i < N; i++) {
        if (idx[i] < m_bidims[i]) continue;

        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                "bidx.");
    }

#endif

    // No evaluation order means the block is allowed
    if (m_eval_size == 0) return true;

    label_group lg(m_eval_size, m_pt.invalid());
    // Determine the order position of the intrinsic labels
    // and assign the respective block labels to the other elements of lg
    size_t iintr = (size_t) -1;
    for (size_t i = 0; i < m_eval_size; i++) {
        if (m_eval_order[i] == N) { iintr = i; continue; }

        label_group &labels = *(m_blk_labels[m_type[m_eval_order[i]]]);
        lg[i] = labels[idx[m_eval_order[i]]];

        // If one of the block labels is invalid, the block is allowed
        if (! m_pt.is_valid(lg[i])) return true;
    }

    // If the intrinsic label is not part of the evaluation order, the product
    // of the block labels needs to contain 0
    if (iintr == (size_t) -1) {
        return m_pt.is_in_product(lg, 0);
    }

    // Otherwise test, if the set of intrinsic labels comprises all labels
    if (m_intr_labels.size() == m_pt.nlabels()) return true;

    // Otherwise loop over all intrinsic labels
    for (label_group::const_iterator it = m_intr_labels.begin();
            it != m_intr_labels.end(); it++) {

        lg[iintr] = *it;
        if (m_pt.is_in_product(lg, 0)) return true;
    }

    return false;
}

} // namespace libtensor

#endif

