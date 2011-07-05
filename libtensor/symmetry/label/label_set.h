#ifndef LIBTENSOR_LABEL_SET_H
#define LIBTENSOR_LABEL_SET_H

#include "../../core/dimensions.h"
#include "../../core/mask.h"
#include "../../exception.h"
#include "../bad_symmetry.h"
#include "product_table_container.h"
#include "product_table_i.h"

namespace libtensor {

/** \brief Subset of a se_label<N, T> %symmetry element.

    A label subset defines allowed blocks with respect to a product table and
    a set of dimensions belonging to the given product table.

    To achieve this it uses
    - block labels along the active dimensions,
    - a set of intrinsic labels, and
    - an evaluation mask
    which can all be set and accessed using the respective member functions.

    The evaluation mask defines which dimensions participate in the evaluation
    of a product of labels to determine if a block is allowed. Thus, only
    dimensions belonging to the product table can be part of the evaluation
    mask. On construction the evaluation mask is set to the mask of dimensions.

    Allowed blocks are determined as follows:
    - Dimensions part of the label set, but not part of the evaluation mask
      are ignored. Thus, if the evaluation mask does not contain any dimensions
      the intrinsic label is used for determining the allowed blocks.
    - If the intrinsic labels comprise all labels, all blocks are allowed.
    - If one of the block indexes is unlabeled, the block is allowed.
    - Build a label sequence from the evaluation mask and one intrinsic label
    - If the product of labels contains label 0, the block is allowed.
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
    mask<N> m_msk; //!< Mask of dimensions belonging to the current
    const product_table_i &m_pt; //!< Associated product table

    sequence<N, size_t> m_type; //!< Dimension types
    sequence<N, label_group*> m_blk_labels; //!< Block labels of dimension type
    label_group m_intr_labels; //!< Group of intrinsic labels

    mask<N> m_eval_msk; //!< Evaluation mask
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

    /** \brief Set the evaluation mask.
        \param emsk Mask.
     **/
    void set_eval_msk(const mask<N> &emsk);

    /** \brief Permute the indexes
        \param p Permutation
     **/
    void permute(const permutation<N> &p);

    /** \brief Minimize the number of label types...
     **/
    void match_blk_labels();

    /** \brief Clear the list of intrinsic labels
     **/
    void clear_intrinsic() { m_intr_labels.clear(); }

    /** \brief Clear the evaluation mask.
     **/
    void clear_eval();
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

    /** \brief Get the i-th element in the evaluation order sequence
     **/
    const mask<N> &get_eval_msk() const { return m_eval_msk; }
    //@}

    //! \name STL-like iterator over intrinsic labels
    //@{
    iterator begin() { return m_intr_labels.begin(); }
    iterator end() { return m_intr_labels.end(); }

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
    m_type((size_t) -1), m_blk_labels(0), m_intr_labels(0),
    m_eval_msk(msk), m_eval_size(0) {

    size_t cur_type = 0;
    for (register size_t i = 0; i < N; i++) {
        if (! m_msk[i]) continue;
        if (m_type[i] != -1) continue;

        m_type[i] = cur_type;
        m_blk_labels[cur_type] = new label_group(m_bidims[i], m_pt.invalid());
        m_eval_size++;

        for (register size_t j = i + 1; j < N; j++) {
            if (! m_msk[j]) continue;

            if (m_bidims[i] == m_bidims[j]) {
                m_type[j] = cur_type;
                m_eval_size++;
            }
        }
        cur_type++;
    }
}

template<size_t N>
label_set<N>::label_set(const label_set<N> &set) : m_bidims(set.m_bidims),
    m_msk(set.m_msk), m_type(set.m_type), m_blk_labels(0),
    m_intr_labels(set.m_intr_labels), m_eval_msk(set.m_eval_msk),
    m_eval_size(set.m_eval_size),
    m_pt(product_table_container::get_instance().req_const_table(
            set.m_pt.get_id())) {

    for (register size_t i = 0; i < N && set.m_blk_labels[i] != 0; i++) {

        m_blk_labels[i] = new label_group(*(set.m_blk_labels[i]));
    }
}

template<size_t N>
label_set<N>::~label_set() {

    clear_intrinsic();
    clear_eval();

    for (register size_t i = 0; i < N && m_blk_labels[i] != 0; i++) {
        delete m_blk_labels[i]; m_blk_labels[i] = 0;
    }
    product_table_container::get_instance().ret_table(m_pt.get_id());
}

template<size_t N>
void label_set<N>::assign(const mask<N> &msk, size_t blk, label_t l) {

    static const char *method = "assign(const mask<N> &, size_t, label_t)";

#ifdef LIBTENSOR_DEBUG
    for (register size_t k = 0; k < N; k++) {
        if (msk[k] && ! m_msk[k]) {
            throw bad_parameter(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "msk");
        }
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
    for (; it != m_intr_labels.end() && *it < l; it++) { }

    if (it == m_intr_labels.end()) {
        m_intr_labels.push_back(l);
        return;
    }
    if (*it == l) return;

    m_intr_labels.insert(it, l);
}

template<size_t N>
void label_set<N>::set_eval_msk(const mask<N> &emsk) {

    static const char *method = "set_eval_msk(size_t)";

    m_eval_size = 0;
    for (register size_t i = 0; i < N; i++) {
        m_eval_msk[i] = emsk[i];
        if (m_eval_msk[i]) {
            m_eval_size++;

#ifdef LIBTENSOR_DEBUG
            if (! m_msk[i]) {
                throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "Evaluation mask.");
            }
#endif
        }
    }
}

template<size_t N>
void label_set<N>::permute(const permutation<N> &p) {

    m_msk.permute(p);
    m_bidims.permute(p);
    p.apply(m_type);
    m_eval_msk.permute(p);
}

template<size_t N>
void label_set<N>::match_blk_labels() {

    sequence<N, size_t> types(m_type);
    sequence<N, label_group*> blk_labels(m_blk_labels);

    for (size_t i = 0; i < N; i++) {
        m_type[i] = (size_t) -1; m_blk_labels[i] = 0;
    }

    size_t cur_type = 0;
    for (register size_t i = 0; i < N; i++) {

        size_t itype = types[i];
        if (itype == (size_t) -1) continue;
        if (blk_labels[itype] == 0) continue;

        m_type[i] = cur_type;
        label_group *lli = m_blk_labels[cur_type] = blk_labels[itype];
        blk_labels[itype] = 0;

        for (size_t j = i + 1; j < N; j++) {
            size_t jtype = types[j];
            if (jtype == (size_t) -1) continue;

            if (itype == jtype) {
                m_type[j] = cur_type;
                continue;
            }

            if (blk_labels[jtype] == 0) continue;
            if (lli->size() != blk_labels[jtype]->size()) continue;

            size_t k = 0;
            for (; k < lli->size(); k++) {
                if (lli->at(k) != blk_labels[jtype]->at(k)) break;
            }
            if (k != lli->size()) continue;

            delete blk_labels[jtype];
            blk_labels[jtype] = 0;
            m_type[j] = cur_type;
        }

        cur_type++;
    }
}

template<size_t N>
void label_set<N>::clear_eval() {

    m_eval_size = 0;
    for (register size_t i = 0; i < N; i++) {
        m_eval_msk[i] = false;
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
    if (m_blk_labels[type]->size() <= blk) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "blk");
    }
#endif

    return m_blk_labels[type]->at(blk);
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

    // No evaluation mask means check the intrinsic labels for zero label
    if (m_eval_size == 0) {
        return (m_intr_labels[0] == 0);
    }

    // If the set of intrinsic labels comprises all labels, all blocks are true
    if (m_intr_labels.size() == m_pt.nlabels()) return true;

    label_group lg(m_eval_size + 1, m_pt.invalid());
    // Assign the block labels to the elements of lg
    // (last is the intrinsic label)
    for (register size_t i = 0, j = 0; i < N; i++) {
        if (! m_eval_msk[i]) continue;

        label_group &labels = *(m_blk_labels[m_type[i]]);
        lg[j] = labels[idx[i]];

        // If one of the block labels is invalid, the block is allowed
        if (! m_pt.is_valid(lg[j])) return true;
        j++;
    }

    // Loop over all intrinsic labels
    for (iterator it = m_intr_labels.begin();
            it != m_intr_labels.end(); it++) {

        lg[m_eval_size] = *it;
        if (m_pt.is_in_product(lg, 0)) return true;
    }

    return false;
}

} // namespace libtensor

#endif // LIBTENSOR_LABEL_SET_H

