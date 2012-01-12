#ifndef LIBTENSOR_SE_LABEL_H
#define LIBTENSOR_SE_LABEL_H

#include "../defs.h"
#include "../core/dimensions.h"
#include "../core/mask.h"
#include "../core/symmetry_element_i.h"
#include "bad_symmetry.h"
#include "label/block_labeling.h"
#include "label/evaluation_rule.h"
#include "label/product_table_container.h"
#include "label/product_table_i.h"
#include "label/transfer_rule.h"

namespace libtensor {

/** \brief Symmetry element for labels assigned to %tensor blocks
    \tparam N Symmetry cardinality (%tensor order).
    \tparam T Tensor element type.

    This %symmetry element establishes a set of allowed blocks on the basis of
    block labels along each dimension, an evaluation rule for block labels and
    intrinsic labels, and a product table.

    The labeling of the blocks in each dimension is setup via the class
    block_labeling.

    Allowed blocks are determined as follows from the evaluation rule:
    - All blocks are forbidden, if the rule setup of the evaluation rule is
      empty
    - A block is allowed, if it is allowed by any of the products in the rule
      setup
    - A block is allowed by a product, if it is allowed by all basic rules in
      this product
    - A block is allowed by a basic rule, if any product of labels specified
      by the rule contains label 0
    - A product of labels is formed from the block labels and one intrinsic
      label using the evaluation order parameter of the basic rule.
    - If this is not possible since the evaluation order has zero length or
      no intrinsic labels are set, but the evaluation order comprises the
      intrinsic label index, all blocks are discarded by this rule.
    - If any product contains an invalid label the block is allowed.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class se_label : public symmetry_element_i<N, T> {
public:
    static const char *k_clazz; //!< Class name
    static const char *k_sym_type; //!< Symmetry type

    typedef product_table_i::label_t label_t;
    typedef product_table_i::label_group label_group;

private:
    block_labeling<N> m_blk_labels; //!< Block index labels
    const product_table_i &m_pt; //!< Product table

    evaluation_rule m_rule; //!< Label evaluation rule

public:
    //!	\name Construction and destruction
    //@{
    /** \brief Initializes the %symmetry element
        \param bidims Block %index dimensions.
     **/
    se_label(const dimensions<N> &bidims, const std::string &id);

    /**	\brief Copy constructor
     **/
    se_label(const se_label<N, T> &elem);

    /**	\brief Virtual destructor
     **/
    virtual ~se_label();
    //@}

    //!	\name Manipulating functions
    //@{

    /** \brief Obtain the block index labeling
     **/
    block_labeling<N> &get_labeling() { return m_blk_labels; }

    /** \brief Obtain the block index labeling (const version)
     **/
    const block_labeling<N> &get_labeling() const { return m_blk_labels; }

    /** \brief Set the evaluation rule to consist of only one basic rule.
        \param intr Intrinsic or target label.
        \param p Permutation of indexes.
        \param pos Position of intrinsic labels.

        Replaces any existing rule with the basic rule given by the parameters.
     **/
    void set_rule(const label_t &intr,
            const permutation<N> &p = permutation<N>(),
            size_t pos = N) {

        set_rule(label_group(1, intr), p, pos);
    }

    /** \brief Set the evaluation rule to consist of only one basic rule.
        \param intr Intrinsic or target labels.
        \param p Permutation of indexes.
        \param pos Position of intrinsic labels.

        Replaces any existing rule with a basic rule.
     **/
    void set_rule(const label_group &intr,
            const permutation<N> &p = permutation<N>(),
            size_t pos = N);

    /** \brief Set the evaluation rule to a composite rule.
        \param rule Composite evaluation rule.

        The function checks the given rule on validity and replaces any
        previously given rule.
     **/
    void set_rule(const evaluation_rule &rule);
    //@}

    //! \name Access functions
    //@{

    /** \brief Return the current evaluation rule.
     **/
    const evaluation_rule &get_rule() const { return m_rule; }

    /** \brief Returns the id of the product table
     **/
    const std::string &get_table_id() const {
        return m_pt.get_id();
    }

    //@}

    //!	\name Implementation of symmetry_element_i<N, T>
    //@{

    /**	\copydoc symmetry_element_i<N, T>::get_type()
     **/
    virtual const char *get_type() const {
        return k_sym_type;
    }

    /**	\copydoc symmetry_element_i<N, T>::clone()
     **/
    virtual symmetry_element_i<N, T> *clone() const {
        return new se_label<N, T>(*this);
    }

    /**	\copydoc symmetry_element_i<N, T>::permute
     **/
    void permute(const permutation<N> &perm);

    /**	\copydoc symmetry_element_i<N, T>::is_valid_bis
     **/
    virtual bool is_valid_bis(const block_index_space<N> &bis) const;

    /**	\copydoc symmetry_element_i<N, T>::is_allowed
     **/
    virtual bool is_allowed(const index<N> &idx) const;

    /**	\copydoc symmetry_element_i<N, T>::apply(index<N>&)
     **/
    virtual void apply(index<N> &idx) const { }

    /**	\copydoc symmetry_element_i<N, T>::apply(
			index<N>&, transf<N, T>&)
     **/
    virtual void apply(index<N> &idx, transf<N, T> &tr) const { }
    //@}

};

template<size_t N, typename T>
const char *se_label<N, T>::k_clazz = "se_label<N, T>";

template<size_t N, typename T>
const char *se_label<N, T>::k_sym_type = "se_label";

template<size_t N, typename T>
se_label<N, T>::se_label(const dimensions<N> &bidims, const std::string &id) :
    m_blk_labels(bidims),
    m_pt(product_table_container::get_instance().req_const_table(id)) {

}

template<size_t N, typename T>
se_label<N, T>::se_label(const se_label<N, T> &el) :
    m_blk_labels(el.m_blk_labels), m_rule(el.m_rule),
    m_pt(product_table_container::get_instance().req_const_table(
            el.m_pt.get_id())) {

}

template<size_t N, typename T>
se_label<N, T>::~se_label() {

    product_table_container::get_instance().ret_table(m_pt.get_id());
}

template<size_t N, typename T>
void se_label<N, T>::set_rule(const label_group &intr,
        const permutation<N> &p, size_t pos) {

    static const char *method =
            "set_rule(const label_group &, const permutation<N> &, size_t)";

#ifdef LIBTENSOR_DEBUG
    if (pos > N + 1) {
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "pos");
    }
#endif

    // Check the intrinsic labels for duplicates and valid labels
    std::map<label_t, bool> lmap;
    for (size_t i = 0; i < intr.size(); i++) {
        if (! m_pt.is_valid(intr[i]))
            throw bad_parameter(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "intr");

        lmap[intr[i]] = true;
    }

    // Now start updating the evaluation rule by clearing the old
    m_rule.clear_all();

    // This is a trivial rule => simplify it
    if (lmap.size() == m_pt.nlabels()) {

        // All blocks allowed: one element, only intrinsic in evaluation order
        evaluation_rule::rule_id id =
                m_rule.add_rule(evaluation_rule::label_group(1, 0),
                        std::vector<size_t>(1, evaluation_rule::k_intrinsic));
        m_rule.add_product(id);

        return;
    }
    else if (lmap.size() == 0) {
        // No blocks allowed: empty rule
        return;
    }

    // Form the evaluation order as required by evaluation_rule
    sequence<N, size_t> tmp_order;
    for (size_t i = 0; i < N; i++) tmp_order[i] = i;
    p.apply(tmp_order);

    // Create an ordered label_group
    label_group new_intr;
    for (std::map<label_t, bool>::iterator it = lmap.begin();
            it != lmap.end(); it++) {
        new_intr.push_back(it->first);
    }

    // Create the evaluation order as required by evaluation_rule
    std::vector<size_t> order(N + 1);
    for (size_t i = 0; i < pos; i++) order[i] = tmp_order[i];
    order[pos] = evaluation_rule::k_intrinsic;
    for (size_t i = pos; i < N; i++) order[i + 1] = tmp_order[i];

    evaluation_rule::rule_id id = m_rule.add_rule(new_intr, order);
    m_rule.add_product(id);
}

template<size_t N, typename T>
void se_label<N, T>::set_rule(const evaluation_rule &rule) {

    static const char *method = "set_rule(const evaluation_rule &)";

    typedef evaluation_rule::rule_id rule_id;
    typedef std::map<rule_id, rule_id> rule_id_map;

    transfer_rule(rule, N, m_pt.get_id()).perform(m_rule);

    m_rule = rule;
}


template<size_t N, typename T>
void se_label<N, T>::permute(const permutation<N> &p) {

    m_blk_labels.permute(p);

    sequence<N, size_t> dims;
    for(size_t i = 0; i < N; i++) dims[i] = i;
    p.apply(dims);

    for (evaluation_rule::rule_iterator it = m_rule.begin();
            it != m_rule.end(); it++) {

        evaluation_rule::basic_rule &br =
                m_rule.get_rule(m_rule.get_rule_id(it));
        for (size_t i = 0; i < br.order.size(); i++) {
            if (br.order[i] == evaluation_rule::k_intrinsic) continue;

            br.order[i] = dims[br.order[i]];
        }
    }
}

template<size_t N, typename T>
bool se_label<N, T>::is_valid_bis(const block_index_space<N> &bis) const {

    const dimensions<N> &bidims = m_blk_labels.get_block_index_dims();
    return bidims.equals(bis.get_block_index_dims());
}

template<size_t N, typename T>
bool se_label<N, T>::is_allowed(const index<N> &idx) const {

    static const char *method = "is_allowed(const index<N> &)";

#ifdef LIBTENSOR_DEBUG
    const dimensions<N> &bidims = m_blk_labels.get_block_index_dims();
    // Test, if index is valid block index
    for (size_t i = 0; i < N; i++) {
        if (idx[i] >= bidims[i]) {
            throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "idx.");
        }
    }
#endif

    label_group blk(N);
    for (size_t i = 0; i < N; i++) {
        size_t dim_type = m_blk_labels.get_dim_type(i);
        blk[i] = m_blk_labels.get_label(dim_type, idx[i]);
    }

    std::map<evaluation_rule::rule_id, bool> allowed;
    for (evaluation_rule::rule_iterator it = m_rule.begin();
            it != m_rule.end(); it++) {

        const evaluation_rule::basic_rule &br = m_rule.get_rule(it);
        evaluation_rule::rule_id rid = m_rule.get_rule_id(it);

        if (br.order.size() == 0) { allowed[rid] = false; continue; }

        label_group lg(br.order.size());
        size_t pos = (size_t) -1;

        bool has_invalid = false;
        for (size_t i = 0; i < br.order.size(); i++) {
            if (br.order[i] == evaluation_rule::k_intrinsic) {
                pos = i; continue;
            }

            lg[i] = blk[br.order[i]];
            if (! m_pt.is_valid(lg[i])) { has_invalid = true; break; }
        }

        if (has_invalid) { allowed[rid] = true; continue; }

        bool cur = false;
        if (pos == (size_t) -1) {
            cur = m_pt.is_in_product(lg, 0);
        }
        else {
            for (size_t k = 0; k < br.intr.size(); k++) {
                if (! m_pt.is_valid(br.intr[k])) { cur = true; break; }
                lg[pos] = br.intr[k];
                cur = cur || m_pt.is_in_product(lg, 0);
            }
        }
        allowed[rid] = cur;
    }

    // loop over sums in the evaluation rule
    for (size_t i = 0; i < m_rule.get_n_products(); i++) {

        bool is_allowed = true;
        for (evaluation_rule::product_iterator it = m_rule.begin(i);
                it != m_rule.end(i); it++) {

            is_allowed = is_allowed && allowed[m_rule.get_rule_id(it)];
        }

        if (is_allowed) return true;
    }

    return false;
}

} // namespace libtensor

#endif // LIBTENSOR_SE_LABEL_H

