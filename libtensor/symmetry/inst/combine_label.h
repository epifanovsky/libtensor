#ifndef LIBTENSOR_COMBINE_LABEL_H
#define LIBTENSOR_COMBINE_LABEL_H

#include <libtensor/exception.h>
#include "../se_label.h"

namespace libtensor {

/** \brief Combine multiple se_label objects

    Combines the rules of multiple se_label objects. All objects have to have
    the same product table associated to them, and the same block labels. The
    rules of the objects are combined as if they were products.

 **/
template<size_t N, typename T>
class combine_label {
public:
    static const char *k_clazz;

private:
    std::string m_table_id;
    block_labeling<N> m_blk_labels;
    evaluation_rule<N> m_rule;

public:
    combine_label(const se_label<N, T> &el);

    void add(const se_label<N, T> &el) throw(bad_parameter);

    const std::string &get_table_id() const { return m_table_id; }
    const block_labeling<N> &get_labeling() const { return m_blk_labels; }
    const evaluation_rule<N> &get_rule() const { return m_rule; }

private:
    combine_label(const combine_label<N, T> &el);
    combine_label<N, T> &operator=(const combine_label<N, T> &el);
};


} // namespace libtensor

#endif // LIBTENSOR_COMBINE_LABEL_H
