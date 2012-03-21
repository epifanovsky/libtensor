#ifndef LIBTENSOR_BASIC_RULE_H
#define LIBTENSOR_BASIC_RULE_H

#include "../../defs.h"
#include "../../core/sequence.h"
#include "product_table_i.h"

namespace libtensor {

/** \brief Basic rule to determine allowed blocks in a block %tensor.

    The basic rule is the most elementary rule to determine allowed blocks
    in a N-dimensional block %tensor by its block labels. It consists of a
    sequence of N numbers (one for each dimension of the block %tensor) and
    a set of target labels.

    For details on the evaluation of a basic rule refer to \sa se_label.

    \ingroup libtensor_symmetry
 **/
template<size_t N>
class basic_rule : public sequence<N, size_t> {
public:
    typedef product_table_i::label_t label_t;
    typedef product_table_i::label_set_t label_set_t;

private:
    label_set_t m_targets; //!< Target labels

public:
    /** \brief Default constructor
        \param lt Target labels
     **/
    basic_rule(const label_set_t &lt = label_set_t()) :
        sequence<N, size_t>(0), m_targets(lt) { }

    /** \brief Add another label to the target labels
        \param lt New label.
     **/
    void set_target(label_t lt) {
        m_targets.insert(lt);
    }

    /** \brief Delete target labels
     **/
    void reset_target() {
        m_targets.clear();
    }

    /** \brief Get the target labels
     **/
    const label_set_t &get_target() const {
        return m_targets;
    }

};

template<size_t N>
bool operator==(const basic_rule<N> &br1, const basic_rule<N> &br2);

template<size_t N>
bool operator!=(const basic_rule<N> &br1, const basic_rule<N> &br2) {
    return !(br1 == br2);
}

} // namespace libtensor


#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

namespace libtensor {

    extern template class basic_rule<1>;
    extern template class basic_rule<2>;
    extern template class basic_rule<3>;
    extern template class basic_rule<4>;
    extern template class basic_rule<5>;
    extern template class basic_rule<6>;

    extern template
    bool operator==(const basic_rule<1> &, const basic_rule<1> &);
    extern template
    bool operator==(const basic_rule<2> &, const basic_rule<2> &);
    extern template
    bool operator==(const basic_rule<3> &, const basic_rule<3> &);
    extern template
    bool operator==(const basic_rule<4> &, const basic_rule<4> &);
    extern template
    bool operator==(const basic_rule<5> &, const basic_rule<5> &);
    extern template
    bool operator==(const basic_rule<6> &, const basic_rule<6> &);

} // namespace libtensor

#else // LIBTENSOR_INSTANTIATE_TEMPLATES
#include "inst/basic_rule_impl.h"
#endif // LIBTENSOR_INSTANTIATE_TEMPLATES

#endif // LIBTENSOR_BASIC_RULE_H
