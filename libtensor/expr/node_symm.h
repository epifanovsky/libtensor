#ifndef LIBTENSOR_EXPR_NODE_SYMM_H
#define LIBTENSOR_EXPR_NODE_SYMM_H

#include <libtensor/core/scalar_transf.h>
#include "node_symm_base.h"

namespace libtensor {
namespace expr {


/** \brief Tensor symmetrization node of the expression tree

    \ingroup libtensor_expr
 **/
template<typename T>
class node_symm : public node_symm_base {
private:
    scalar_transf<T> m_trp, m_trc; //!< Pair and cyclic transform

public:
    /** \brief Creates an symmetrization node
     **/
    node_symm(const node &arg, const std::vector<size_t> &sym, size_t n,
            const scalar_transf<T> &trp, const scalar_transf<T> &trc) :
        node_symm_base(arg, sym, n), m_trp(trp), m_trc(trc)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_symm() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node_symm<T> *clone() const {
        return new node_symm<T>(*this);
    }

    const scalar_transf<T> &get_pair_tr() const { return m_trp; }

    const scalar_transf<T> &get_cyclic_tr() const { return m_trc; }
};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_SYMM_BASE_H
