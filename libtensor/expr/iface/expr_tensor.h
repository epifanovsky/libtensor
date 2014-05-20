#ifndef LIBTENSOR_EXPR_EXPR_TENSOR_H
#define LIBTENSOR_EXPR_EXPR_TENSOR_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/expr/dag/node_null.h>
#include "any_tensor.h"
#include "labeled_lhs_rhs.h"

namespace libtensor {
namespace expr {


/** \brief Tensor-like object that stores a tensor expression

    \ingroup libtensor_iface
 **/
template<size_t N, typename T = double>
class expr_tensor :
    public any_tensor<N, T>,
    public expr_lhs<N, T>,
    public noncopyable {

private:
    expr_tree *m_expr; //!< Expression

public:
    /** \brief Constructs an empty object
     **/
    expr_tensor() :
        any_tensor<N, T>(*this) {

        m_expr = new expr_tree(expr::node_null(N));
    }

    /** \brief Virtual destructor
     **/
    virtual ~expr_tensor();

    /** \brief Attaches a letter label to expr_tensor
     **/
    labeled_lhs_rhs<N, T> operator()(const label<N> &lab) {
        return labeled_lhs_rhs<N, T>(*this, lab, expr_rhs<N, T>(*m_expr, lab));
    }

    /** \brief Saves the given expression in this container
     **/
    virtual void assign(const expr_rhs<N, T> &rhs, const label<N> &lab);

};


template<size_t N, typename T>
expr_tensor<N, T>::~expr_tensor() {

    delete m_expr;
}


template<size_t N, typename T>
void expr_tensor<N, T>::assign(const expr_rhs<N, T> &rhs,
    const label<N> &lab) {

    if(m_expr) delete m_expr;

    permutation<N> px = lab.permutation_of(rhs.get_label());
    if(px.is_identity()) {
        m_expr = new expr_tree(rhs.get_expr());
    } else {
        std::vector<size_t> perm(N);
        for(size_t i = 0; i < N; i++) perm[i] = px[i];
        node_transform<T> nt(perm, scalar_transf<T>());
        expr_tree e(nt);
        expr_tree::node_id_t id = e.get_root();
        e.add(id, rhs.get_expr());
        m_expr = new expr_tree(e);
    }
}


} // namespace expr
} // namespace libtensor


namespace libtensor {

using expr::expr_tensor;

} // namespace libtensor

#endif // LIBTENSOR_EXPR_EXPR_TENSOR_H
